"""Build one pooled Gaussian-process (GP) surrogate of corridor OD delays.

A surrogate is a fast statistical approximation of an expensive calculation.
We run the transport model for selected conditions once, then use the learned
relationship to evaluate many future plans without repeating road assignment.

Student-facing idea
-------------------
1. A case combines one infrastructure configuration with the registered physical
   inputs: by default demand growth, PT preference, road freight growth and
   e-bike share. Fixed inputs do not add training dimensions.
   It describes one operating condition; annual future trajectories later
   query many such conditions as demand and preferences change over time.
2. A Latin-hypercube design spreads cases across the configured input ranges.
   The default is 100 training simulations TOTAL: 25 in each of the four
   configurations (baseline, stations, tunnel, or both). Labels are balanced
   across the fixed design.
3. The full model supplies the training answers. Its coupled MSA (successive
   averaging) repeatedly updates road assignment, travel times and mode choice.
   Future builds check the road relative gap every four iterations, starting
   at iteration 8, and stop at a gap of 2% or less, with a cap of 64 iterations.
   The road gap measures the remaining benefit of switching assigned traffic
   to current shortest routes. Small OD flows are retained (cutoff zero).
   Reaching the cap can leave a case unconverged; the diagnostics report this
   as a warning while preparation continues.
4. One GP learns across all stages. Continuous inputs are scaled to [0, 1];
   one-hot flags such as [1, 0, 0, 0] identify the infrastructure configuration
   without treating stage numbers as a numerical distance.
5. The learned output is one directional origin-destination (OD) matrix:
   additional car minutes from local congestion between corridor zones and
   boundary gates. All matrix entries are retained directly. There is no PCA,
   separately fitted link output, or separately fitted scalar output.
6. Predicted delays update the car skims. The existing model then recalculates
   mode choice and transport indicators. Appraisal uses the same matched
   OD/submode rule of half, annualisation and cost formulas as the full model.
7. Validation cases are excluded from fitting. By default, 20%
   of the training budget adds 20 simulations: five new physical conditions,
   each evaluated at all four configurations. These paired cases check OD errors and
   baseline/project benefit differences, including individual cost components.
8. Optional fill-in sampling scores GP uncertainty in small batches and adds
   MSA cases where uncertainty is largest. It refits until the configured
   validation targets are reached or the extra-simulation budget is used.
   It is disabled by default; when enabled, validation also guides stopping.

The build is resumable: every expensive simulation is cached with its physical
inputs, ordered OD labels and solver settings. Compatible cached cases are
reused after an interruption. Source-file hashes are retained as provenance;
editing explanations or appraisal settings does not invalidate transport runs.

Student adaptation map
----------------------
Start with the corridor and uncertainty definitions in ``parameters.py`` and
the infrastructure packages in ``stages.py``. Notebook 03 displays the training
bounds derived from those uncertainties before fitting the GP. Bounded inputs
use their full support; unbounded sides retain at least 99.8% marginal coverage
over the planning horizon. If adding or changing an input,
update its explicit native-model mapping in ``additional.uncertainty`` too: declaring
a new input alone does not make it affect the transport model.

Rebuild after changing transport data, the corridor, physical stages/section
coverage, mode-choice or assignment equations, or the surrogate input domain.

Appraisal-only changes to unit values, discounting, peak/day conversion,
crowding valuation, external passenger counts or investment/operating costs
normally require rerunning appraisal, without retraining the transport GP.
Changing plan triggers or activation dates requires rerunning the plans;
the same fitted stages remain usable within their validated physical domain.
See ``surrogate_model/README.md`` for the workflow and saved diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess
import time
import warnings
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import qmc

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import parameters as p
from additional import uncertainty as uc
import stages as stages_module
import transport_model_interface as tmi
from surrogate_model.transport_surrogate import (current_model_inputs, portable_model_inputs, stable_fingerprint,
    reference_equations, _input_changes)

INPUT_NAMES = list(uc.TRANSPORT_SURROGATE_INPUTS)
DEFAULT_WORKERS = 4
OD_ERROR_LIMITS = {"weighted_mae_min": 0.1, "weighted_rmse_min": 0.2}
_WORKER_BACKEND = None


def _bounds():
    return np.asarray([(uc.TRANSPORT_SURROGATE_INPUTS[name]["minimum"],
                        uc.TRANSPORT_SURROGATE_INPUTS[name]["maximum"])
                       for name in INPUT_NAMES], dtype=float).reshape(-1, 2)


def _scale_unit_design(unit):
    bounds = _bounds()
    return np.asarray(unit) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def _unit_inputs(records):
    bounds = _bounds()
    values = np.asarray([[row["inputs"][name] for name in INPUT_NAMES] for row in records])
    return (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _row_id(stage, x):
    return stable_fingerprint({"stage": int(stage), **dict(zip(INPUT_NAMES, map(float, x)))})


def _cache_path(cache_dir, stage, x):
    return Path(cache_dir) / f"stage_{int(stage)}" / f"{_row_id(stage, x)}.joblib"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path, value):
    temporary = Path(path).with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_design(stage_ids, n_train=100, validation_fraction=.2, seed=42):
    """Continuous LHS, balanced stage labels, independent paired-stage holdout.

    Validation fraction is relative to training count, rounded up to complete
    configuration sets: 100 × .20 = five futures × four states = 20 extra runs.
    """
    stage_ids = tuple(map(int, stage_ids))
    if len(stage_ids) < 2 or len(set(stage_ids)) != len(stage_ids) or 0 not in stage_ids:
        raise ValueError("Use unique stage IDs including baseline 0 and at least one project stage.")
    if int(n_train) != n_train or n_train < 2 * len(stage_ids):
        raise ValueError("n_train is the total budget; every stage needs at least two rows.")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction <= 1:
        raise ValueError("validation_fraction must be in (0, 1].")
    n_train = int(n_train)
    if not INPUT_NAMES:
        warnings.warn("All transport inputs are fixed: only the stage configurations need simulations; "
                      "holdout checks reuse those same conditions.", UserWarning)
        n_train, validation_fraction = 2 * len(stage_ids), .5
    train = _scale_unit_design(qmc.LatinHypercube(len(INPUT_NAMES), seed=seed).random(n_train))
    allocation = np.resize(np.asarray(stage_ids), n_train)
    np.random.default_rng(seed + 17).shuffle(allocation)
    n_pairs = max(1, math.ceil(n_train * validation_fraction / len(stage_ids)))
    test = _scale_unit_design(qmc.LatinHypercube(len(INPUT_NAMES), seed=seed + 1).random(n_pairs))
    rows = [{"id": f"train_{i:03d}", "role": "train", "pair_id": None,
             "stage": int(stage), "inputs": dict(zip(INPUT_NAMES, map(float, x)))}
            for i, (stage, x) in enumerate(zip(allocation, train))]
    rows += [{"id": f"test_{i:03d}_stage{stage}", "role": "test", "pair_id": f"test_{i:03d}",
              "stage": int(stage), "inputs": dict(zip(INPUT_NAMES, map(float, x)))}
             for i, x in enumerate(test) for stage in stage_ids]
    identities = [_row_id(row["stage"], [row["inputs"][name] for name in INPUT_NAMES]) for row in rows]
    if INPUT_NAMES and len(set(identities)) != len(identities):
        raise ValueError("Training/validation contain a duplicate physical case.")
    return rows


def _source_inventory():
    """Record source bytes for provenance, independently of cache compatibility."""
    paths = sorted(CODE_DIR.rglob("*.py"))
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _physical_inputs_on_disk():
    # A fresh import also detects edits to parameters/stages while the notebook
    # kernel retains the old modules. This does not run the transport model.
    script = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); "
        "import parameters as p, stages; "
        "from surrogate_model.transport_surrogate import current_model_inputs; "
        "from surrogate_model.transport_surrogate import reference_equations; "
        "print(json.dumps(dict(current_model_inputs(stages.get_stages(p.NOMINAL_PARAMS)), "
        "reference_equations=reference_equations())))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script, str(CODE_DIR)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _reference_cache(directory, contract):
    """Resume the saved design when its physical model and solver agree."""
    design_path = directory / "design.json"
    if design_path.is_file():
        saved = json.loads(design_path.read_text(encoding="utf-8"))
        previous = saved.get("cache_contract", {})
        # Older rows include a hash of every Python file. Preserve their
        # original metadata, but compare only the inputs used by the solver.
        physical = {key: value for key, value in previous.items() if key != "source_fingerprint"}
        if "reference_equations" not in physical and saved.get("reference_equations"):
            physical["reference_equations"] = stable_fingerprint(saved["reference_equations"])
        if physical == contract or saved.get("compatible_contract") == contract:
            return previous, saved.get("reference_source_sha256", saved.get("source_sha256", {}))
    return contract, None


def _worker_backend(stage_ids, contract):
    global _WORKER_BACKEND
    identity = stable_fingerprint({"contract": contract, "stages": list(stage_ids)})
    if _WORKER_BACKEND is None or _WORKER_BACKEND["identity"] != identity:
        context = tmi.load_transport_context(PROJECT_ROOT)
        specs = stages_module.get_stages(p.NOMINAL_PARAMS)
        base = tmi.build_corridor_context(context, corridor_municipalities=p.CORRIDOR_MUNICIPALITIES,
                                          name="Surrogate training corridor")
        corridors = {stage: tmi.apply_road_capacity_stage(base, specs[stage])[0] for stage in stage_ids}
        # Diagnostic weights cover all nominal passengers whose gate mapping
        # uses the local skim. Do not apply the road-assignment pass-through
        # fraction: it would underweight gate-to-gate prediction errors.
        weights, _ = tmi.collapse_od_to_gates(context.baseline_od, context, base,
                                            passthrough_fraction=1.0)
        _WORKER_BACKEND = dict(identity=identity, context=context, stage_specs=specs,
                               corridors=corridors, nominal_weights=weights, reachable={})
    return _WORKER_BACKEND


def _load_cached(path, record, contract):
    if not path.exists():
        return None
    saved = joblib.load(path)
    if (saved.get("stage") != record["stage"] or saved.get("inputs") != record["inputs"]
            or saved.get("cache_contract") != contract):
        raise ValueError(f"Incompatible cached reference: {path}; use a new cache namespace.")
    labels = list(saved.get("labels", []))
    delay, weights = np.asarray(saved.get("delay", [])), np.asarray(saved.get("weights", []))
    reachable = np.asarray(saved.get("reachable", []))
    if (not labels or len(set(labels)) != len(labels) or delay.shape != (len(labels), len(labels))
            or not np.isfinite(delay).all() or (delay < 0).any()
            or weights.shape != delay.shape or not np.isfinite(weights).all() or (weights < 0).any()
            or reachable.shape != delay.shape or np.any(np.diag(delay) != 0)):
        raise ValueError(f"Invalid cached OD layout/values: {path}")
    return saved


def _evaluate_one(record, *, stage_ids, contract, cache_dir):
    values, stage = record["inputs"], int(record["stage"])
    path = _cache_path(cache_dir, stage, [values[name] for name in INPUT_NAMES])
    if _load_cached(path, record, contract) is not None:
        return str(path)
    backend = _worker_backend(stage_ids, contract)
    corridor, context = backend["corridors"][stage], backend["context"]
    started = time.perf_counter()
    print(f"Simulating {record['id']} (stage {stage})", flush=True)
    native_values = uc.native_transport_inputs(values)
    mode = tmi.run_transport_mode_choice(
        context, stage=stage, stage_specs=backend["stage_specs"], corridor_zone_ids=corridor.zone_ids,
        pt_asc_shift=native_values["pt_asc_shift"], bike_asc_shift=native_values["bike_asc_shift"],
        ebike_share=native_values["ebike_share"],
        demand_multiplier=1.0)
    assignment = tmi.run_coupled_corridor_assignment(
        context, corridor, mode, demand_multiplier=native_values["passenger_demand_multiplier"],
        background_multiplier=native_values["road_freight_multiplier"], **contract["solver_settings"])
    delay = assignment.local_od_delay_min
    if not isinstance(delay, pd.DataFrame) or not delay.index.equals(delay.columns):
        raise ValueError("Native solver did not return aligned final OD delays.")
    labels = list(map(str, delay.index))
    if stage not in backend["reachable"]:
        free = tmi._bpr_link_times(corridor.edges, np.zeros(len(corridor.edges)))
        table = tmi._corridor_shortest_time_table(corridor.edges, corridor.zone_node_map, free)
        if list(map(str, table.index)) != labels:
            raise ValueError("Free-flow and delay OD label orders differ.")
        reachable = np.isfinite(table.to_numpy())
        np.fill_diagonal(reachable, False)
        backend["reachable"][stage] = reachable
    weights = backend["nominal_weights"].copy()
    weights.index, weights.columns = weights.index.astype(str), weights.columns.astype(str)
    weights = weights.reindex(index=labels, columns=labels, fill_value=0.0).to_numpy(dtype=np.float32)
    weights *= backend["reachable"][stage]
    saved = {"stage": stage, "inputs": record["inputs"], "cache_contract": contract,
             "labels": labels, "delay": delay.to_numpy(dtype=np.float32),
             "reachable": backend["reachable"][stage], "weights": weights,
             "solver_diagnostics": dict(assignment.diagnostics),
             "runtime_seconds": time.perf_counter() - started}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.joblib")
    joblib.dump(saved, temporary, compress=3)
    temporary.replace(path)
    print(f"Saved {record['id']}: {saved['runtime_seconds']:.1f}s, "
          f"{assignment.diagnostics['iterations']} iterations, "
          f"{assignment.diagnostics.get('termination_reason')}", flush=True)
    return str(path)


def _evaluate_many(records, *, stage_ids, contract, cache_dir, n_jobs, reuse_cached_only=False):
    # Reuse in the parent; workers begin with actual solves, including their
    # persistent native context in joblib's initial memory baseline.
    paths, references, missing = [], [], []
    for index, record in enumerate(records):
        path = _cache_path(cache_dir, record["stage"], [record["inputs"][name] for name in INPUT_NAMES])
        paths.append(path)
        saved = _load_cached(path, record, contract)
        references.append(saved)
        if saved is None:
            missing.append(index)
    if missing and reuse_cached_only:
        raise RuntimeError(f"Cached-only recovery needs {len(missing)} missing references; no simulations started.")
    print(f"References: {len(records)-len(missing)} cached, {len(missing)} to simulate.", flush=True)
    kwargs = dict(stage_ids=stage_ids, contract=contract, cache_dir=cache_dir)
    if n_jobs == 1:
        for index in missing:
            _evaluate_one(records[index], **kwargs)
    elif missing:
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            joblib.Parallel(n_jobs=n_jobs, verbose=10)(
                joblib.delayed(_evaluate_one)(records[index], **kwargs) for index in missing)
    for index in missing:
        references[index] = _load_cached(paths[index], records[index], contract)
    return paths, references


def _od_validation(model, records, truth, weights, reachable):
    weights = np.asarray(weights, dtype=float).reshape(-1) * np.asarray(reachable).reshape(-1)
    if weights.sum() <= 0:
        raise ValueError("No positive fixed nominal OD validation weights.")
    weights /= weights.sum()
    X = _unit_inputs(records)
    mean, latent = np.empty_like(truth), np.empty_like(truth)
    cases = []
    for index, record in enumerate(records):
        raw, std = model.predict(X[index:index+1], [record["stage"]], return_std=True,
                                 clip_nonnegative=False)
        raw, std = raw[0], std[0]
        raw[~reachable.reshape(-1)] = 0.0
        std[~reachable.reshape(-1)] = 0.0
        prediction = np.maximum(raw, 0.0)
        mean[index], latent[index] = prediction, std
        error = prediction - truth[index]
        truth_rms = float(np.sqrt(truth[index]**2 @ weights))
        rmse = float(np.sqrt(error**2 @ weights))
        cases.append({"id": record["id"], "stage": record["stage"],
                      "weighted_mae_min": float(np.abs(error) @ weights),
                      "weighted_mse_min2": float(error**2 @ weights),
                      "weighted_rmse_min": rmse,
                      "weighted_truth_rms_min": truth_rms,
                      "scaled_error": rmse / max(truth_rms, 0.05),
                      "weighted_bias_min": float(error @ weights),
                      "weighted_95pct_latent_coverage": float(
                          (np.abs(raw-truth[index]) <= 1.959963984540054*std+1e-12) @ weights),
                      "negative_prediction_fraction_weighted": float((raw < 0) @ weights)})
    def summarize(rows):
        return {"n_cases": len(rows),
                "weighted_mae_min": float(np.mean([row["weighted_mae_min"] for row in rows])),
                "weighted_rmse_min": float(np.sqrt(np.mean([row["weighted_mse_min2"] for row in rows]))),
                "weighted_bias_min": float(np.mean([row["weighted_bias_min"] for row in rows])),
                "maximum_case_mae_min": max(row["weighted_mae_min"] for row in rows),
                "weighted_95pct_latent_coverage": float(np.mean([row["weighted_95pct_latent_coverage"] for row in rows]))}
    report = summarize(cases)
    overall_truth_rms = float(np.sqrt(np.mean([row["weighted_truth_rms_min"]**2 for row in cases])))
    report["weighted_nrmse"] = report["weighted_rmse_min"] / max(overall_truth_rms, 0.05)
    report["maximum_case_scaled_error"] = max(row["scaled_error"] for row in cases)
    report.update({"limits": dict(OD_ERROR_LIMITS), "cases": cases,
                   "by_stage": {str(stage): summarize([row for row in cases if row["stage"] == stage])
                                for stage in sorted({row["stage"] for row in cases})},
                   "weighting": "fixed nominal all-passenger OD collapsed to corridor, reachable off-diagonal",
                   "uncertainty_note": "Latent GP marginal intervals; OD cells are not independent validation samples.",
                   "normalization": "NRMSE is divided by holdout weighted truth RMS; maximum scaled error is the worst case RMSE divided by that case's truth RMS (both use a 0.05-minute floor)."})
    report["passed"] = all(report[name] <= limit for name, limit in OD_ERROR_LIMITS.items())
    return report, mean, latent, weights


def _fill_in_uncertainty_scores(model, X, stages, weights, *, batch_size=256):
    """Weighted RMS latent OD uncertainty, evaluated in bounded input batches."""
    X, stages = model._inputs(X, stages)
    pooled = model.models_["pooled"]
    weights = np.asarray(weights, dtype=float).reshape(-1)
    total = weights.sum()
    if (len(weights) != pooled.n_outputs_ or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isfinite(total) or total <= 0):
        raise ValueError("OD weights must be finite, nonnegative and have a positive total.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("The candidate batch size must be a positive integer.")
    scores = np.zeros(len(X), dtype=float)
    active = pooled.active_indices_
    # All varying outputs share one latent variance; constant outputs have zero.
    scale = np.sqrt((weights[active] / total) @ pooled.target_scale_[active]**2)
    if pooled.kernel_ is None or scale == 0:
        return scores
    for start in range(0, len(X), batch_size):
        stop = start + batch_size
        encoded = np.column_stack([
            X[start:stop], *[(stages[start:stop] == stage).astype(float) for stage in model.stages]
        ])
        cross = pooled.kernel_(encoded, pooled.X_train_)
        solved = solve_triangular(pooled.L_, cross.T, lower=True, check_finite=False)
        variance = pooled.kernel_.k1.diag(encoded) - np.einsum("ij,ij->j", solved, solved)
        scores[start:stop] = np.sqrt(np.maximum(variance, 0.0)) * scale
    return scores


def _select_fill_in_records(model, candidates, existing_records, weights, reachable, n, fill_number):
    """Choose high-posterior-uncertainty cases, without touching holdouts.

    The score is the passenger-weighted RMS latent standard deviation of the
    complete OD-delay prediction.  Candidate points are pre-generated LHS
    points; selecting a point neither observes nor uses validation outcomes.
    """
    if n <= 0:
        return []
    weights = np.asarray(weights, dtype=float).reshape(-1) * np.asarray(reachable).reshape(-1)
    if weights.sum() <= 0:
        raise ValueError("No positive OD weights are available for fill-in selection.")
    weights /= weights.sum()
    seen = {_row_id(row["stage"], [row["inputs"][name] for name in INPUT_NAMES]) for row in existing_records}
    usable = [row for row in candidates
              if _row_id(row["stage"], [row["inputs"][name] for name in INPUT_NAMES]) not in seen]
    if not usable:
        return []
    scores = _fill_in_uncertainty_scores(
        model, _unit_inputs(usable), [row["stage"] for row in usable], weights,
    )
    # Do not spend a batch on near-identical physical inputs for one stage.
    chosen = []
    for index in np.argsort(scores)[::-1]:
        row = usable[int(index)]
        x = np.asarray([row["inputs"][name] for name in INPUT_NAMES], dtype=float)
        same_stage = [other for other in existing_records + chosen if other["stage"] == row["stage"]]
        if same_stage:
            distances = np.linalg.norm(_unit_inputs(same_stage) -
                                        ((x - _bounds()[:, 0]) / (_bounds()[:, 1] - _bounds()[:, 0])), axis=1)
            if np.min(distances) < 0.03:
                continue
        chosen.append({"id": f"fill_{fill_number + len(chosen):03d}_stage{row['stage']}",
                       "role": "fill", "pair_id": None, "stage": row["stage"], "inputs": row["inputs"]})
        if len(chosen) == n:
            break
    return chosen


def build_transport_surrogate(*, stages: Sequence[int] | None = None, n_train: int = 100,
                              validation_fraction: float = .2, min_iterations: int = 8,
                              max_iterations: int = 64, check_every: int = 4,
                              road_gap_threshold: float = .02, n_jobs: int = DEFAULT_WORKERS, seed: int = 42,
                              adaptive_fill_in: bool = False, max_fill_in_points: int = 0,
                              fill_in_batch_size: int = 4, fill_in_rmse_target: float | None = None,
                              fill_in_targets: Mapping[str, float | None] | None = None,
                              artifact_dir: Path | None = None,
                              reuse_cached_only: bool = False) -> dict[str, Any]:
    """Build a surrogate; optionally add uncertainty-guided MSA fill-in cases."""
    from surrogate_model.od_gp import ODDelayGP
    from surrogate_model.transport_surrogate import TransportEmulator
    from surrogate_model.validate_surrogate_appraisal import validate_emulator

    started = time.perf_counter()
    specs = stages_module.get_stages(p.NOMINAL_PARAMS)
    stage_ids = tuple(sorted(map(int, specs if stages is None else stages)))
    if set(stage_ids) - set(map(int, specs)):
        raise ValueError("Requested stages are absent from stages.get_stages().")
    if not (1 <= min_iterations <= max_iterations and check_every >= 1
            and all(int(v) == v for v in (min_iterations, max_iterations, check_every))
            and np.isfinite(road_gap_threshold) and 0 <= road_gap_threshold < 1
            and isinstance(n_jobs, int) and n_jobs >= 1
            and isinstance(adaptive_fill_in, bool) and isinstance(max_fill_in_points, int)
            and max_fill_in_points >= 0 and isinstance(fill_in_batch_size, int) and fill_in_batch_size >= 1
            and (fill_in_rmse_target is None or (np.isfinite(fill_in_rmse_target) and fill_in_rmse_target > 0))):
        raise ValueError("Invalid solver effort, road-gap threshold or worker count.")
    if adaptive_fill_in and max_fill_in_points <= 0:
        raise ValueError("adaptive_fill_in=True requires max_fill_in_points > 0.")
    if adaptive_fill_in and not INPUT_NAMES:
        raise ValueError("adaptive fill-in needs at least one continuous transport input.")
    valid_fill_targets = {"weighted_mae_min", "weighted_rmse_min", "weighted_nrmse",
                          "maximum_case_scaled_error"}
    if fill_in_targets is not None:
        if not isinstance(fill_in_targets, Mapping) or set(fill_in_targets) - valid_fill_targets:
            raise ValueError(f"fill_in_targets may only contain {sorted(valid_fill_targets)}.")
        active_fill_targets = {name: float(value) for name, value in fill_in_targets.items() if value is not None}
        if (adaptive_fill_in and not active_fill_targets) or any(not np.isfinite(value) or value <= 0
                                                               for value in active_fill_targets.values()):
            raise ValueError("Enable at least one finite positive fill-in target.")
    else:
        active_fill_targets = {"weighted_rmse_min": OD_ERROR_LIMITS["weighted_rmse_min"]
                               if fill_in_rmse_target is None else float(fill_in_rmse_target)}
    records = _make_design(stage_ids, n_train, validation_fraction, seed)
    from surrogate_model.response_table import response_table_design
    response_design = response_table_design(stage_ids)
    print(f"Response preparation: {response_design['conditions']} input conditions, "
          f"{response_design['stage_evaluations']} stage evaluations; "
          f"knots per input: {response_design['input_knots']}", flush=True)
    directory = Path(artifact_dir or PROJECT_ROOT / "data/processed/transport_surrogate").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    model_inputs = current_model_inputs(specs)
    portable_inputs = portable_model_inputs(specs)
    equations = reference_equations()
    physical_inputs = json.loads(json.dumps(dict(model_inputs, reference_equations=equations), default=_json_default))
    changes = _input_changes(physical_inputs, _physical_inputs_on_disk())
    if changes:
        raise RuntimeError("Loaded transport settings differ from the saved files: " + ", ".join(changes)
                           + ". Restart the notebook kernel before training.")
    fingerprint = stable_fingerprint(model_inputs)
    sources = _source_inventory()
    solver = dict(stopping_rule="road_relative_gap", min_iterations=int(min_iterations),
                  max_iterations=int(max_iterations), check_every=int(check_every),
                  road_gap_threshold=float(road_gap_threshold), od_threshold=0.0,
                  drive_occupancy=float(tmi.ASSIGNMENT_SETTINGS["drive_occupancy"]))
    contract = {"format_version": 3, "model_fingerprint": fingerprint,
                "reference_equations": stable_fingerprint(equations),
                "solver_settings": solver,
                "route_loader": "native_predecessor_tree",
                "target": "directional_internal_zone_gate_od_delay_minutes"}
    requested_contract = contract
    contract, reference_sources = _reference_cache(directory, contract)
    cache_namespace = stable_fingerprint(contract)
    cache_dir = directory / "msa_cache" / cache_namespace
    design_document = {"n_train_total": int(n_train),
                "validation_fraction_of_training": float(validation_fraction),
                "n_validation_total": sum(row["role"] == "test" for row in records),
                "adaptive_fill_in": {"enabled": adaptive_fill_in, "max_points": max_fill_in_points,
                    "batch_size": fill_in_batch_size, "targets": active_fill_targets},
                "seed": int(seed), "records": records, "cache_contract": contract,
                **({"compatible_contract": requested_contract} if requested_contract != contract else {}),
                "reference_equations": equations,
                "physical_inputs": physical_inputs, "source_sha256": sources,
                "reference_source_sha256": reference_sources or sources}
    if not reuse_cached_only:
        _write_json(directory / "design.json", design_document)
    paths, references = _evaluate_many(records, stage_ids=stage_ids, contract=contract, cache_dir=cache_dir,
                                      n_jobs=n_jobs, reuse_cached_only=reuse_cached_only)
    if reuse_cached_only:
        _write_json(directory / "design.json", design_document)
    changes = _input_changes(physical_inputs, _physical_inputs_on_disk())
    if changes:
        raise RuntimeError("Physical transport inputs changed during reference generation: "
                           + ", ".join(changes) + ". Cached rows retain their original inputs.")
    labels, reachable = references[0]["labels"], np.asarray(references[0]["reachable"], dtype=bool)
    for reference in references:
        if (reference["labels"] != labels or not np.array_equal(reference["reachable"], reachable)
                or not np.array_equal(reference["weights"], references[0]["weights"])):
            raise ValueError("References differ in OD layout/reachability; one pooled output is not valid.")
    train_records = [row for row in records if row["role"] in ("train", "fill")]
    test_records = [row for row in records if row["role"] == "test"]
    targets = np.stack([row["delay"].reshape(-1) for row, spec in zip(references, records) if spec["role"] == "train"])
    truth = np.stack([row["delay"].reshape(-1) for row, spec in zip(references, records) if spec["role"] == "test"])
    def fit_and_validate():
        fitted = ODDelayGP(stages=stage_ids, output_scale_floor=.01,
                            n_restarts_optimizer=1, random_state=int(seed))
        fitted.fit(_unit_inputs(train_records), [row["stage"] for row in train_records], targets)
        checked, predicted, uncertainty, od_weights = _od_validation(
            fitted, test_records, truth, references[0]["weights"], reachable)
        return fitted, checked, predicted, uncertainty, od_weights

    model, validation, mean, std, weights = fit_and_validate()
    fill_history = []
    candidates = []
    if adaptive_fill_in:
        candidate_inputs = _scale_unit_design(qmc.LatinHypercube(len(INPUT_NAMES), seed=seed + 2).random(2500))
        candidates = [{"stage": stage, "inputs": dict(zip(INPUT_NAMES, map(float, x)))}
                      for x in candidate_inputs for stage in stage_ids]
    def fill_targets_reached(report):
        return all(report[name] <= limit for name, limit in active_fill_targets.items())

    while adaptive_fill_in and not fill_targets_reached(validation) and len(fill_history) < max_fill_in_points:
        count = min(fill_in_batch_size, max_fill_in_points - len(fill_history))
        additions = _select_fill_in_records(model, candidates, records, references[0]["weights"], reachable,
                                            count, len(fill_history))
        if not additions:
            warnings.warn("No sufficiently distinct fill-in candidates remain; stopping adaptive sampling.")
            break
        add_paths, add_references = _evaluate_many(additions, stage_ids=stage_ids, contract=contract,
                                                   cache_dir=cache_dir, n_jobs=n_jobs,
                                                   reuse_cached_only=reuse_cached_only)
        for reference in add_references:
            if (reference["labels"] != labels or not np.array_equal(reference["reachable"], reachable)
                    or not np.array_equal(reference["weights"], references[0]["weights"])):
                raise ValueError("Fill-in references differ in OD layout/reachability; cannot refit pooled GP.")
        records.extend(additions); paths.extend(add_paths); references.extend(add_references)
        train_records.extend(additions)
        targets = np.vstack([targets, np.stack([row["delay"].reshape(-1) for row in add_references])])
        fill_history.extend({"id": row["id"], "stage": row["stage"]} for row in additions)
        model, validation, mean, std, weights = fit_and_validate()
        progress = ", ".join(f"{name}={validation[name]:.4f}/{limit:.4f}"
                             for name, limit in active_fill_targets.items())
        print(f"Fill-in: {len(fill_history)}/{max_fill_in_points}; holdout targets: {progress}.", flush=True)
    # Persist the final design as well: fill-in records are ordinary cached MSA
    # cases and must remain reproducible on a later cached-only rebuild.
    design_document["records"] = records
    design_document["n_train_total"] = len(train_records)
    design_document["n_validation_total"] = len(test_records)
    design_document["adaptive_fill_in"].update({"points_added": len(fill_history),
        "final_holdout_metrics": {name: validation[name] for name in active_fill_targets},
        "target_reached": fill_targets_reached(validation)})
    _write_json(directory / "design.json", design_document)
    bundle = {"model": model, "od_labels": labels, "input_names": list(INPUT_NAMES),
              "input_bounds": _bounds().tolist(), "stages": list(stage_ids),
              "source_fingerprint": fingerprint, "reachable": reachable}
    changes = _input_changes(portable_inputs, portable_model_inputs(specs))
    if changes:
        raise RuntimeError("Prepared transport inputs changed during reference generation: "
                           + ", ".join(changes) + ". Cached rows retain their original inputs.")
    # Invalidate the old publication before replacing any payload files.
    _write_json(directory / "manifest.json", {"format_version": 3,
                "emulator_kind": "pooled_od_delay_gp", "validation_passed": False,
                "validation_status": "publishing", "stages": list(stage_ids)})
    temporary = directory / "od_emulator.tmp.joblib"
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(directory / "od_emulator.joblib")
    shape = (len(test_records), len(labels), len(labels))
    with (directory / "od_validation.tmp.npz").open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray([row["id"] for row in test_records]),
                            stages=np.asarray([row["stage"] for row in test_records]),
                            inputs=np.asarray([[row["inputs"][name] for name in INPUT_NAMES] for row in test_records]),
                            labels=np.asarray(labels), actual=truth.reshape(shape), mean=mean.reshape(shape),
                            latent_std=std.reshape(shape), weights=weights.reshape(len(labels),len(labels)))
    (directory / "od_validation.tmp.npz").replace(directory / "od_validation.npz")
    table = []
    for record, reference, path in zip(records, references, paths):
        diagnostics = reference["solver_diagnostics"]
        actual_iterations = int(diagnostics["iterations"])
        road_gap = float(diagnostics["final_road_relative_gap_feasible_averaged_od"])
        if (not np.isfinite(road_gap) or road_gap < 0
                or not min_iterations <= actual_iterations <= max_iterations
                or bool(diagnostics["stopping_rule_met"]) != (road_gap <= road_gap_threshold)):
            raise ValueError("Cached solver diagnostics contradict the requested road-gap policy.")
        table.append({key: record[key] for key in ("id", "role", "pair_id", "stage")} |
                     record["inputs"] | {"msa_iterations": actual_iterations,
                     "road_relative_gap": road_gap,
                     "stopping_rule_met": bool(diagnostics["stopping_rule_met"]),
                     "termination_reason": str(diagnostics["termination_reason"]),
                     "runtime_seconds": reference["runtime_seconds"], "cache_path": str(path.relative_to(directory))})
    pd.DataFrame(table).to_parquet(directory / "msa_design_and_outputs.parquet", index=False)
    source_converged = all(row["stopping_rule_met"] for row in table)
    manifest = {"format_version": 3, "emulator_kind": "pooled_od_delay_gp",
                "method": "Pooled one-hot ARD Matérn-5/2 GP of full directional OD delays; no PCA",
                "artifact_file": "od_emulator.joblib", "input_names": list(INPUT_NAMES),
                "input_domain": uc.TRANSPORT_SURROGATE_INPUTS, "stages": list(stage_ids),
                "od_labels": labels, "od_labels_hash": stable_fingerprint({"labels": labels}),
                "n_train_total": len(train_records), "n_validation_total": len(test_records),
                "adaptive_fill_in": design_document["adaptive_fill_in"],
                "training_record_ids": [row["id"] for row in train_records],
                "validation_record_ids": [row["id"] for row in test_records],
                "design_sha256": _file_sha256(directory / "design.json"),
                "training_counts_by_stage": {str(s): sum(row["stage"] == s for row in train_records) for s in stage_ids},
                "seed": int(seed), "design": "continuous LHS with balanced stage labels; independent paired-stage LHS holdout",
                "model_fingerprint": fingerprint, "cache_namespace": cache_namespace,
                "portable_model_inputs": portable_inputs,
                "reference_equations": equations,
                "cache_contract": contract, "training_solver_settings": solver,
                "source_fingerprint": stable_fingerprint(sources), "source_convergence_passed": source_converged,
                "gp_diagnostics": model.diagnostics_, "validation": {"od": validation},
                "artifact_sha256": _file_sha256(directory / "od_emulator.joblib"),
                "validation_data_sha256": _file_sha256(directory / "od_validation.npz"),
                "validation_passed": False, "validation_status": "paired_appraisal_pending"}
    _write_json(directory / "manifest.json", manifest)
    # Validate the pending artifact before publication. No extra MSA is run.
    emulator = TransportEmulator(directory, manifest, bundle)
    appraisal = validate_emulator(emulator, design=pd.DataFrame(table), write_outputs=False)
    manifest["validation_passed"] = bool(validation["passed"] and appraisal["passed"] and source_converged)
    manifest["validation"].update(paired_appraisal=appraisal, passed=manifest["validation_passed"])
    manifest["validation_status"] = "passed" if manifest["validation_passed"] else "failed"
    manifest["runtime_seconds"] = time.perf_counter() - started
    _write_json(directory / "validation.json", manifest["validation"])
    _write_json(directory / "manifest.json", manifest)
    # Prepare the physical indicators and matched welfare moments once, so all
    # planning notebooks can use the same inexpensive response interface.
    from surrogate_model.response_table import prepare_response_table
    prepare_response_table(PROJECT_ROOT, force=True, progress=True, artifact_dir=directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["runtime_seconds"] = time.perf_counter() - started
    _write_json(directory / "manifest.json", manifest)
    return manifest


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-only", action="store_true",
                        help="Prepare the shared response table from an existing fitted GP; do not train.")
    parser.add_argument("--project-root", "--root", type=Path, default=PROJECT_ROOT,
                        help="Project containing the saved GP for --response-only; training uses this checkout.")
    parser.add_argument("--stages", type=int, nargs="+", default=None)
    parser.add_argument("--n-train", type=int, default=100, help="Total training simulations across stages.")
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--min-iterations", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=64)
    parser.add_argument("--check-every", type=int, default=4)
    parser.add_argument("--road-gap-threshold", type=float, default=.02)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--reuse-cached-only", action="store_true",
                        help="Fit, validate and prepare responses only if all reference simulations are already cached.")
    return parser.parse_args()


def _main():
    options = vars(_arguments())
    response_only = options.pop("response_only")
    project_root = options.pop("project_root").resolve()
    if response_only:
        from surrogate_model.response_table import prepare_response_table
        prepare_response_table(project_root, force=False, progress=True,
                               artifact_dir=options["artifact_dir"])
        print("Shared transport response table is ready.")
        return
    if project_root != PROJECT_ROOT.resolve():
        raise ValueError("GP training uses this checkout's source modules. Use --response-only "
                         "with --project-root to prepare an existing model in another project.")
    result = build_transport_surrogate(**options)
    print(json.dumps({name: result[name] for name in
                     ("validation_passed", "n_train_total", "n_validation_total", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    _main()
