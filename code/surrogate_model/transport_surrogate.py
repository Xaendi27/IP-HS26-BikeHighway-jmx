"""Shared OD-delay GP and prepared transport responses for Notebooks 03–05."""
from __future__ import annotations
from additional import uncertainty as uc
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import ast
import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence
import joblib
import numpy as np
import pandas as pd

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "data/processed/transport_surrogate"
FORMAT_VERSION = 3

class SurrogateError(RuntimeError):
    """User-readable artifact or prediction failure."""

class OutOfDomainError(SurrogateError):
    """Physical inputs are outside the fitted domain."""

def warn_surrogate_validation(manifest):
    """Report completed accuracy/convergence checks without stopping analysis."""
    report = manifest.get("validation", {})
    findings = []
    if report.get("od", {}).get("passed") is False:
        findings.append("OD-delay accuracy")
    appraisal = report.get("paired_appraisal", {})
    if appraisal_diagnostics_status(manifest) == "current":
        findings.extend(f"appraisal {name}" for name, item in appraisal.get("paired_components", {}).items()
                        if item.get("passed") is False)
        findings.extend(f"appraisal {name}" for name, item in appraisal.get("checks", {}).items()
                        if item.get("passed") is False and name != "material_component_failures")
    if manifest.get("source_convergence_passed") is False:
        findings.append("MSA reference convergence")
    if not findings:
        return
    details = ", ".join(findings)
    warnings.warn(
        f"Surrogate validation warning: {details} did not meet the configured targets. "
        "Continuing; review validation.json and manifest.json in data/processed/transport_surrogate.",
        UserWarning, stacklevel=2,
    )

def stable_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]

@lru_cache(maxsize=128)
def _file_digest(path, size, modified_ns):
    # Text configs are hashed with LF endings, so git autocrlf checkouts
    # on different machines keep the same physical model identity.
    if Path(path).suffix.lower() == ".json":
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_file_key(path, root):
    """Use portable relative keys, allowing explicitly configured external files."""
    path, root = Path(path).resolve(), Path(root).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def _model_inputs(stage_specs, *, portable=False):
    """Collect physical settings and either source-data or local-cache identity."""
    import parameters as p
    import transport_model_interface as tmi
    from additional.section_flows import physical_signature
    from additional.uncertainty import complete_transport_inputs
    root = Path(tmi.__file__).resolve().parents[1]
    network = tmi.configured_detailed_network_path(root)
    paths = list(sorted((root / "data/transport/config").glob("*.json")))
    paths.extend(tmi.configured_input_paths(root).values() if portable else [root / "cache/tmi_context.pkl"])
    if network is not None:
        paths.append(network)
    files = {}
    for path in paths:
        if path.is_file():
            stat = path.stat()
            files[_input_file_key(path, root)] = _file_digest(str(path), stat.st_size, stat.st_mtime_ns)
        else:
            if portable:
                raise SurrogateError(f"Missing prepared transport input: {_input_file_key(path, root)}")
            files[_input_file_key(path, root)] = None
    fixed_inputs = {name: value for name, value in complete_transport_inputs().items()
                    if name not in uc.TRANSPORT_SURROGATE_INPUTS}
    return {
        **({"fixed_transport_inputs": fixed_inputs} if fixed_inputs else {}),
        "identity_version": 3 if portable else 2,
        "corridor": {"mode": getattr(p, "CORRIDOR_DEFINITION_MODE", "municipalities"),
                     "municipalities": sorted(map(str, p.CORRIDOR_MUNICIPALITIES)),
                     "zone_ids": sorted(map(str, getattr(p, "CORRIDOR_ZONE_IDS", None) or []))},
        "stages": {str(k): _physical_settings(v) for k, v in stage_specs.items()},
        "section_signatures": {str(k): physical_signature(v.get("section_config")) for k, v in stage_specs.items()},
        "input_domain": {name: {k: spec[k] for k in ("minimum", "maximum", "nominal")}
                         for name, spec in uc.TRANSPORT_SURROGATE_INPUTS.items()},
        "drive_occupancy": tmi.ASSIGNMENT_SETTINGS.get("drive_occupancy", 1.14),
        "prepared_inputs": files}


def current_model_inputs(stage_specs):
    """Physical inputs used to identify local reference-generation caches.

    Each artifact records its actual solver policy separately. Changing future
    training defaults or the supplementary appraisal cohort does not refit it.
    """
    return _model_inputs(stage_specs)


def portable_model_inputs(stage_specs):
    """Identify a shared fitted model from prepared data, not a local pickle cache."""
    return _model_inputs(stage_specs, portable=True)


def _physical_settings(value):
    """Remove presentation names without changing physical selector/effect values."""
    if isinstance(value, Mapping):
        return {key: _physical_settings(item) for key, item in value.items()
                if key not in {"name", "description"}}
    if isinstance(value, (list, tuple)):
        return [_physical_settings(item) for item in value]
    return value


def model_identity(manifest):
    """Current physical identity, including explicitly migrated saved artifacts."""
    return manifest.get("compatibility", {}).get("model_fingerprint", manifest.get("model_fingerprint"))


def current_model_fingerprint(stage_specs):
    """Identify the physical model independently of planning/appraisal sources."""
    return stable_fingerprint(current_model_inputs(stage_specs))

def _input_changes(before, after, prefix=""):
    """Name changed physical settings instead of reporting an opaque hash."""
    changed = []
    for key in sorted(before.keys() | after.keys()):
        name = f"{prefix}.{key}" if prefix else key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changed.extend(_input_changes(left, right, name))
        elif left != right:
            changed.append(name)
    return changed


def _check_model_inputs(manifest, stage_specs, root):
    """Validate published source identities, retaining legacy cache checks."""
    import transport_model_interface as tmi
    saved = manifest.get("portable_model_inputs")
    if saved is not None:
        changed = _input_changes(saved, portable_model_inputs(stage_specs))
        if changed:
            raise SurrogateError("Saved GP physical inputs differ from current configuration: "
                                 + ", ".join(changed))
    else:
        freshness = tmi.context_cache_status(root)
        if not freshness["ready"]:
            raise SurrogateError("Transport input cache is not current: " + freshness["reason"]
                                 + ". Refresh prepared transport inputs before rebuilding the surrogate.")
        if model_identity(manifest) != current_model_fingerprint(stage_specs):
            raise SurrogateError("Saved GP physical inputs differ from current configuration.")


def equation_identity(roots, source_texts=None):
    """Hash selected calculations and local dependencies without presentation text."""
    class WithoutDocstrings(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

    equations = {}
    for relative, names in roots.items():
        source = ((source_texts or {}).get(relative) if source_texts is not None else None)
        if source is None:
            source = (Path(__file__).resolve().parents[2] / relative).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        nodes = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nodes[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        nodes[target.id] = node
        # Effective reference solver settings have their own cache contract.
        if relative == "code/transport_model_interface.py":
            nodes.pop("ASSIGNMENT_SETTINGS", None)
            nodes.pop("ROUTE_ASSIGNMENT_DEFAULTS", None)
        pending, visited = list(names), set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            node = WithoutDocstrings().visit(nodes[name])
            equations[f"{relative}:{name}"] = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()).hexdigest()
            pending.extend(item.id for item in ast.walk(node)
                           if isinstance(item, ast.Name) and item.id in nodes and item.id not in visited)
    return equations


def reference_equations(source_texts=None):
    """Physical transport equations; optional saved sources allow audited comparison."""
    roots = {
        "code/transport_model_interface.py": (
            "load_transport_context", "build_corridor_context", "apply_road_capacity_stage",
            "run_transport_mode_choice", "run_coupled_corridor_assignment",
            "collapse_od_to_gates", "_bpr_link_times", "_corridor_shortest_time_table"),
        "code/additional/uncertainty.py": (
            "complete_transport_inputs", "native_transport_inputs", "TRANSPORT_BINDINGS"),
        "code/additional/section_flows.py": ("apply_section_time_saving",),
        "code/transport_core/interventions.py": ("apply_interventions",),
        "code/transport_core/mode_choice_zurich.py": (
            "load_mode_choice_parameters", "mode_split_aggregated"),
        "code/transport_core/travel_times.py": (
            "load_travel_times", "apply_perceived_time_policies", "apply_ebike_share",
            "standalone_walk_mask", "rebuild_pt_out_of_vehicle_times"),
        "code/transport_core/input_packages.py": ("load_package", "align_matrix"),
        "code/transport_core/zoning.py": ("load_zones", "load_model_inputs"),
        "code/transport_core/config.py": (
            "ZONES_FILE", "DEMAND_PACKAGE_FILE", "SKIM_PACKAGE_FILE", "LOOKUP_PACKAGE_FILE",
            "MODE_CHOICE_FILE", "MODE_ORDER", "NO_PATH_TIME_MIN", "MAX_WALK_DISTANCE_KM",
            "EXTERNAL_BACKGROUND_CITY_FACTOR", "EXTERNAL_BACKGROUND_CANTON_FACTOR",
            "DRIVE_CONNECTOR_CITY_MIN", "DRIVE_CONNECTOR_CANTON_MIN",
            "CAR_TERMINAL_CITY_MIN", "CAR_TERMINAL_CANTON_MIN", "DRIVE_INTRAZONAL_MIN",
            "PT_INTRAZONAL_MIN", "PT_ACCESS_FLOOR_MIN", "PT_EGRESS_FLOOR_MIN",
            "PT_TRANSFER_PHYSICAL_FLOOR_MIN", "EBIKE_SPEED_MULTIPLIER"),
    }
    return equation_identity(roots, source_texts)


def appraisal_diagnostics_status(manifest):
    """Currency of saved monetary diagnostics; accuracy findings remain advisory."""
    import parameters as p
    import simulation_engine as m
    report = manifest.get("validation", {}).get("paired_appraisal")
    if not report:
        return "missing"
    return "current" if (
        report.get("surrogate_fingerprint") == manifest.get("model_fingerprint")
        and report.get("appraisal_welfare_version") == m.APPRAISAL_WELFARE_VERSION
        and stable_fingerprint(report.get("current_parameters", {})) == stable_fingerprint(p.NOMINAL_PARAMS)
    ) else "stale"


def artifact_status(project_root=None, *, stage_specs=None, artifact_dir=None,
                    check_payloads=False, validate_fingerprint=True):
    """Read-only GP/table readiness without loading the GP or transport context."""
    import parameters as p
    import stages
    import transport_model_interface as tmi
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    directory = Path(artifact_dir) if artifact_dir is not None else root / "data/processed/transport_surrogate"
    result = {"gp_status": "missing", "response_status": "missing", "diagnostics_status": "missing",
              "ready": False, "reasons": []}
    path = directory / "manifest.json"
    if not path.is_file():
        result["reasons"].append("No saved surrogate manifest. Build in Notebook 03, Section 3.1.")
        return result
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result["diagnostics_status"] = appraisal_diagnostics_status(manifest)
        if manifest.get("validation_status") in {"publishing", "paired_appraisal_pending"}:
            result["gp_status"] = "incomplete"
            result["reasons"].append("The surrogate build is incomplete.")
            return result
        if manifest.get("format_version") != FORMAT_VERSION or manifest.get("emulator_kind") != "pooled_od_delay_gp":
            raise SurrogateError("The saved artifact is not the current OD surrogate format.")
        specs = stage_specs if stage_specs is not None else stages.get_stages(p.NOMINAL_PARAMS)
        missing = set(map(int, specs)) - set(map(int, manifest["stages"]))
        if missing:
            raise SurrogateError(f"The surrogate lacks infrastructure states {sorted(missing)}.")
        if validate_fingerprint:
            _check_model_inputs(manifest, specs, root)
            saved_equations = manifest.get("compatibility", {}).get("reference_equations", manifest.get("reference_equations"))
            if saved_equations:
                changed = _input_changes(saved_equations, reference_equations())
                if changed:
                    raise SurrogateError("Transport calculations changed: " + ", ".join(changed))
        artifact = directory / manifest.get("artifact_file", "od_emulator.joblib")
        if not artifact.is_file():
            raise SurrogateError("The fitted OD model file is missing.")
        if check_payloads and manifest.get("artifact_sha256"):
            stat = artifact.stat()
            if _file_digest(str(artifact), stat.st_size, stat.st_mtime_ns) != manifest["artifact_sha256"]:
                raise SurrogateError("OD model and manifest do not belong to the same fitted artifact.")
        result["gp_status"] = "ready"
        from .response_table import FILENAME, _fingerprint
        table = directory / FILENAME
        record = manifest.get("response_table", {})
        if not record or not table.is_file():
            result["reasons"].append("Prepare the missing response table in Notebook 03, Section 3.1.")
        elif record.get("fingerprint") != _fingerprint(manifest):
            result["response_status"] = "refresh required"
            result["reasons"].append("Refresh the response table in Notebook 03, Section 3.1.")
        else:
            stat = table.stat()
            if check_payloads and _file_digest(str(table), stat.st_size, stat.st_mtime_ns) != record.get("sha256"):
                result["response_status"] = "refresh required"
                result["reasons"].append("Response-table payload and manifest differ.")
            else:
                result["response_status"] = "ready"
        result["ready"] = result["gp_status"] == result["response_status"] == "ready"
    except (OSError, ValueError, KeyError, TypeError, SurrogateError) as exc:
        if result["gp_status"] == "ready":
            result["response_status"] = "refresh required"
        else:
            result["gp_status"] = "incompatible"
        result["reasons"].append(str(exc))
    return result


@dataclass
class TransportEmulator:
    artifact_dir: Path
    manifest: dict[str, Any]
    bundle: dict[str, Any] | None = None

    def __post_init__(self):
        self._backend = None
        self._state_cache = {}
        self._response_table = None

    @property
    def input_names(self):
        return list(self.manifest["input_names"])

    @property
    def validation(self):
        return deepcopy(self.manifest.get("validation", {}))

    def _get_bundle(self):
        """Load large GP coefficients only for direct OD predictions or preparation."""
        if self.bundle is None:
            path = self.artifact_dir / self.manifest.get("artifact_file", "od_emulator.joblib")
            stat = path.stat()
            if _file_digest(str(path), stat.st_size, stat.st_mtime_ns) != self.manifest.get("artifact_sha256"):
                raise SurrogateError("The fitted OD model changed after this emulator was loaded.")
            bundle = joblib.load(path)
            if bundle.get("source_fingerprint") != self.manifest["model_fingerprint"]:
                raise SurrogateError("OD model and manifest do not belong to the same build.")
            if (list(bundle["input_names"]) != self.input_names
                    or set(map(int, bundle["stages"])) != set(map(int, self.manifest["stages"]))):
                raise SurrogateError("OD model inputs/stages differ from its manifest.")
            labels = list(map(str, bundle["od_labels"]))
            if len(labels) != len(set(labels)) or bundle["model"].n_outputs_ != len(labels)**2:
                raise SurrogateError("Invalid OD target layout in fitted model.")
            self.bundle = bundle
        return self.bundle

    def _vector(self, values):
        missing = set(self.input_names) - set(values)
        if missing:
            raise KeyError(f"Missing surrogate inputs: {sorted(missing)}")
        vector = np.asarray([values[name] for name in self.input_names], dtype=float)
        bounds = np.asarray([(self.manifest["input_domain"][name]["minimum"],
                              self.manifest["input_domain"][name]["maximum"])
                             for name in self.input_names], dtype=float).reshape(-1, 2)
        if not np.isfinite(vector).all():
            raise OutOfDomainError("Surrogate inputs must be finite.")
        outside = (vector < bounds[:, 0] - 1e-12) | (vector > bounds[:, 1] + 1e-12)
        if outside.any():
            names = [self.input_names[i] for i in np.flatnonzero(outside)]
            raise OutOfDomainError(f"Inputs outside the fitted domain: {names}. Extend the domain and rebuild in Notebook 03, Section 3.1.")
        return ((vector - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])).reshape(1, -1)

    def predict_od_delay(self, *, stage, return_std=False, **physical_inputs):
        """Directional internal-zone/gate road delay in minutes; no assignment."""
        stage = int(stage)
        bundle = self._get_bundle()
        if stage not in list(map(int, bundle["stages"])):
            raise SurrogateError(f"Stage {stage} is absent from this fitted model.")
        result = bundle["model"].predict(self._vector(physical_inputs), np.array([stage]),
                                              return_std=return_std, clip_nonnegative=True)
        mean, std = result if return_std else (result, None)
        labels = list(map(str, bundle["od_labels"]))
        values = np.asarray(mean[0], dtype=float).reshape(len(labels), len(labels)).copy()
        if not np.isfinite(values).all():
            raise SurrogateError("GP returned non-finite OD delays.")
        mask = np.asarray(bundle.get("reachable", np.ones_like(values, dtype=bool)), dtype=bool).reshape(values.shape)
        values[~mask] = 0.0
        np.fill_diagonal(values, 0.0)
        table = pd.DataFrame(values, index=labels, columns=labels)
        if not return_std:
            return table
        uncertainty = np.asarray(std[0]).reshape(values.shape).copy()
        uncertainty[~mask] = 0.0
        np.fill_diagonal(uncertainty, 0.0)
        return table, pd.DataFrame(uncertainty, index=labels, columns=labels)

    def _get_backend(self):
        import parameters as p
        import stages
        import transport_model_interface as tmi
        specs = stages.get_stages(p.NOMINAL_PARAMS)
        root = Path(tmi.__file__).resolve().parents[1]
        _check_model_inputs(self.manifest, specs, root)
        if self._backend is None:
            context = tmi.load_transport_context(root)
            base = tmi.build_corridor_context(context, corridor_municipalities=p.CORRIDOR_MUNICIPALITIES,
                                              name="surrogate post-processing corridor")
            self._backend = {"context": context, "stage_specs": specs, "base_modes": {},
                "corridors": {int(s): tmi.apply_road_capacity_stage(base, specs[int(s)])[0]
                              for s in self.manifest["stages"]}}
        return self._backend

    def transport_state_from_od_delay(self, *, stage, delay_table, include_welfare=False, **physical_inputs):
        """Native mode choice/welfare from predicted or saved exact OD delays."""
        import transport_model_interface as tmi
        from additional.uncertainty import native_transport_inputs
        self._vector(physical_inputs)
        backend = self._get_backend()
        stage = int(stage)
        corridor = backend["corridors"][stage]
        if stage not in backend["base_modes"]:
            backend["base_modes"][stage] = tmi.run_transport_mode_choice(
                backend["context"], stage=stage, stage_specs=backend["stage_specs"],
                corridor_zone_ids=corridor.zone_ids, demand_multiplier=1.,
                pt_asc_shift=0., bike_asc_shift=0.)
        state = tmi.transport_state_from_surrogate_od(
            backend["context"], corridor, delay_table, stage=stage,
            stage_specs=backend["stage_specs"], include_welfare=include_welfare,
            base_mode_result=backend["base_modes"][stage], **native_transport_inputs(physical_inputs))
        state["input_values"] = dict(physical_inputs)
        return state

    def predict_transport_state(self, *, stage, include_links=False, include_od_pairs=False,
                                include_welfare=False, return_std=False,
                                **physical_inputs):
        """Shared planning interface; financial calculations remain native.

        OD standard deviations concern interpolation, not benefit error bounds.
        """
        if include_links or include_od_pairs:
            raise SurrogateError("The OD surrogate predicts road OD delays. Use the exact FSM for link maps or municipal trip tables.")
        self._vector(physical_inputs)
        if self._response_table is not None and not return_std:
            return self._response_table.state(stage, physical_inputs, include_welfare=include_welfare)
        self._get_backend()
        key = (int(stage), bool(include_welfare), *(float(physical_inputs[n]) for n in self.input_names))
        if not return_std and key in self._state_cache:
            return deepcopy(self._state_cache[key])
        prediction = self.predict_od_delay(stage=stage, return_std=return_std, **physical_inputs)
        delay, std = prediction if return_std else (prediction, None)
        state = self.transport_state_from_od_delay(stage=stage, delay_table=delay,
                                                   include_welfare=include_welfare, **physical_inputs)
        state["surrogate_representation"] = "pooled_od_delay_gp"
        if return_std:
            state["od_delay_std"] = std
        else:
            limit = 12 if include_welfare else 512
            while len(self._state_cache) >= limit:
                self._state_cache.pop(next(iter(self._state_cache)))
            self._state_cache[key] = deepcopy(state)
        return state

    def predict(self, *, stage, return_std=False, return_links=False, return_od_pairs=False,
                **physical_inputs):
        """Physical metrics from the shared transport response."""
        state = self.predict_transport_state(stage=stage, return_std=return_std,
            include_links=return_links, include_od_pairs=return_od_pairs,
            **physical_inputs)
        result = dict(state["metrics"])
        if return_std:
            result["od_delay_std"] = state["od_delay_std"]
        return result

    def describe(self):
        report = self.manifest.get("validation", {}).get("od", {})
        groups = report.get("by_stage", {})
        return pd.DataFrame([{"stage": int(stage),
            "n_training": self.manifest.get("training_counts_by_stage", {}).get(str(stage)),
            "n_validation": groups.get(str(stage), {}).get("n_cases"),
            "weighted_mae_min": groups.get(str(stage), {}).get("weighted_mae_min", report.get("weighted_mae_min")),
            "weighted_rmse_min": groups.get(str(stage), {}).get("weighted_rmse_min", report.get("weighted_rmse_min")),
            "od_passed": all(groups.get(str(stage), report).get(name, float("inf")) <= limit
                             for name, limit in report.get("limits", {}).items()),
            "reference_convergence_passed": self.manifest.get("source_convergence_passed"),
            "appraisal_diagnostics": appraisal_diagnostics_status(self.manifest)} for stage in self.manifest["stages"]])

def load_transport_surrogate(project_root: str | Path | None = None, *,
                             required_stages: Sequence[int] | None = None,
                             validate_fingerprint=True,
                             use_response_table=True, artifact_dir=None):
    """Load shared response metadata/table; GP coefficients are loaded on demand."""
    import parameters as p
    import stages
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    directory = Path(artifact_dir) if artifact_dir is not None else root / "data/processed/transport_surrogate"
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("No fitted OD surrogate. Build it once in Notebook 03, Section 3.1.")
    specs = stages.get_stages(p.NOMINAL_PARAMS)
    status = artifact_status(root, stage_specs=specs, artifact_dir=directory,
                             check_payloads=True, validate_fingerprint=validate_fingerprint)
    if status["gp_status"] != "ready" or (use_response_table and status["response_status"] != "ready"):
        raise SurrogateError(" ".join(status["reasons"]))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if required_stages is not None and not set(map(int, required_stages)).issubset(map(int, manifest["stages"])):
        raise SurrogateError("The surrogate is missing a requested infrastructure state.")
    emulator = TransportEmulator(directory, manifest)
    if use_response_table:
        from .response_table import load_response_table
        emulator._response_table = load_response_table(directory, manifest)
    warn_surrogate_validation(manifest)
    return emulator

def load_surrogate_validation(project_root=None):
    """Saved diagnostics: plotting them requires no transport simulations."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    directory = root / "data/processed/transport_surrogate"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    status = artifact_status(root)
    if status["gp_status"] != "ready":
        raise SurrogateError(" ".join(status["reasons"]))
    path = directory / "od_validation.npz"
    stat = path.stat()
    if _file_digest(str(path), stat.st_size, stat.st_mtime_ns) != manifest.get("validation_data_sha256"):
        raise SurrogateError("Saved OD diagnostics do not match the fitted model manifest.")
    with np.load(path, allow_pickle=False) as package:
        data = {key: package[key].copy() for key in package.files}
    if (list(map(str, data["labels"])) != list(map(str, manifest["od_labels"]))
            or list(map(str, data["ids"])) != manifest["validation_record_ids"]):
        raise SurrogateError("Saved diagnostic OD labels/case IDs differ from the fitted model.")
    error = np.asarray(data["mean"], dtype=float) - data["actual"]
    weights = np.asarray(data["weights"], dtype=float).reshape(error.shape[1:])
    weights = weights / weights.sum()
    cases = pd.DataFrame({"id": data["ids"].astype(str), "stage": data["stages"],
        "weighted_mae_min": np.sum(np.abs(error) * weights, axis=(1, 2)),
        "weighted_rmse_min": np.sqrt(np.sum(error**2 * weights, axis=(1, 2))),
        "weighted_observed_delay_min": np.sum(data["actual"] * weights, axis=(1, 2)),
        "weighted_predicted_delay_min": np.sum(data["mean"] * weights, axis=(1, 2))})
    return {"manifest": manifest, "cases": cases, **data}
