"""
adaptive_planning.py
===========
Deployment-plan definitions and the dynamic adaptive planning simulation engine.

This file contains both the user-friendly definitions of the six plans (at the top)
and the simulation engine that turns each plan into a 40-year system pathway.

SURROGATE ADAPTATION RULE
-------------------------
Students may change plan timing, costs, triggers, persistence, and lead times
without rebuilding the transport surrogate. Rebuild only when a plan change
also changes the physical transport state represented by a stage; physical
stage packages belong in stages.py, not in this file.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real

import numpy as np
import pandas as pd

from parameters import (
    N_YEARS,
    NOMINAL_PARAMS,
    MAX_AVG_TT,
    PT_SHARE_TARGET,
)
import stages
import simulation_engine as m
from additional import uncertainty as uc


# =============================================================================
# 1. ADAPTIVE TRIGGERS (STUDENT-EDITABLE)
# =============================================================================
# Triggers are OPERATIONAL ACTION THRESHOLDS that initiate building a package:
#   - Signpost: the real-time operational metric being monitored
#   - Threshold: the value that indicates the current system needs an upgrade
#   - Persistence: consecutive years the threshold must be exceeded before deciding
#   - Lead time: construction/implementation duration from decision to opening
#
# MehrSpur monitors daily passenger-trips through parameters.SECTION, summed
# across the configured directions. The count includes modeled passengers and
# the optional supplementary cohort. It is distinct from all municipal-corridor
# PT trips and from the comfort-capacity thresholds used to price crowding.
# Both rules use the previous year's observations and operate independently.
# Either package can be built first, or both can be under construction together.

# Common signposts and their units. Other numeric simulate_year() result fields
# can also be monitored; a misspelled or missing result field raises an error.
# Every trigger tests STRICTLY GREATER THAN its threshold (value > threshold).
# Shares without "_trips" use passenger-kilometres (PKM), not trip counts.
SIGNPOSTS = {
    "section_combined_daily": {"label": "Counting-section passengers/day", "unit": "passenger-trips/day"},
    "section_modeled_daily": {"label": "Modeled counting-section passengers/day", "unit": "passenger-trips/day"},
    "section_load_ratio": {"label": "Combined peak section load / comfort capacity", "unit": "ratio"},
    "pt_trips": {"label": "Peak-hour PT demand", "unit": "trips/peak hour"},
    "annual_pt_trips": {"label": "Annual PT demand", "unit": "trips/year"},
    "pt_share": {"label": "PT share of passenger-kilometres", "unit": "PKM share (0-1)"},
    "pt_share_trips": {"label": "PT share of trips", "unit": "trip share (0-1)"},
    "avg_tt_min": {"label": "Average corridor travel time", "unit": "minutes"},
    "congestion_delay_hours": {"label": "Annual passenger congestion delay", "unit": "person-hours/year"},
    "bike_trips": {"label": "Peak-hour bike demand", "unit": "trips/peak hour"},
    "bike_share": {"label": "Bike share of passenger-kilometres", "unit": "PKM share (0-1)"},
    "bike_share_trips": {"label": "Bike share of trips", "unit": "trip share (0-1)"},
}

DEFAULT_TRIGGERS = {
    # Trigger 1: Section demand -> package 1 (local stations and access)
    "trigger1": {
        "signpost":        "section_combined_daily",
        "threshold":       125_000,
        "persistence":     2,       # Consecutive years above threshold
        "lead_time":       6,       # Years of planning and station civil works
    },
    # Trigger 2: The same section count -> package 2 (Brüttenertunnel)
    # An absolute daily count keeps the signpost comparable when stations add
    # capacity. The action threshold is separate from comfort capacity.
    "trigger2": {
        "signpost":        "section_combined_daily",
        "threshold":       122_000,
        "persistence":     2,
        # Years of tunnel planning and construction after the decision.
        "lead_time":       10,
    }
}


# =============================================================================
# 2. DEPLOYMENT PLAN DEFINITIONS (STUDENT-EDITABLE)
# =============================================================================
# Packages and transport states are different. Edit named packages here:
#   initial_packages lists the packages already open in Year 1.
#   stations and tunnel specify when to add the respective package.
# Internally these combinations map to transport states: 0 = baseline,
# 1 = stations only, 2 = tunnel only, 3 = both packages. Either package can
# open first; adding the other produces state 3.
#
# Transition examples:
#   None                                      -> never add this package
#   {"type": "fixed", "opening_year": 9}       -> open in Year 9
#   {"type": "trigger", "trigger": "trigger1"} -> apply that trigger's rule
# Fixed years must fall within 1..N_YEARS. Trigger decisions use the previous
# year's observation; opening occurs in decision year + lead_time.
# Construction emissions are spread across that lead time, before opening.
# Fixed/initial packages use the matching DEFAULT_TRIGGERS lead time (6/10 years)
# unless their fixed rule specifies lead_time. Pre-horizon emissions are booked
# at time zero, the start of Year 1, using its carbon price.
# Costs come from parameters.py; physical effects and asset lives from stages.py.
# Names and descriptions are generated below so timing edits need only one change.
PLAN_DEFINITIONS = {
    # Package "stations" = east section Altstetten–Schlieren (stages.py)
    # Package "tunnel"   = west section Schlieren–Dietikon (stages.py)
    "baseline": {
        "name": "Baseline (no bike highway)",
        "initial_packages": [], "stations": None, "tunnel": None,
    },
    "static1": {
        "name": "Static 1 (east section only)",
        "initial_packages": ["stations"], "stations": None, "tunnel": None,
    },
    "static2": {
        "name": "Static 2 (full bike highway)",
        "initial_packages": ["stations", "tunnel"], "stations": None, "tunnel": None,
    },
    "staged1": {
        "name": "Staged 1",
        "initial_packages": [],
        "stations": {"type": "fixed", "opening_year": 8},
        "tunnel": {"type": "fixed", "opening_year": 11},
    },
    "staged2": {
        "name": "Staged 2",
        "initial_packages": [],
        "stations": {"type": "fixed", "opening_year": 6},
        "tunnel": {"type": "fixed", "opening_year": 15},
    },
    "flexible": {
        "name": "Flexible",
        "initial_packages": [],
        "stations": {"type": "trigger", "trigger": "trigger1"},
        "tunnel": {"type": "trigger", "trigger": "trigger2"},
    },
}


# =============================================================================
# HELPERS: VALIDATION, DISPLAY TEXT, AND DERIVED COSTS
# =============================================================================
def _validate_trigger(trigger: dict, label: str) -> None:
    """Reject ambiguous rules before evaluating a pathway."""
    signpost = trigger.get("signpost")
    if not isinstance(signpost, str) or not signpost.strip():
        raise ValueError(f"{label}: signpost must name an annual result field.")
    threshold = trigger.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not np.isfinite(threshold):
        raise ValueError(f"{label}: threshold must be a finite number.")
    for field, minimum in (("persistence", 1), ("lead_time", 0)):
        value = trigger.get(field)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{label}: {field} must be an integer >= {minimum}.")


def _describe_trigger(trigger: dict, name: str) -> dict:
    result = deepcopy(trigger)
    _validate_trigger(result, name)
    metric = SIGNPOSTS.get(result["signpost"], {"label": result["signpost"], "unit": "result units"})
    result["name"] = name
    result["signpost_label"] = metric["label"]
    result["description"] = (
        f"When {metric['label']} exceeds {result['threshold']:,g} {metric['unit']} "
        f"for {result['persistence']} consecutive years, decide to build; "
        f"open after a {result['lead_time']}-year lead time."
    )
    return result


def get_triggers(params: dict | None = None) -> dict:
    """Return validated rules and generated text, with optional parameter overrides."""
    triggers = deepcopy(DEFAULT_TRIGGERS)
    for key, trigger in triggers.items():
        for field in ("threshold", "signpost", "persistence", "lead_time"):
            override = f"{key.upper()}_{field.upper()}"
            if params is not None and override in params:
                trigger[field] = params[override]
        triggers[key] = _describe_trigger(trigger, f"Trigger {key.removeprefix('trigger')}")
    return triggers


def _validate_plan(spec: dict, name: str) -> None:
    initial = spec.get("initial_stage")
    if isinstance(initial, bool) or not isinstance(initial, Integral) or initial not in stages.STATE_IDS:
        raise ValueError(f"Plan '{name}': initial_stage must be one of {stages.STATE_IDS}.")
    for key in ("to1", "to2"):
        transition = spec[key]
        if transition is None:
            continue
        label = f"Plan '{name}', {key}"
        if transition.get("type") == "fixed":
            year = transition.get("year")
            if isinstance(year, bool) or not isinstance(year, Integral) or not 1 <= year <= N_YEARS:
                raise ValueError(f"{label}: fixed year must be an integer within 1..{N_YEARS}.")
            if "lead_time" in transition:
                lead = transition["lead_time"]
                if isinstance(lead, bool) or not isinstance(lead, Integral) or lead < 0:
                    raise ValueError(f"{label}: lead_time must be an integer >= 0.")
        elif transition.get("type") == "trigger":
            _validate_trigger(transition, label)
        else:
            raise ValueError(f"{label}: type must be 'fixed' or 'trigger'.")


def validate_section_signpost(rule: dict | None, params: dict | None = None,
                              *, label: str = "Adaptive trigger") -> None:
    """Section observations require an active section; capacity ratios need crowding."""
    if not rule or rule.get("type", "trigger") != "trigger":
        return
    signpost = rule.get("signpost", "")
    if not signpost.startswith("section_"):
        return
    from additional.section_flows import section_config
    config = section_config(params)
    if not config["active"]:
        raise ValueError(
            f"{label}: '{signpost}' requires an active SECTION. Configure a section "
            "or select a different signpost in adaptive_planning.py."
        )
    if signpost == "section_load_ratio" and not config.get("crowding_enabled", False):
        raise ValueError(f"{label}: section_load_ratio requires an active PT comfort-capacity model.")


def _previous_signpost(metrics: dict, transition: dict, label: str):
    """Read the selected observation; Year 1 has no previous observation."""
    if not metrics:
        return None
    signpost = transition["signpost"]
    if signpost.startswith("section_") and not metrics.get("section_active", False):
        raise ValueError(f"{label}: section observations are unavailable; enable SECTION or change the signpost.")
    if signpost not in metrics:
        raise ValueError(
            f"{label}: signpost '{signpost}' is absent from the annual results. "
            "Choose a SIGNPOSTS entry or another returned numeric result field."
        )
    return metrics[signpost]


def advance_trigger(year, active, opening, persistence, previous, rule, *, threshold=None):
    """Advance one package's trigger for scalars or arrays; zero means unscheduled.

    The previous year's observation drives the decision. Construction can finish
    in the decision year when lead_time is zero.
    """
    active, opening = np.asarray(active, dtype=bool), np.asarray(opening, dtype=int)
    persistence = np.asarray(persistence, dtype=int)
    previous = np.asarray(np.nan if previous is None else previous, dtype=float)
    threshold = rule["threshold"] if threshold is None else threshold
    eligible = ~active & (opening == 0)
    above = np.isfinite(previous) & (previous > threshold)
    persistence = np.where(eligible & above, persistence + 1, 0)
    decision = eligible & (persistence >= rule["persistence"])
    opening = np.where(decision, year + rule["lead_time"], opening)
    opened = ~active & (opening > 0) & (year >= opening)
    return active | opened, opening, persistence, decision, opened


def construction_lead_time(plan_def: dict, package: int, params: dict | None = None) -> int:
    """Construction duration: the package rule, or its default trigger lead time."""
    if package not in (1, 2):
        raise ValueError("Construction package must be 1 (stations) or 2 (tunnel).")
    transition = plan_def.get(f"to{package}") or {}
    lead = transition.get("lead_time", get_triggers(params)[f"trigger{package}"]["lead_time"])
    if isinstance(lead, bool) or not isinstance(lead, Integral) or lead < 0:
        raise ValueError("Construction lead_time must be an integer >= 0.")
    return int(lead)


def construction_emissions_in_year(package: int, year: int, opening_year, lead_time: int):
    """Return (annual, time-zero) tonnes for scalar or array package openings.

    Zero means unscheduled. Positive lead times spread emissions evenly from
    opening minus lead_time through the year before opening. A zero lead time
    emits at opening. Pre-horizon shares appear once in the Year-1 row; later
    shares beyond the horizon are counted only if that year is evaluated.
    """
    if isinstance(year, bool) or not isinstance(year, Integral) or year < 1:
        raise ValueError("Construction year must be an integer >= 1.")
    if isinstance(lead_time, bool) or not isinstance(lead_time, Integral) or lead_time < 0:
        raise ValueError("Construction lead_time must be an integer >= 0.")
    opening = np.asarray(0 if opening_year is None else opening_year, dtype=float)
    if not np.isfinite(opening).all() or (opening < 0).any() or (opening != np.floor(opening)).any():
        raise ValueError("Opening years must be nonnegative integers; zero means unscheduled.")
    total = stages.construction_emissions_tonnes(package)
    scheduled = opening > 0
    start = opening - lead_time
    duration = max(lead_time, 1)
    constructing = ((start <= year) & (year < opening)) if lead_time else (opening == year)
    annual = np.where(scheduled & constructing, total / duration, 0.0)
    pre_years = np.clip(1 - start, 0, lead_time)
    upfront = np.where(scheduled & (year == 1), total * pre_years / duration, 0.0)
    return ((float(annual), float(upfront)) if opening.ndim == 0 else (annual, upfront))


def _describe_plan(spec: dict) -> str:
    parts = []
    if spec["initial_stage"]:
        parts.append(f"{stages.STATE_LABELS[spec['initial_stage']]} from Year 1")
    for key, package in (("to1", "Stations"), ("to2", "Tunnel")):
        transition = spec[key]
        if transition is None:
            continue
        if transition["type"] == "fixed":
            parts.append(f"{package} in Year {transition['year']}")
        else:
            parts.append(f"{package} via {transition['name']}")
    return ", ".join(parts) if parts else f"Baseline throughout the {N_YEARS}-year horizon"


def get_plans(params: dict | None = None, triggers: dict | None = None) -> dict:
    """Translate named package plans into runner fields, text, and package costs."""
    if params is None:
        params = NOMINAL_PARAMS
    triggers = {**get_triggers(params), **(triggers or {})}
        
    c_inv1 = params["C_INV_STAGE1"]
    c_inv2 = params["C_INV_STAGE2"]
    # These are already annual costs. The plan runner books them exactly once
    # per operating year; peak-to-annual demand factors must not touch OPEX.
    # OPEX_RATE applies to base construction costs, before CAPEX_MULTIPLIER.
    opex_rate = params["OPEX_RATE"]
    c_op1 = c_inv1 * opex_rate
    c_op2 = c_inv2 * opex_rate
    c_flex = params["C_FLEX"]

    plans = {}
    for key, definition in deepcopy(PLAN_DEFINITIONS).items():
        initial_packages = definition["initial_packages"]
        if (not isinstance(initial_packages, (list, tuple))
                or any(package not in ("stations", "tunnel") for package in initial_packages)
                or len(set(initial_packages)) != len(initial_packages)):
            raise ValueError(f"Plan '{key}': initial_packages must list 'stations' and/or 'tunnel' once each.")
        spec = {
            "name": definition["name"],
            "initial_stage": stages.state_for_components(
                "stations" in initial_packages, "tunnel" in initial_packages
            ),
        }
        for package, transition_name in (("stations", "to1"), ("tunnel", "to2")):
            transition = definition[package]
            if package in initial_packages and transition is not None:
                raise ValueError(f"Plan '{key}': {package} is initially open and cannot also have an opening rule.")
            if transition and transition.get("type") == "fixed":
                transition = {"type": "fixed", "year": transition["opening_year"],
                              **({"lead_time": transition["lead_time"]} if "lead_time" in transition else {})}
            if transition and transition.get("type") == "trigger":
                trigger_key = transition.get("trigger")
                if trigger_key is not None:
                    rule = triggers.get(trigger_key)
                    if rule is None:
                        raise ValueError(f"Plan '{key}': unknown trigger '{trigger_key}'.")
                    transition = {**rule, **transition}
                transition = _describe_trigger(
                    transition, transition.get("name", transition_name)
                )
            spec[transition_name] = transition
        _validate_plan(spec, key)
        spec["description"] = _describe_plan(spec)
        spec["name"] += " — " + spec["description"]
        initial_stage = spec["initial_stage"]
        stations_initial, tunnel_initial = stages.stage_components(initial_stage)
        option_premium_stage1 = (
            c_flex
            if spec["to1"] is not None and spec["to1"]["type"] == "trigger"
            else 0.0
        )
        option_premium_stage2 = (
            c_flex
            if spec["to2"] is not None and spec["to2"]["type"] == "trigger"
            else 0.0
        )

        # Derive each plan's cost/trigger bookkeeping fields from its raw
        # to1/to2 definition: which stage(s) it starts with or is entitled to
        # build, what the flexibility option costs, and the CAPEX/OPEX it
        # will draw on for stages it can reach (immediately or via a trigger).
        spec.update({
            "stage2_adaptive": (
                spec["to2"] is not None and spec["to2"]["type"] == "trigger"
            ),
            "stage2_year": (
                spec["to2"]["year"]
                if spec["to2"] is not None and spec["to2"]["type"] == "fixed"
                else None
            ),
            "option_premium_stage1": option_premium_stage1,
            "option_premium_stage2": option_premium_stage2,
            "upfront_fee": option_premium_stage1 + option_premium_stage2,
            "inv_stage1": (
                c_inv1 if stations_initial or spec["to1"] is not None else 0.0
            ),
            "op_stage1": (
                c_op1 if stations_initial or spec["to1"] is not None else 0.0
            ),
            "inv_stage2": (
                c_inv2 if tunnel_initial or spec["to2"] is not None else 0.0
            ),
            "op_stage2": (
                c_op2 if tunnel_initial or spec["to2"] is not None else 0.0
            ),
        })
        plans[key] = spec

    return plans


def get_reference_plan(stage: int, params: dict | None = None) -> dict:
    """Return a constant-state plan for annual lookup construction."""
    params = NOMINAL_PARAMS if params is None else params
    stations_active, tunnel_active = stages.stage_components(stage)
    costs = get_plans(params)["static2"]
    return {
        "name": stages.STATE_LABELS[stage],
        "initial_stage": stage, "to1": None, "to2": None,
        "upfront_fee": 0.0,
        "inv_stage1": costs["inv_stage1"] if stations_active else 0.0,
        "op_stage1": costs["op_stage1"] if stations_active else 0.0,
        "inv_stage2": costs["inv_stage2"] if tunnel_active else 0.0,
        "op_stage2": costs["op_stage2"] if tunnel_active else 0.0,
    }


# Module-level plan defaults for convenience across notebooks
TRIGGER_1, TRIGGER_2 = (get_triggers()[key] for key in ("trigger1", "trigger2"))
PLANS = get_plans(NOMINAL_PARAMS)
PLAN_NAMES = list(PLANS.keys())


def validate_case_study_configuration() -> None:
    """Check the cross-file contract students are most likely to break.

    Two independently operating packages produce four transport states.
    """
    stage_ids = set(map(int, stages.get_stages(NOMINAL_PARAMS)))
    missing = sorted(set(stages.STATE_IDS) - stage_ids)
    if missing:
        raise ValueError(
            f"The independent-package template requires states {stages.STATE_IDS}; "
            f"missing {missing}. Restore those IDs in stages.py or redesign the "
            "package-state mapping explicitly."
        )
    # Rebuild from student configuration so edits are checked rather than an old
    # module-level snapshot. to1/to2 target packages, not transport-state IDs.
    get_plans(NOMINAL_PARAMS)


validate_case_study_configuration()




# =============================================================================
# 3. SHAPED UNCERTAINTY TRAJECTORIES
# =============================================================================

# The interpolation shapes a 40-year structural uncertainty can follow between
# its Year-1 and Year-40 nominal values: "linear" (constant rate), "early"
# (front-loaded change), "late" (back-loaded change), "logistic" (S-curve
# transition), "random_walk" (irregular progress toward the endpoint),
# "almost_flat" (most change occurs near the end).
SHAPES = uc.SHAPES

def generate_shaped_trajectory(uncertainty_name: str, u: float | None = None,
                               shape: str = "linear") -> np.ndarray:
    """Use the shared distribution and trajectory configuration."""
    return uc.get_trajectory(uncertainty_name, u, shape=shape)


def demand_growth_trajectory(u: float | None = None, shape: str = "linear") -> np.ndarray:
    return uc.get_trajectory("PASSENGER_DEMAND_GROWTH", u, shape=shape)


def pt_asc_shift_trajectory(u: float | None = None, shape: str = "linear") -> np.ndarray:
    return uc.get_trajectory("PT_ASC_SHIFT", u, shape=shape)


# =============================================================================
# 4. PLAN EVALUATION AND PATHWAY GENERATION
# =============================================================================

def run_plan(plan_def: str | dict, stage_metrics: dict, params: dict | None = None,
             context=None, mode_results_by_stage: dict | None = None,
             corridor_municipalities: list | None = None, include_welfare: bool = False,
             transport_emulator=None, shapes: dict | None = None,
             return_details: bool = False, **draws) -> tuple[pd.DataFrame, dict]:
    """Evaluate configured paths; return_details adds nested transport diagnostics."""
    params = NOMINAL_PARAMS if params is None else params
    paths = uc.build_paths(draws, shapes=shapes, nominal=params)
    return run_plan_from_trajectories(
        plan_def, stage_metrics, params=params, extra_trajectories=paths,
        context=context, mode_results_by_stage=mode_results_by_stage,
        corridor_municipalities=corridor_municipalities,
        include_welfare=include_welfare, transport_emulator=transport_emulator,
        return_details=return_details,
    )


def run_plan_from_trajectories(plan_def: str  | dict, stage_metrics: dict, g_traj: np.ndarray | None = None,
                                   params: dict | None = None,
                                   pt_asc_shift_traj: np.ndarray | None = None, cost_traj: np.ndarray | None = None,
                                   capex_traj: np.ndarray | None = None,
                                   bike_asc_shift_traj: np.ndarray | None = None,
                                   road_freight_growth_traj: np.ndarray | None = None,
                                   extra_trajectories: dict[str, np.ndarray] | None = None,
                                   tt_pt_traj: np.ndarray | None = None, tt_car_traj: np.ndarray | None = None, co2_traj: np.ndarray | None = None,
                                   context=None, mode_results_by_stage: dict | None = None, corridor_municipalities: list | None = None,
                                   include_welfare: bool = False,
                                   transport_emulator=None, return_details: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Same as run_plan(), but takes ready-made demand and optional PT ASC-shift trajectories.
    co2_traj supplies reference-year carbon values in constant 2019 CHF/tonne;
    the appraisal applies calendar-year carbon-value growth to each entry.
    """
    if params is None:
        params = m.NOMINAL_PARAMS
        
    if isinstance(plan_def, str):
        plan_def = get_plans(params)[plan_def]
    else:
        plan_def = deepcopy(plan_def)
        # Preserve the historical defaults for raw custom plans. An explicit
        # signpost must match a result field; it never falls back to travel time.
        for key, signpost in (("to1", "pt_trips"), ("to2", "avg_tt_min")):
            transition = plan_def[key]
            if transition and transition.get("type") == "trigger":
                transition.setdefault("signpost", signpost)
    _validate_plan(plan_def, plan_def.get("name", "custom"))
    for key in ("to1", "to2"):
        validate_section_signpost(plan_def[key], params, label=f"Plan {plan_def.get('name', 'custom')}, {key}")
    
    to1, to2 = plan_def["to1"], plan_def["to2"]
    # Raw custom plans may omit get_plans()'s derived fee. Each adaptive package
    # still purchases an option; an explicit upfront_fee remains authoritative.
    upfront_fee = plan_def.get("upfront_fee", params["C_FLEX"] * sum(
        bool(transition and transition.get("type") == "trigger") for transition in (to1, to2)
    ))

    paths = uc.nominal_paths(nominal=params)
    paths.update(extra_trajectories or {})
    explicit = {
        "PASSENGER_DEMAND_GROWTH": g_traj, "PT_ASC_SHIFT": pt_asc_shift_traj,
        "BIKE_ASC_SHIFT": bike_asc_shift_traj, "ROAD_FREIGHT_GROWTH": road_freight_growth_traj,
        "OPEX_MULTIPLIER": cost_traj, "CAPEX_MULTIPLIER": capex_traj,
        "C_TT_PT": tt_pt_traj, "C_TT_CAR": tt_car_traj, "CO2_VALUE_CHF_PER_TONNE": co2_traj,
    }
    for name, path in explicit.items():
        if path is not None:
            paths[name] = np.asarray(path, dtype=float)
    paths = uc.clip_transport_paths(paths)
    g_traj = uc.target_path(paths, "PASSENGER_DEMAND_GROWTH")
    cost_traj = uc.target_path(paths, "OPEX_MULTIPLIER")
    capex_traj = uc.target_path(paths, "CAPEX_MULTIPLIER")

    stage = plan_def["initial_stage"]
    stations_initial, tunnel_initial = stages.stage_components(stage)
    stations_active, tunnel_active = stations_initial, tunnel_initial
    decision1_year = activation1_year = None
    decision2_year = activation2_year = None
    persist1 = persist2 = 0
    prev_metrics = {}
    # Actual openings differ from scheduled activation years when construction
    # would finish beyond the horizon. Residuals use only commissioned assets.
    opening_years = {1: 1 if stations_initial else None, 2: 1 if tunnel_initial else None}
    capital_paid = {1: 0.0, 2: 0.0}
    construction_leads = {package: construction_lead_time(plan_def, package, params) for package in (1, 2)}

    records = []

    for t in range(N_YEARS):
        year = t + 1

        # ---- fixed-year transitions (deterministic, no lag needed) --------
        if to1 and to1["type"] == "fixed" and not stations_active and year == to1["year"]:
            stations_active = True
            activation1_year = year
        if to2 and to2["type"] == "fixed" and not tunnel_active and year == to2["year"]:
            tunnel_active = True
            activation2_year = year

        # ---- triggered transitions (based on the PREVIOUS year's signpost) --
        if to1 and to1["type"] == "trigger":
            val1 = (_previous_signpost(prev_metrics, to1, "Station-package trigger")
                    if not stations_active and activation1_year is None else None)
            stations_active, opening, persist1, decided, _ = advance_trigger(
                year, stations_active, activation1_year or 0, persist1, val1, to1)
            stations_active, persist1 = bool(stations_active), int(persist1)
            activation1_year = int(opening) or None
            if decided:
                decision1_year = year

        if to2 and to2["type"] == "trigger":
            val2 = (_previous_signpost(prev_metrics, to2, "Tunnel-package trigger")
                    if not tunnel_active and activation2_year is None else None)
            tunnel_active, opening, persist2, decided, _ = advance_trigger(
                year, tunnel_active, activation2_year or 0, persist2, val2, to2)
            tunnel_active, persist2 = bool(tunnel_active), int(persist2)
            activation2_year = int(opening) or None
            if decided:
                decision2_year = year
        stage = stages.state_for_components(stations_active, tunnel_active)

        # ---- simulate this year at the resulting stage ---------------------
        base_m = stage_metrics.get(stage, {})
        mode_result = mode_results_by_stage.get(stage) if mode_results_by_stage else None
        transport_state = None
        
        year_params = uc.year_parameters(paths, t, params=params)
        
        cost_mult = float(cost_traj[t])
        capex_mult = float(capex_traj[t])
        def plan_cost(key, target):
            reference = float(params[target])
            configured = float(plan_def.get(key, 0.0))
            return (configured * float(year_params[target]) / reference
                    if reference else float(year_params[target]))

        def plan_operating_cost(package):
            target = f"C_INV_STAGE{package}"
            reference = float(params[target]) * float(params["OPEX_RATE"])
            current = float(year_params[target]) * float(year_params["OPEX_RATE"])
            configured = float(plan_def.get(f"op_stage{package}", 0.0))
            return configured * current / reference if reference else current

        inv_this_year = 0.0
        # Buy the option at time zero even if it is never exercised. Keep the
        # premium separate from annual CAPEX so only it escapes Year-1 discounting.
        upfront_this_year = (upfront_fee * (year_params["C_FLEX"] / params["C_FLEX"] if params["C_FLEX"] else 1.0)
                             * capex_mult if year == 1 else 0.0)
        for package, initially_active, activation_year in (
            (1, stations_initial, activation1_year),
            (2, tunnel_initial, activation2_year),
        ):
            if (year == 1 and initially_active) or activation_year == year:
                paid = plan_cost(f"inv_stage{package}", f"C_INV_STAGE{package}") * capex_mult
                inv_this_year += paid
                capital_paid[package] += paid
                opening_years[package] = year
        
        op_this_year = (
            (plan_operating_cost(1) if stations_active else 0.0)
            + (plan_operating_cost(2) if tunnel_active else 0.0)
        ) * cost_mult

        if transport_emulator is not None:
            # Query the surrogate using this year's configured physical inputs.
            physical_inputs = uc.transport_surrogate_inputs_from_paths(paths, t)
            transport_state = transport_emulator.predict_transport_state(
                stage=stage,
                include_links=False,
                include_welfare=include_welfare,
                **physical_inputs,
            )
            base_m = transport_state["metrics"]

        row = m.simulate_year(
            base_m, stage, t, g_traj[t], year_params,
            context=context, mode_result=mode_result, corridor_municipalities=corridor_municipalities,
            inv_cost=inv_this_year,
            op_cost=op_this_year,
            flex_cost=upfront_this_year,
            return_details=return_details,
            include_welfare=include_welfare,
            transport_state=transport_state,
        )

        annual_construction, upfront_construction = 0.0, 0.0
        for package, initially_active, transition, activation in (
            (1, stations_initial, to1, activation1_year),
            (2, tunnel_initial, to2, activation2_year),
        ):
            opening = (1 if initially_active else transition["year"]
                       if transition and transition["type"] == "fixed" else activation or 0)
            annual, upfront = construction_emissions_in_year(
                package, year, opening, construction_leads[package])
            annual_construction += annual
            upfront_construction += upfront
        row = m.add_construction_emissions(row, annual_construction, upfront_construction, params=year_params)

        # Re-expose peak-hour outputs from simulate_year() alongside a few
        # convenience fields (annualized trip counts, total corridor travel
        # time) that downstream notebooks read directly from the trajectory
        # dataframe rather than recomputing.
        row.update({
            "year": year,
            "appraisal_residual_version": m.APPRAISAL_RESIDUAL_VERSION,
            "discount_rate": float(year_params["DISCOUNT_RATE"]),
            "stage": stage,
            "stations_active": stations_active,
            "tunnel_active": tunnel_active,
            "car_share": row.get("car_share", 0.0),
            "pt_share": row.get("pt_share", 0.0),
            "bike_share": row.get("bike_share", 0.0),
            "walk_share": row.get("walk_share", 0.0),
            "car_share_trips": row.get("car_share_trips", 0.0),
            "pt_share_trips": row.get("pt_share_trips", 0.0),
            "bike_share_trips": row.get("bike_share_trips", 0.0),
            "walk_share_trips": row.get("walk_share_trips", 0.0),
            "pt_trips": row.get("pt_trips", 0.0),
            "annual_pt_trips": row.get("pt_trips", 0.0) * m.annualizers(year_params)["pt"],
            "car_trips": row.get("car_trips", 0.0),
            "annual_car_trips": row.get("car_trips", 0.0) * m.annualizers(year_params)["car"],
        })
        # Credit the remaining asset value once, at the end of the horizon.
        # The investment basis includes the capital multiplier actually paid,
        # excludes option premiums/OPEX, and ages from the actual opening year.
        for package in (1, 2):
            opening = opening_years[package]
            row[f"residual_value_stage{package}"] = (
                stages.asset_residual_value(package, capital_paid[package], N_YEARS - opening + 1)
                if year == N_YEARS and opening is not None else 0.0
            )
        row["residual_value"] = row["residual_value_stage1"] + row["residual_value_stage2"]
        row["total_cost"] -= row["residual_value"]
        if "total" in row:
            row["total"] -= row["residual_value"]
        records.append(row)

        prev_metrics = dict(row)

    df = pd.DataFrame(records)
    meta = {
        "decision1_year": decision1_year, "activation1_year": activation1_year,
        "decision2_year": decision2_year, "activation2_year": activation2_year,
        "opening1_year": opening_years[1], "opening2_year": opening_years[2],
        "capital_paid_stage1": capital_paid[1], "capital_paid_stage2": capital_paid[2],
    }
    return df, meta



# =============================================================================
# 5. PERFORMANCE REQUIREMENT, TIPPING POINTS, OPPORTUNITY POINTS
# =============================================================================

def is_acceptable_year(avg_tt_min: float, pt_share: float, max_avg_tt: float | None = None, pt_share_target: float | None = None, bike_share: float = 0.0, bike_share_target: float | None = None) -> bool:
    """True if this year's indicators meet all configured policy targets
    (travel time at/under the ceiling, mode shares at/over their targets)."""
    if max_avg_tt is None:
        max_avg_tt = MAX_AVG_TT
    if pt_share_target is None:
        pt_share_target = PT_SHARE_TARGET
        
    acceptable = (avg_tt_min <= max_avg_tt) and (pt_share >= pt_share_target)
    if bike_share_target is not None:
        acceptable = acceptable and (bike_share >= bike_share_target)
        
    return acceptable


def _stage_forever_trajectory(stage: int, stage_metrics: dict, g_traj: np.ndarray,
                               params: dict | None = None,
                               context=None, mode_results_by_stage: dict | None = None,
                               corridor_municipalities: list | None = None,
                               transport_emulator=None) -> pd.DataFrame:
    """Simulate one infrastructure stage held fixed for the whole horizon."""
    trajectory, _ = run_plan_from_trajectories(
        get_reference_plan(stage, params), stage_metrics, g_traj, params,
        context=context, mode_results_by_stage=mode_results_by_stage,
        corridor_municipalities=corridor_municipalities,
        transport_emulator=transport_emulator,
    )
    return trajectory


def find_tipping_point(stage: int, stage_metrics: dict, g_traj: np.ndarray,
                        params: dict | None = None,
                        context=None, mode_results_by_stage: dict | None = None,
                        corridor_municipalities: list | None = None,
                        transport_emulator=None) -> int | None:
    """Adaptation tipping point for one stage."""
    if params is None:
        params = m.NOMINAL_PARAMS
    max_tt = params.get("MAX_AVG_TT", MAX_AVG_TT)
    pt_target = params.get("PT_SHARE_TARGET", PT_SHARE_TARGET)
    if max_tt is None:
        max_tt = MAX_AVG_TT
    if pt_target is None:
        pt_target = PT_SHARE_TARGET
    
    bike_target = params.get("BIKE_SHARE_TARGET")
    df = _stage_forever_trajectory(stage, stage_metrics, g_traj, params,
                                   context=context, mode_results_by_stage=mode_results_by_stage,
                                   corridor_municipalities=corridor_municipalities,
                                   transport_emulator=transport_emulator)
    failing = df[~df.apply(lambda r: is_acceptable_year(r.get("avg_tt_min", 0), r.get("pt_share", 0), max_tt, pt_target, r.get("bike_share", 0), bike_target), axis=1)]
    return int(failing.iloc[0]["year"]) if not failing.empty else None


def find_opportunity_point(stage_from: int, stage_to: int, stage_metrics: dict, g_traj: np.ndarray,
                            params: dict | None = None, benefit_margin: float = 2.0,
                            context=None, mode_results_by_stage: dict | None = None,
                            corridor_municipalities: list | None = None,
                            transport_emulator=None) -> int | None:
    """Opportunity point for upgrading stage_from -> stage_to.
    Checks if avg_tt_min drops by at least benefit_margin (e.g., 2 mins)."""
    df_from = _stage_forever_trajectory(stage_from, stage_metrics, g_traj, params,
                                        context=context, mode_results_by_stage=mode_results_by_stage,
                                        corridor_municipalities=corridor_municipalities,
                                        transport_emulator=transport_emulator)
    df_to = _stage_forever_trajectory(stage_to, stage_metrics, g_traj, params,
                                      context=context, mode_results_by_stage=mode_results_by_stage,
                                      corridor_municipalities=corridor_municipalities,
                                      transport_emulator=transport_emulator)
    relief = df_from["avg_tt_min"].values - df_to["avg_tt_min"].values
    crossing = np.where(relief >= benefit_margin)[0]
    return int(crossing[0] + 1) if len(crossing) else None
