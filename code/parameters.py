"""
parameters.py
=============
Model inputs, corridor definitions and assumptions for the SBB MehrSpur
Zürich–Winterthur infrastructure analysis.

HOW TO USE
----------
Import assumptions and corridor definitions wherever the model needs them:

    from parameters import CORRIDOR_REGIONS, N_YEARS, DISCOUNT_RATE, ...

STUDENT INSTRUCTIONS
--------------------
This file is configured for the SBB MehrSpur project. For your own project
(e.g., a bike highway or mobility hub), adapt these three sections:
1. General parameters: set nominal values, shared appraisal assumptions,and
   physical targets. Stages and their package-specific parameters 
   are defined together in stages.py. A KEY_Y40 value defines that parameter's
   nominal endpoint.
2. Corridor: redefine CORRIDOR_REGIONS/CORRIDOR_MUNICIPALITIES or use explicit
   zone IDs. Configure the counting section and any supplementary flow.
3. Uncertainties: use the same general parameter key, classify its effect as
   transport_modelling or post_modelling, and choose scenarios or sensitivity.
   Removing an uncertainty keeps its general nominal trajectory.
4. Rebuild the surrogate (nor notebooks 03 onwards) after changing the corridor, physical stages,
   transport behaviour, or a transport input or its training domain.

See code/surrogate_model/README.md for the adaptation checklist.
"""
from additional import uncertainty as uc
from stages import package_parameter_defaults

# =============================================================================
# 1. GENERAL PARAMETERS (STUDENT EDITABLE)
# =============================================================================

NOMINAL_PARAMS = {
    # -------------------------------------------------------------------------
    # Appraisal calendar and discounting
    # -------------------------------------------------------------------------
    "N_YEARS": 40,  # Appraisal horizon (years); *_Y40 denotes its final-year endpoint.
    "APPRAISAL_START_YEAR": 2026,  # Calendar year represented by Year 1 of the appraisal.
    "DISCOUNT_RATE": 0.02,  # Annual social discount rate; 0.02 means 2%.

    # -------------------------------------------------------------------------
    # Passenger demand, transport preferences, cycling and background traffic
    # Present-day values; *_Y40 values define the nominal endpoints.
    # -------------------------------------------------------------------------
    "PASSENGER_DEMAND_GROWTH": 0.0,  # Cumulative passenger OD growth relative to baseline; 0.20 means +20%.
    "PASSENGER_DEMAND_GROWTH_Y40": 0.20,  # Nominal final passenger-demand growth.
    "PT_ASC_SHIFT": 0.0,  # Additive preference shift for both PT access modes (utility units, not percent).
    "PT_ASC_SHIFT_Y40": 0.10,  # Nominal final PT preference shift (utility units).
    "BIKE_ASC_SHIFT": 0.0,  # Additive preference shift for standalone cycling (utility units).
    # "BIKE_ASC_SHIFT_Y40": 0.20,  # Optional nominal final cycling preference shift (utility units).
    "EBIKE_SHARE": 0.25,  # Baseline e-bike fraction of cycling and PT bicycle access/egress trips (0 to 1).
    "EBIKE_SHARE_Y40": 0.50,  # Nominal final e-bike share.
    "EBIKE_SPEED_MULTIPLIER": 1.5,  # E-bike speed divided by conventional-bike speed; 1.5 means 50% faster.
    "ROAD_FREIGHT_GROWTH": 0.0,  # Growth in prepared commercial/freight vehicles and their synthetic uplift.
    "ROAD_FREIGHT_GROWTH_Y40": 0.10,  # Nominal final growth of prepared background road vehicles.

    # -------------------------------------------------------------------------
    # Peak-hour to annual conversions
    # -------------------------------------------------------------------------
    # National NPVM processing ratios, not measured local hourly counts.
    # Cycling/walking use the processor's teaching assumption; no off-peak solve is run.
    "PT_PEAK_SHARE": 0.0946,  # Representative peak-hour PT trips / weekday PT trips; NPVM conversion assumption.
    "CAR_PEAK_SHARE": 0.1049,  # Representative peak-hour car trips / weekday car trips; NPVM conversion assumption.
    "BIKE_PEAK_SHARE": 0.15,  # Peak-hour share of daily cycling trips; editable teaching assumption.
    "WALK_PEAK_SHARE": 0.15,  # Peak-hour share of daily walking trips; editable teaching assumption.
    "EQUIVALENT_DAYS_PER_YEAR": 300,  # Equivalent operating weekdays represented in one appraisal year.
    "NUMBER_OF_PEAK_HOURS": 4,  # Hours/day exposed to modeled congestion and crowding; integer sensitivity.
    # Ordinary journey time applies all day; congestion and crowding apply only
    # during the configured peak hours. The engine derives conversions from days
    # and each mode's peak share.
    "PEAK_TO_ANNUAL_TRAIN": 3391.5,  # Peak-hour supplied train-km to annual train-km; independent of passenger demand.

    # -------------------------------------------------------------------------
    # PT service, section comfort capacity and train supply
    # -------------------------------------------------------------------------
    "PT_HEADWAY_BASELINE": 20.0,  # Baseline interval between PT services (minutes).
    # Package headway reductions and capacity increases are configured in stages.py.
    # Section comfort thresholds (persons/peak hour).
    # Adopted 120,000-passenger daily count * PT peak share; thresholds do not grow with demand.
    "PT_CAPACITY_BASELINE": 11_352.0,  # Fixed baseline section comfort threshold (persons/peak hour).
    "CROWDING_SLOPE": 2.0,  # Extra time-penalty slope above Q/C = 1 in min(CROWDING_MAX, 1 + slope*max(Q/C-1, 0)).
    "CROWDING_MAX": 3.0,  # Maximum total multiplier on section in-vehicle time; only the excess over 1 is added.
    "TRAIN_GROSS_TONNES": 375.0,  # Reference gross tonnes per train for rail external costs and energy consumption.
    # Fixed Stage-0 peak-hour train supply, calibrated once from the nominal
    # Stage-0 model (512,188 passenger-km / 150 passengers per train). Keeping
    # it independent of ridership prevents mode shift from creating fictitious
    # trains in the external-cost calculation.
    "PT_TRAIN_KM_PEAK_STAGE0": 3_414.6,  # Fixed baseline supplied train-km/peak hour; higher ridership alone adds no trains.

    # -------------------------------------------------------------------------
    # Values of passenger time (CHF/person-hour)
    # -------------------------------------------------------------------------
    "C_TT_CAR": 42.54,  # Car travel-time value, including peak congestion delay (CHF/person-hour).
    "C_TT_PT": 26.52,  # PT in-vehicle time value (CHF/person-hour).
    "C_TT_PT_WAITING": 26.52,  # Initial and transfer waiting-time value (CHF/person-hour).
    "C_TT_PT_ACCESS": 26.52,  # PT access and egress walking/cycling time value (CHF/person-hour).
    "C_TT_PT_TRANSFER": 26.52,  # Physical transfer-walking time value (CHF/person-hour); excludes waiting.
    # Active-mode time is valued per person-hour, like PT/car passenger time.
        "C_TT_BIKE": 35.0,  # Cycling value of time (CHF/person-hour)
    "C_TT_WALK": 16.0,  # Walking value of time (CHF/person-hour)

    # -------------------------------------------------------------------------
    # Optional additional benefits (uncomment to include in appraisal)
    # -------------------------------------------------------------------------
    # Health rates apply to standalone cycling/walking person-km, including e-bikes.
    # The PT rate applies once per person-trip, including supplementary PT passengers.
    # Positive rates reduce societal costs; omitted rates default to zero.
    # "BENEFIT_HEALTH_BIKE_PER_KM": 1.0,  # Cycling health benefit (CHF/person-km).
    # "BENEFIT_HEALTH_WALK_PER_KM": 1.0,  # Walking health benefit (CHF/person-km).
    # "BENEFIT_SOCIOECONOMIC_PT_PER_TRIP": 1.0,  # Additional PT benefit (CHF/person-trip).

    # -------------------------------------------------------------------------
    # Physical CO2 emissions and their common monetary value
    # -------------------------------------------------------------------------
    "CAR_CO2_KG_PER_VEHICLE_KM": 0.139,  # Car emissions (kg CO2/vehicle-km).
    "CAR_CO2_KG_PER_VEHICLE_KM_Y40": 0.050,  # Nominal final car emissions (kg CO2/vehicle-km).
    "RAIL_ENERGY_WH_PER_GROSS_TONNE_KM": 30.0,  # Rail electricity consumption (Wh/gross-tonne-km).
    "RAIL_ELECTRICITY_CO2_G_PER_KWH": 19.58,  # Rail electricity emissions (g CO2/kWh).
    # Both road and rail emissions use this carbon value. Its reference-year
    # value is in constant 2019 CHF; annual real growth is compounded from the
    # reference year to APPRAISAL_START_YEAR + the zero-based appraisal year.
    "CO2_VALUE_CHF_PER_TONNE": 123.2,  # Reference-year CO2 value (constant 2019 CHF/tonne CO2).
    "CO2_REFERENCE_YEAR": 2015,  # Calendar year to which the reference CO2 value applies.
    "CO2_VALUE_ANNUAL_GROWTH": 0.03,  # Real annual CO2-value growth (fraction/year); 0.03 means 3% compounded.

    # -------------------------------------------------------------------------
    # Other NIBA external costs (road and passenger rail)
    # -------------------------------------------------------------------------
    "C_NOISE_CAR": 0.01490,  # Road noise cost (CHF/vehicle-km); course NIBA coefficient.
    "C_AIR_CAR": 0.01862,  # Road local air-pollution cost (CHF/vehicle-km); course NIBA coefficient.
    "C_ACCIDENT_CAR": 0.08441,  # Road accident cost (CHF/vehicle-km); course NIBA coefficient.
    "C_AIR_PT": 0.00344,  # Rail local air-pollution cost (CHF/gross-tonne-km); course NIBA coefficient.
    "C_NOISE_PT": 0.00204,  # Rail noise cost (CHF/gross-tonne-km); course NIBA coefficient.
    "C_ACCIDENT_PT": 0.22210,  # Rail accident cost (CHF/train-km); course NIBA coefficient.
    "ACCIDENT_COST_MULTIPLIER": 1.0,  # Multiplier on road and rail accident unit costs.
    "ACCIDENT_COST_MULTIPLIER_Y40": 0.85,  # Nominal final accident-cost factor; 0.85 means 15% lower unit costs.
    "LOCAL_AIR_COST_MULTIPLIER": 1.0,  # Multiplier on road and rail local air-pollution unit costs.
    "LOCAL_AIR_COST_MULTIPLIER_Y40": 0.65,  # Nominal final local air-pollution cost factor.

    # -------------------------------------------------------------------------
    # Shared operation, investment multipliers and flexibility options
    # -------------------------------------------------------------------------
    # Package-specific CAPEX is defined in stages.py, alongside each intervention.
    "OPEX_RATE": 0.02,  # Annual infrastructure operating/maintenance cost as a fraction of base construction cost.
    "OPEX_MULTIPLIER": 1.0,  # Multiplier on annual infrastructure operating/maintenance costs.
    "CAPEX_MULTIPLIER": 1.0,  # Multiplier on construction costs and upfront option premiums.
    # Flexibility Premium (for pre-engineering or reservation)
    "C_FLEX": 50_000_000,  # Upfront CHF per adaptive package; two options are paid at time zero even if unused.

    # -------------------------------------------------------------------------
    # Performance requirements and strategic mode-share indicators
    # -------------------------------------------------------------------------
    "MAX_AVG_TT": 15.0,  # Acceptability ceiling for peak car/PT in-vehicle time plus car delay (min/trip).
    "PT_SHARE_TARGET": 0.35,  # Teaching minimum for PT share of strategic passenger-km; baseline is about 31.9%.
    "MIN_STRATEGIC_PKM_DISTANCE": 5.0,  # Minimum OD distance for strategic mode-share indicators (km); not an appraisal cutoff.

    # Internal shared keys for appraisal and uncertainties; edit values in stages.py.
    **package_parameter_defaults(),
}

# Preserve convenient module-level imports without a second configuration block.
globals().update(NOMINAL_PARAMS)

# =============================================================================
# 2. CORRIDOR
# =============================================================================

# -----------------------------------------------------------------------------
# OPTION A: Define by Municipalities / Regions (Empty when in "zones" mode)
# -----------------------------------------------------------------------------

CORRIDOR_REGIONS = { # Named municipality groups defining the modeled corridor.
    "Limmattal": [
        "Zürich",
        "Schlieren",
        "Dietikon",
        "Unterengstringen",
    ]
}

# Flat list of all corridor municipalities
CORRIDOR_MUNICIPALITIES = [  # Derived flat list used by corridor selection and stage definitions.
    muni for munis in CORRIDOR_REGIONS.values() for muni in munis
]

# Optional project-relative or absolute detailed road-network cache path.
# Notebook 02 prepares a missing cache; later notebooks require it to exist.
# None uses the existing default cache. For a new project, select a new filename.
DETAILED_NETWORK_FILE = "data/processed/limmattal_altstetten_detailed_network.pkl"

# -----------------------------------------------------------------------------
# OPTION B: Define by Explicit Zone IDs
# -----------------------------------------------------------------------------
# Example: enable zone selection, define groups, then flatten their IDs.
# CORRIDOR_DEFINITION_MODE = "zones"
# PROJECT_ZONES = {
#     # 1. Zürich Eastern Alignment into CBD: Bellevue, Stadelhofen, Kreuzplatz, Römerhof,
#     #    Hegibachplatz, Burgwies, Balgrist, Rehalp (S18 border loop), and Witikon
#     "Zurich_S18_Alignment": [
#         "26101165", "26101169", "26101171", "26101173", "26101174",
#         "26101180", "26101181", "26101183", "26101185", "26101187",
#         "26101189", "26101190", "26101194", "26101195", "26101198",
#         "26101199", "26101201", "26101202", "26101203", "26101204",
#         "26101209", "26101210", "26101215", "26101216", "26101218",
#         "26101220", "26101221", "26101222", "26101225", "26101227",
#         "26101228", "26101229", "26101231", "26101234", "26101235",
#         "26101236", "26101242", "26101243", "26101245", "26101248",
#         "26101251", "26101252", "26101255", "26101259", "26101262",
#         "26101263", "26101264", "26101265", "26101267", "26101272",
#         "26101273", "26101274", "26101275", "26101278", "26101281",
#         "26101282", "26101283", "26101285", "26101287", "26101288",
#         "26101289", "26101290", "26101294", "26101295", "26101296",
#         "26101300", "26101301", "26101303", "26101304", "26101305",
#     ]

# }
# Flat list of active corridor zone IDs:
# CORRIDOR_ZONE_IDS = [z for zone_list in PROJECT_ZONES.values() for z in zone_list]
# -----------------------------------------------------------------------------

CORRIDOR_DEFINITION_MODE = "zones"

PROJECT_ZONES = {
    "Altstetten": [
        "26101001", "26101002", "26101003", "26101004",
        "26101006", "26101008", "26101009", "26101010",
        "26101012", "26101013", "26101015", "26101016",
        "26101017", "26101018", "26101019", "26101020",
        "26101021", "26101023", "26101028", "26101029",
        "26101030", "26101032", "26101033", "26101034",
        "26101037", "26101039", "26101041", "26101046",
        "26103306",
    ],
    "Schlieren": [
        f"247010{i:02d}" for i in range(1, 21)
    ],
    "Dietikon": [
        f"243010{i:02d}" for i in range(1, 28)
    ],
    "Unterengstringen": [
        "24901001", "24901002", "24901003"
    ],
}

CORRIDOR_ZONE_IDS = [
    z for zone_list in PROJECT_ZONES.values() for z in zone_list
]
# Optional route coverage, including trips with endpoints outside the corridor.
# Coverage is regenerated from the shared routing inputs when settings change.
# PT approaches use service connectivity, not physical railway geometries.
# Use `python code/additional/section_flows.py --help` for preparation/calibration.
# Omit SECTION and EXTERNAL_FLOW for projects that do not use this feature.
SECTION_DEFAULTS = {  # Generic fallback settings; configure the project's SECTION below.
    "active": False,                      # Enable route coverage and section-specific appraisal.
    "mode": "PT",                         # PT, CAR, BIKE or WALK
    "origin": {},                         # {"municipality_name": "X"} or {"zone_ids": [...]}
    "destination": {},                    # Destination selector for reference section travel time.
    "both_directions": True,              # Count both directions and use both reference OD directions.
    "coverage_file": None,                # None uses an automatically regenerated cache.
    "crowding_enabled": False,            # this version values crowding only for PT
}
SECTION = {  # Active project section and endpoints used for its travel-time reference.
    **SECTION_DEFAULTS,
    "active": True,                       # Include this count and its effects in model/appraisal outputs.
    "kind": "pt_approach",                # PT station approach, cycling route or individual PT stop.
    "station": "Winterthur",              # Station whose approach is counted.
    "towards": "Effretikon",              # Select the side facing this station, including express services.
    "coverage_file": "data/processed/section_coverage.npz",  # Supplied example; refreshed if inputs or definition change.
    "mode": "PT",                         # Mode using this section; must match an enabled external flow.
    "origin": {"municipality_name": "Zürich"},  # Origin of the modeled travel-time reference.
    "destination": {"municipality_name": "Winterthur"},  # Destination of that reference.
    "crowding_enabled": True,             # Value extra PT discomfort using the section comfort threshold.
}

# Alternative definitions (replace SECTION above; disable PT crowding for these).
# Disable EXTERNAL_FLOW below, or calibrate a new cohort of the matching mode:
# SECTION = {"active": True, "kind": "bike_route", "mode": "BIKE",
#            "origin_station": "Wallisellen", "destination_station": "Dübendorf"}
# A route reports its midpoint-link count and unique users of any project link.
# SECTION = {"active": True, "kind": "pt_stop", "mode": "PT", "station": "Wallisellen"}
# A stop reports access, egress and represented walking transfers separately.
# Same-platform train changes cannot all be identified from the routing inputs.

# An optional existing passenger flow, separate from modeled OD demand.
# Mode defaults to the selected section's mode. These passengers affect
# appraisal but not mode choice or road assignment.
# Enter ONE fixed baseline additional count, then grow it explicitly. The calibration
# script in NB02 reports a suggested residual without editing this configuration.
EXTERNAL_FLOW_DEFAULTS = {  # Generic fallback; configure the project's EXTERNAL_FLOW below.
    "enabled": False,                     # Include an existing supplementary cohort in appraisal.
    "mode": None,                         # Inherit SECTION.mode; no separate mode-choice response.
    "additional_trips_daily": 0.0,        # Existing person-trips/weekday missing from the retained model.
    "growth": "general",                  # "general" demand growth or "fixed"
}
EXTERNAL_FLOW = {  # One cohort shared by baseline and projects, added after mode choice.
    **EXTERNAL_FLOW_DEFAULTS,
    "enabled": True,                      # Include this cohort's time costs and PT comfort loading.
    "mode": "PT",                         # Existing PT passengers; must match SECTION.mode.
    # One-time nominal Stage-0 calibration (25% e-bikes; 2% road-gap target):
    # 120,000 target - 66,891.950467 modeled passengers/day, both directions.
    # Winterthur approach towards Effretikon; connectivity-refined road network.
    # Keep this cohort fixed across alternatives; only general demand growth applies.
    "additional_trips_daily": 53_108.0495332632,  # Baseline supplementary person-trips/day, both directions.
}


# =============================================================================
# 3. UNCERTAINTIES
# =============================================================================
# The key is the same general parameter used by the model and surrogate.
# category: transport_modelling changes travel; post_modelling changes appraisal.
# use: scenarios joins the future ensemble; sensitivity is tested separately.
# transient=True: draw the final-year value, then interpolate from the known
# general baseline. The nominal endpoint is KEY_Y40, or KEY if no endpoint is set.
# transient=False: one sampled value stays constant through the whole horizon.
# Removing an uncertainty retains its general nominal trajectory.
# Only supported model parameters can be registered; see code/README.md for the list.
# Normal uses sigma (a fraction when relative_sigma=True); uniform/triangular/
# beta use endpoint minimum/maximum. Triangular nominal is its mode. Beta also
# requires alpha and beta. Lognormal sigma is in log space.
# Uniform with integer=True assigns equal probability to each integer in its bounds.
# Advanced projects can add initial_distribution={...} for an uncertain baseline.
TRANSPORT_SURROGATE_TAIL_PROBABILITY = 0.002  # 0.1% in each unbounded marginal tail

STRUCTURAL_UNCERTAINTIES = {  # Distributions for selected general parameters; notebooks select by use.
    "PASSENGER_DEMAND_GROWTH": {  # Final cumulative passenger growth; -0.20 means 20% below baseline.
        "category": "transport_modelling",
        "use": "scenarios",
        "transient": True,
        "distribution": "triangular",
        "minimum": -0.20,
        "maximum": 0.50,
    },
    "PT_ASC_SHIFT": {  # Final additive PT preference shift; negative values reduce PT attractiveness.
        "category": "transport_modelling",
        "use": "scenarios",
        "transient": True,
        "distribution": "triangular",
        "minimum": -0.10,
        "maximum": 0.30,
    },
    # Uncomment together with BIKE_ASC_SHIFT_Y40 above to include cycling preferences.
    # "BIKE_ASC_SHIFT": {
    #     "category": "transport_modelling",
    #     "use": "scenarios",
    #     "transient": True,
    #     "distribution": "triangular",
    #     "minimum": -0.50,
    #     "maximum": 1.00,
    # },
    "EBIKE_SHARE": {  # Final e-bike fraction; starts from the known baseline share.
        "category": "transport_modelling",
        "use": "scenarios",
        "transient": True,
        "distribution": "triangular",
        "minimum": 0.35,
        "maximum": 0.80,
    },
    "ROAD_FREIGHT_GROWTH": {  # Final prepared-background growth; sigma is in growth-fraction units.
        "category": "transport_modelling",
        "use": "scenarios",
        "transient": True,
        "distribution": "normal",
        "sigma": 0.05,
    },
    "C_TT_PT": {  # PT value-of-time sensitivity; one CHF/person-hour draw for the whole horizon.
        "category": "post_modelling",
        "use": "sensitivity",
        "transient": False,
        "distribution": "uniform",
        "minimum": 19.89,
        "maximum": 33.15,
    },
    "C_TT_CAR": {  # Car value-of-time sensitivity, including congestion time.
        "category": "post_modelling",
        "use": "sensitivity",
        "transient": False,
        "distribution": "uniform",
        "minimum": 31.91,
        "maximum": 53.18,
    },
    "CO2_VALUE_CHF_PER_TONNE": {  # Draw the reference-year value once (constant 2019 CHF/tonne CO2).
        # Triangular mode is the general reference value; the engine applies
        # CO2_VALUE_ANNUAL_GROWTH from CO2_REFERENCE_YEAR to each appraisal year.
        "category": "post_modelling",
        "use": "sensitivity",
        "transient": False,
        "distribution": "triangular",
        "minimum": 70.0,
        "maximum": 217.0,
    },
    "CAR_CO2_KG_PER_VEHICLE_KM": {  # Final car emissions (kg CO2/vehicle-km); sigma is relative to its nominal endpoint.
        "category": "post_modelling",
        "use": "scenarios",
        "transient": True,
        "distribution": "normal",
        "sigma": 0.15,
        "relative_sigma": True,
    },
    "NUMBER_OF_PEAK_HOURS": {  # Sensitivity: equally likely 3, 4 or 5 peak hours, constant over time.
        "category": "post_modelling",
        "use": "sensitivity",
        "transient": False,
        "distribution": "uniform",
        "minimum": 3,
        "maximum": 5,
        "integer": True,
    },
    "OPEX_MULTIPLIER": {  # Operating-cost multiplier, sampled once per future.
        "category": "post_modelling",
        "use": "scenarios",
        "transient": False,
        "distribution": "uniform",
        "minimum": 0.50,
        "maximum": 1.50,
    },
    "CAPEX_MULTIPLIER": {  # Construction/option-cost multiplier; sampled once per future.
        "category": "post_modelling",
        "use": "scenarios",
        "transient": False,
        "distribution": "triangular",
        "minimum": 0.80,
        "maximum": 1.20,
    },
    "ACCIDENT_COST_MULTIPLIER": {  # Final road/rail accident-cost factor.
        "category": "post_modelling",
        "use": "scenarios",
        "transient": True,
        "distribution": "normal",
        "sigma": 0.15,
        "relative_sigma": True,
    },
    "LOCAL_AIR_COST_MULTIPLIER": {  # Final road/rail local air-pollution cost factor.
        "category": "post_modelling",
        "use": "scenarios",
        "transient": True,
        "distribution": "normal",
        "sigma": 0.15,
        "relative_sigma": True,
    },
}

# Validate settings and prepare internal surrogate metadata.
uc.configure(
    registry=STRUCTURAL_UNCERTAINTIES, nominal=NOMINAL_PARAMS,
    n_years=N_YEARS, tail_probability=TRANSPORT_SURROGATE_TAIL_PROBABILITY,
)
