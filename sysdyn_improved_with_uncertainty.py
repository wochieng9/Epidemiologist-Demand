"""
Kenya Field Epidemiology Workforce System Dynamics Template

This Streamlit app is a decision-support template for estimating the demand,
supply, effective capacity, and workforce gap for field epidemiologists in Kenya.

It intentionally separates:
1. trained headcount,
2. active/protected epidemiology time,
3. effective tier-weighted capacity,
4. risk- and workload-adjusted demand, and
5. fiscal/training delays.

Defaults are illustrative. Replace them with K-FELTP, MOH HRH, county payroll,
PHEOC, IDSR, DHIS2, and outbreak line-list data before using results for policy.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from uncertainty_tools_improved import streamlit_uncertainty_panel
except ImportError:
    streamlit_uncertainty_panel = None

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Kenya Field Epidemiology Workforce Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Core dimensions and model constants
# -----------------------------------------------------------------------------

TIERS = ["Frontline", "Intermediate", "Advanced"]
TIER_KEYS = {"Frontline": "frontline", "Intermediate": "intermediate", "Advanced": "advanced"}

AGE_STARTS = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65], dtype=float)
AGE_MIDS = AGE_STARTS + 2.5
AGE_LABELS = [f"{int(a)}-{int(a + 4)}" for a in AGE_STARTS]
COHORT_WIDTH_YEARS = 5.0
DT = 1.0 / 12.0

# Approximate annual mortality by age band. Mortality is tiny relative to other
# workforce flows, but keeping it explicit helps avoid immortal cohorts.
AGE_MORTALITY = np.array([0.002, 0.002, 0.003, 0.004, 0.006, 0.009, 0.013, 0.020, 0.030])

# Relative contribution to effective advanced-equivalent epidemiology capacity.
DEFAULT_SKILL_WEIGHT = {
    "frontline": 0.35,
    "intermediate": 0.70,
    "advanced": 1.00,
}

# Age distribution of graduates entering each tier.
GRAD_AGE_MEAN_SD = {
    "frontline": (34.0, 8.0),
    "intermediate": (38.0, 8.5),
    "advanced": (40.0, 9.0),
}

# Burnout is assumed to bite more strongly among staff whose jobs depend most on
# outbreak deployment and analytical response work.
BURNOUT_MULTIPLIER = {
    "frontline": 0.55,
    "intermediate": 0.85,
    "advanced": 1.00,
}


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def normal_age_weights(mean: float, sd: float) -> np.ndarray:
    """Return normalized weights over age bands for new graduates."""
    weights = np.exp(-0.5 * ((AGE_MIDS - mean) / sd) ** 2)
    weights = weights / weights.sum()
    return weights


GRAD_AGE_WEIGHTS = {
    tier_key: normal_age_weights(*mean_sd)
    for tier_key, mean_sd in GRAD_AGE_MEAN_SD.items()
}


SCENARIOS: Dict[str, Dict[str, float]] = {
    "Status quo": {
        "training_scale": 1.00,
        "budget_coverage_target": 0.90,
        "policy_delay": 2.0,
        "emigration_multiplier": 1.00,
        "attrition_multiplier": 1.00,
        "protected_time_multiplier": 1.00,
        "shock_multiplier": 1.00,
    },
    "Training scale-up": {
        "training_scale": 1.60,
        "budget_coverage_target": 1.05,
        "policy_delay": 1.5,
        "emigration_multiplier": 0.95,
        "attrition_multiplier": 0.95,
        "protected_time_multiplier": 1.05,
        "shock_multiplier": 1.00,
    },
    "Retention package": {
        "training_scale": 1.15,
        "budget_coverage_target": 1.05,
        "policy_delay": 1.5,
        "emigration_multiplier": 0.65,
        "attrition_multiplier": 0.70,
        "protected_time_multiplier": 1.20,
        "shock_multiplier": 1.00,
    },
    "Outbreak shock": {
        "training_scale": 1.15,
        "budget_coverage_target": 1.00,
        "policy_delay": 1.2,
        "emigration_multiplier": 1.15,
        "attrition_multiplier": 1.10,
        "protected_time_multiplier": 1.00,
        "shock_multiplier": 2.50,
    },
    "Fiscal constraint": {
        "training_scale": 0.85,
        "budget_coverage_target": 0.75,
        "policy_delay": 3.0,
        "emigration_multiplier": 1.10,
        "attrition_multiplier": 1.15,
        "protected_time_multiplier": 0.90,
        "shock_multiplier": 1.00,
    },
}


def base_parameters() -> Dict[str, float]:
    """Default parameter set. Values are deliberately transparent and editable."""
    params: Dict[str, float] = {
        # Country and benchmark assumptions
        "population_start": 56_432_944.0,
        "population_growth": 0.020,
        "benchmark_people_per_fe": 200_000.0,
        "counties": 47.0,
        "pheocs": 25.0,
        "productive_hours_per_fte": 1_500.0,
        # Baseline trained headcount by tier. These are counts ever/previously trained
        # and should be reconciled with current active-status records.
        "initial_frontline": 1_100.0,
        "initial_intermediate": 108.0,
        "initial_advanced": 287.0,
        # Effective capacity assumptions by tier
        "active_share_frontline": 0.75,
        "active_share_intermediate": 0.75,
        "active_share_advanced": 0.75,
        "protected_time_frontline": 0.15,
        "protected_time_intermediate": 0.35,
        "protected_time_advanced": 0.65,
        "skill_frontline": DEFAULT_SKILL_WEIGHT["frontline"],
        "skill_intermediate": DEFAULT_SKILL_WEIGHT["intermediate"],
        "skill_advanced": DEFAULT_SKILL_WEIGHT["advanced"],
        # Training pipeline assumptions
        "seats_frontline": 300.0,
        "seats_intermediate": 90.0,
        "seats_advanced": 24.0,
        "duration_months_frontline": 3.0,
        "duration_months_intermediate": 6.0,
        "duration_months_advanced": 24.0,
        "completion_frontline": 0.95,
        "completion_intermediate": 0.90,
        "completion_advanced": 0.85,
        "training_response_to_gap": 0.80,
        "max_training_scale": 2.50,
        "mentor_share_advanced": 0.12,
        "advanced_seats_per_mentor": 0.30,
        # Attrition and outflow assumptions
        "attrition_frontline": 0.035,
        "attrition_intermediate": 0.030,
        "attrition_advanced": 0.025,
        "emigration_frontline": 0.010,
        "emigration_intermediate": 0.020,
        "emigration_advanced": 0.030,
        "retirement_loss_65_plus": 0.15,
        "burnout_threshold_workload": 1.15,
        "burnout_sensitivity": 0.06,
        # Routine surveillance workload assumptions
        "reporting_units": 11_000.0,
        "hours_per_reporting_unit_week": 0.20,
        "annual_alerts": 1_000.0,
        "hours_per_alert": 4.0,
        "dqa_events_per_county": 2.0,
        "hours_per_dqa": 40.0,
        # Outbreak response workload assumptions
        "annual_outbreak_events": 25.0,
        "response_team_epi": 4.0,
        "deployment_days": 10.0,
        "field_hours_per_day": 8.0,
        "outbreak_duration_sensitivity": 0.60,
        # Readiness and analytics assumptions
        "county_readiness_fte": 1.0,
        "pheoc_epi_fte": 1.0,
        "national_readiness_fte": 10.0,
        "analytics_fte_per_county": 0.50,
        "national_analytics_fte": 15.0,
        # Fiscal/policy adjustment assumptions
        "initial_budget_funded_share": 1.0,
        "budget_coverage_target": 0.90,
        "policy_delay": 2.0,
        # Shock assumptions
        "shock_start_year": 2.0,
        "shock_duration_years": 3.0,
        "shock_multiplier": 1.0,
        # Cost assumptions, in 2026 USD, illustrative only
        "cost_per_effective_fte": 65_000.0,
        "training_cost_frontline": 2_500.0,
        "training_cost_intermediate": 7_500.0,
        "training_cost_advanced": 45_000.0,
        # Scenario multipliers
        "training_scale": 1.0,
        "emigration_multiplier": 1.0,
        "attrition_multiplier": 1.0,
        "protected_time_multiplier": 1.0,
    }
    return params


def apply_scenario(params: Dict[str, float], scenario_name: str) -> Dict[str, float]:
    """Overlay scenario-specific parameters onto a base parameter dictionary."""
    updated = params.copy()
    updated.update(SCENARIOS[scenario_name])
    return updated


def tier_value(params: Dict[str, float], prefix: str, tier: str) -> float:
    """Fetch a tier-specific value from the parameter dictionary."""
    return float(params[f"{prefix}_{TIER_KEYS[tier]}"])


def initialize_workforce(params: Dict[str, float]) -> np.ndarray:
    """Create baseline trained workforce matrix with shape tier x age."""
    workforce = np.zeros((len(TIERS), len(AGE_STARTS)))
    for i, tier in enumerate(TIERS):
        key = TIER_KEYS[tier]
        workforce[i, :] = params[f"initial_{key}"] * GRAD_AGE_WEIGHTS[key]
    return workforce


def initialize_training_queues(params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Initialize training queues at steady-state occupancy for each tier."""
    queues: Dict[str, np.ndarray] = {}
    for tier in TIERS:
        key = TIER_KEYS[tier]
        duration = max(1, int(round(params[f"duration_months_{key}"])))
        monthly_seats = params[f"seats_{key}"] / 12.0
        queues[key] = np.full(duration, monthly_seats)
    return queues


def trained_headcount_by_tier(workforce: np.ndarray) -> Dict[str, float]:
    return {tier: float(workforce[i, :].sum()) for i, tier in enumerate(TIERS)}


def potential_effective_capacity(workforce: np.ndarray, params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Return potential effective advanced-equivalent FTE capacity."""
    by_tier: Dict[str, float] = {}
    total = 0.0
    for i, tier in enumerate(TIERS):
        key = TIER_KEYS[tier]
        headcount = float(workforce[i, :].sum())
        active_share = params[f"active_share_{key}"]
        protected_time = min(1.0, params[f"protected_time_{key}"] * params["protected_time_multiplier"])
        skill_weight = params[f"skill_{key}"]
        capacity = headcount * active_share * protected_time * skill_weight
        by_tier[tier] = capacity
        total += capacity
    return total, by_tier


def shock_factor(t: float, params: Dict[str, float]) -> float:
    start = params["shock_start_year"]
    end = start + params["shock_duration_years"]
    if start <= t < end:
        return params["shock_multiplier"]
    return 1.0


def compute_demand(
    t: float,
    population: float,
    realized_capacity: float,
    params: Dict[str, float],
) -> Dict[str, float]:
    """Compute demand components in advanced-equivalent FTEs."""
    benchmark = population / params["benchmark_people_per_fe"]

    # Better capacity generally reveals more alerts, so detected workload rises as
    # surveillance sensitivity improves. The benchmark prevents this from hiding
    # true need when detection is poor.
    capacity_ratio_to_benchmark = realized_capacity / max(benchmark, 1.0)
    surveillance_sensitivity = 0.70 + 0.30 * min(max(capacity_ratio_to_benchmark, 0.0), 1.0)
    detected_alerts = params["annual_alerts"] * surveillance_sensitivity

    routine_hours = (
        params["reporting_units"] * 52.0 * params["hours_per_reporting_unit_week"]
        + detected_alerts * params["hours_per_alert"]
        + params["counties"] * params["dqa_events_per_county"] * params["hours_per_dqa"]
    )
    routine = routine_hours / params["productive_hours_per_fte"]

    event_multiplier = shock_factor(t, params)
    shortage_fraction = max(0.0, 1.0 - capacity_ratio_to_benchmark)
    deployment_days = params["deployment_days"] * (
        1.0 + params["outbreak_duration_sensitivity"] * shortage_fraction
    )
    outbreak_hours = (
        params["annual_outbreak_events"]
        * event_multiplier
        * params["response_team_epi"]
        * deployment_days
        * params["field_hours_per_day"]
    )
    outbreak = outbreak_hours / params["productive_hours_per_fte"]

    readiness = (
        params["counties"] * params["county_readiness_fte"]
        + params["pheocs"] * params["pheoc_epi_fte"]
        + params["national_readiness_fte"]
    )
    analytics = params["counties"] * params["analytics_fte_per_county"] + params["national_analytics_fte"]

    workload_based = routine + outbreak + readiness + analytics
    total = max(benchmark, workload_based)

    return {
        "benchmark": benchmark,
        "routine": routine,
        "outbreak": outbreak,
        "readiness": readiness,
        "analytics": analytics,
        "workload_based": workload_based,
        "total_demand": total,
        "detected_alerts": detected_alerts,
        "deployment_days": deployment_days,
        "event_multiplier": event_multiplier,
    }


def update_workforce_ageing(
    workforce: np.ndarray,
    params: Dict[str, float],
    workload_ratio: float,
) -> np.ndarray:
    """Age the workforce one month and apply mortality, retirement, attrition, emigration and burnout."""
    next_workforce = np.zeros_like(workforce)
    ageing_rate = 1.0 / COHORT_WIDTH_YEARS

    excess_workload = max(0.0, workload_ratio - params["burnout_threshold_workload"])

    for i, tier in enumerate(TIERS):
        key = TIER_KEYS[tier]
        base_attrition = params[f"attrition_{key}"] * params["attrition_multiplier"]
        emigration = params[f"emigration_{key}"] * params["emigration_multiplier"]
        burnout = params["burnout_sensitivity"] * excess_workload * BURNOUT_MULTIPLIER[key]

        annual_loss_rate = base_attrition + emigration + burnout + AGE_MORTALITY.copy()
        annual_loss_rate[-1] += params["retirement_loss_65_plus"]
        annual_loss_rate = np.clip(annual_loss_rate, 0.0, 0.95)

        survivors = workforce[i, :] * (1.0 - annual_loss_rate * DT)
        ageing_out = survivors * ageing_rate * DT
        remaining = survivors - ageing_out

        next_workforce[i, :] += remaining
        next_workforce[i, 1:] += ageing_out[:-1]
        # ageing_out[-1] leaves the modeled active field epidemiology workforce.

    return np.maximum(next_workforce, 0.0)

def update_training_and_graduation(
    workforce: np.ndarray,
    queues: Dict[str, np.ndarray],
    params: Dict[str, float],
    gap_ratio: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    """Advance training queues by one month and add graduates to the workforce."""
    new_workforce = workforce.copy()
    new_queues: Dict[str, np.ndarray] = {}
    annual_seats_used: Dict[str, float] = {}
    monthly_graduates: Dict[str, float] = {}

    dynamic_scale = params["training_scale"] * (1.0 + params["training_response_to_gap"] * max(0.0, gap_ratio))
    dynamic_scale = min(dynamic_scale, params["max_training_scale"])

    advanced_headcount = float(workforce[TIERS.index("Advanced"), :].sum())
    # Mentor-derived capacity now scales with training investment too, so a
    # genuine scale-up can actually raise the advanced-tier ceiling instead of
    # being silently capped at the un-scaled baseline seat count.
    mentor_capacity = advanced_headcount * params["mentor_share_advanced"] * params["advanced_seats_per_mentor"]
    scaled_base_advanced_seats = params["seats_advanced"] * dynamic_scale
    mentor_limited_advanced_seats = max(scaled_base_advanced_seats, mentor_capacity)

    for tier in TIERS:
        key = TIER_KEYS[tier]
        base_annual_seats = params[f"seats_{key}"]
        annual_seats = base_annual_seats * dynamic_scale
        if key == "advanced":
            # Ceiling is now the larger of (scaled target, mentor-supported
            # capacity) rather than clamping back down to the unscaled baseline.
            annual_seats = min(annual_seats, mentor_limited_advanced_seats)
            annual_seats = max(annual_seats, scaled_base_advanced_seats * 0.0)  # keep non-negative, explicit for clarity

        annual_seats_used[tier] = annual_seats
        monthly_new_trainees = annual_seats / 12.0

        queue = queues[key]
        graduates = queue[0] * params[f"completion_{key}"]
        monthly_graduates[tier] = graduates

        shifted = np.empty_like(queue)
        if len(queue) > 1:
            shifted[:-1] = queue[1:]
        shifted[-1] = monthly_new_trainees
        new_queues[key] = shifted

        tier_index = TIERS.index(tier)
        new_workforce[tier_index, :] += graduates * GRAD_AGE_WEIGHTS[key]

    return new_workforce, new_queues, annual_seats_used, monthly_graduates

def performance_indicators(realized_capacity: float, total_demand: float) -> Dict[str, float]:
    """Very simple heuristic performance indicators for dashboard use."""
    coverage = realized_capacity / max(total_demand, 1.0)
    detection_days = max(1.0, 12.0 - 5.0 * min(coverage, 1.4))
    notify_investigate_days = max(0.5, 3.0 - 1.8 * min(coverage, 1.2))
    response_start_days = max(2.0, 14.0 - 7.0 * min(coverage, 1.5))
    probability_717 = 1.0 / (1.0 + np.exp(-8.0 * (coverage - 0.95)))
    return {
        "coverage_ratio": coverage,
        "expected_detection_days": detection_days,
        "expected_notify_investigate_days": notify_investigate_days,
        "expected_response_start_days": response_start_days,
        "probability_meet_717": probability_717,
    }


def run_model(params: Dict[str, float], years: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the monthly stock-flow model.

    Returns:
        summary_df: one row per month with demand, supply, gap and performance
        tier_df: one row per month per tier
        age_df: one row per month per tier per age band
    """
    n_steps = int(years * 12) + 1
    workforce = initialize_workforce(params)
    queues = initialize_training_queues(params)

    potential0, _ = potential_effective_capacity(workforce, params)
    funded_capacity = potential0 * params["initial_budget_funded_share"]

    records = []
    tier_records = []
    age_records = []

    last_realized_capacity = min(potential0, funded_capacity)
    last_total_demand = max(params["population_start"] / params["benchmark_people_per_fe"], 1.0)

    for step in range(n_steps):
        t = step * DT
        year = 2026.0 + t
        population = params["population_start"] * ((1.0 + params["population_growth"]) ** t)

        potential_capacity, potential_by_tier = potential_effective_capacity(workforce, params)
        realized_capacity = min(potential_capacity, funded_capacity)
        demand = compute_demand(t, population, realized_capacity, params)
        total_demand = demand["total_demand"]
        gap = total_demand - realized_capacity
        gap_ratio = max(0.0, gap) / max(total_demand, 1.0)
        workload_ratio = total_demand / max(realized_capacity, 1.0)
        perf = performance_indicators(realized_capacity, total_demand)

        trained_counts = trained_headcount_by_tier(workforce)
        total_headcount = sum(trained_counts.values())
        training_cost = (
            params["seats_frontline"] * params["training_cost_frontline"]
            + params["seats_intermediate"] * params["training_cost_intermediate"]
            + params["seats_advanced"] * params["training_cost_advanced"]
        ) * params["training_scale"]
        annualized_capacity_cost = realized_capacity * params["cost_per_effective_fte"]

        records.append(
            {
                "time_years": t,
                "calendar_year": year,
                "population": population,
                "trained_headcount": total_headcount,
                "potential_capacity": potential_capacity,
                "funded_capacity": funded_capacity,
                "realized_capacity": realized_capacity,
                "total_demand": total_demand,
                "gap": gap,
                "gap_positive": max(0.0, gap),
                "coverage_ratio": perf["coverage_ratio"],
                "benchmark_demand": demand["benchmark"],
                "workload_demand": demand["workload_based"],
                "routine_demand": demand["routine"],
                "outbreak_demand": demand["outbreak"],
                "readiness_demand": demand["readiness"],
                "analytics_demand": demand["analytics"],
                "detected_alerts": demand["detected_alerts"],
                "deployment_days": demand["deployment_days"],
                "event_multiplier": demand["event_multiplier"],
                "expected_detection_days": perf["expected_detection_days"],
                "expected_notify_investigate_days": perf["expected_notify_investigate_days"],
                "expected_response_start_days": perf["expected_response_start_days"],
                "probability_meet_717": perf["probability_meet_717"],
                "annualized_capacity_cost": annualized_capacity_cost,
                "annual_training_cost": training_cost,
            }
        )

        for i, tier in enumerate(TIERS):
            tier_records.append(
                {
                    "time_years": t,
                    "calendar_year": year,
                    "tier": tier,
                    "trained_headcount": trained_counts[tier],
                    "potential_capacity": potential_by_tier[tier],
                }
            )
            for j, age_label in enumerate(AGE_LABELS):
                age_records.append(
                    {
                        "time_years": t,
                        "calendar_year": year,
                        "tier": tier,
                        "age_band": age_label,
                        "trained_headcount": workforce[i, j],
                    }
                )

        if step == n_steps - 1:
            break

        # Policy and budget delay. Budget targets are expressed as effective FTE
        # coverage rather than currency to keep the core model transparent.
        funded_target = total_demand * params["budget_coverage_target"]
        adjustment = (funded_target - funded_capacity) * DT / max(params["policy_delay"], DT)
        funded_capacity = max(0.0, funded_capacity + adjustment)

        # Age, lose and graduate people into the trained workforce.
        workforce = update_workforce_ageing(workforce, params, workload_ratio)
        workforce, queues, _, _ = update_training_and_graduation(workforce, queues, params, gap_ratio)

        last_realized_capacity = realized_capacity
        last_total_demand = total_demand

    summary_df = pd.DataFrame(records)
    tier_df = pd.DataFrame(tier_records)
    age_df = pd.DataFrame(age_records)
    return summary_df, tier_df, age_df


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def add_policy_band(fig: go.Figure, params: Dict[str, float], yref: str = "paper") -> None:
    """Shade outbreak shock years if the shock multiplier is active."""
    if params.get("shock_multiplier", 1.0) <= 1.0:
        return
    start = params["shock_start_year"]
    end = start + params["shock_duration_years"]
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor="rgba(255, 165, 0, 0.12)",
        line_width=0,
        annotation_text="shock window",
        annotation_position="top left",
        yref=yref,
    )


def make_overview_figure(summary_df: pd.DataFrame, params: Dict[str, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary_df["time_years"],
            y=summary_df["total_demand"],
            name="Total demand",
            mode="lines",
            line=dict(width=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary_df["time_years"],
            y=summary_df["realized_capacity"],
            name="Realized effective capacity",
            mode="lines",
            line=dict(width=4, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary_df["time_years"],
            y=summary_df["potential_capacity"],
            name="Potential capacity if funded/protected",
            mode="lines",
            line=dict(width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary_df["time_years"],
            y=summary_df["gap_positive"],
            name="Positive gap",
            mode="lines",
            fill="tozeroy",
            line=dict(width=1),
        )
    )
    add_policy_band(fig, params)
    fig.update_layout(
        title="Demand, effective capacity and positive gap",
        xaxis_title="Years from model start",
        yaxis_title="Advanced-equivalent effective FTE",
        hovermode="x unified",
        template="plotly_white",
        height=470,
    )
    return fig


def make_demand_stack(summary_df: pd.DataFrame, params: Dict[str, float]) -> go.Figure:
    fig = go.Figure()
    components = [
        ("routine_demand", "Routine surveillance"),
        ("outbreak_demand", "Outbreak response"),
        ("readiness_demand", "Readiness and PHEOC"),
        ("analytics_demand", "Analytics and evaluation"),
    ]
    for col, label in components:
        fig.add_trace(
            go.Scatter(
                x=summary_df["time_years"],
                y=summary_df[col],
                name=label,
                mode="lines",
                stackgroup="demand",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=summary_df["time_years"],
            y=summary_df["benchmark_demand"],
            name="Population benchmark",
            mode="lines",
            line=dict(width=3, dash="dash"),
        )
    )
    add_policy_band(fig, params)
    fig.update_layout(
        title="Demand components",
        xaxis_title="Years from model start",
        yaxis_title="Advanced-equivalent effective FTE",
        hovermode="x unified",
        template="plotly_white",
        height=470,
    )
    return fig


def make_tier_figure(tier_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    for tier in TIERS:
        tier_slice = tier_df[tier_df["tier"] == tier]
        fig.add_trace(
            go.Scatter(
                x=tier_slice["time_years"],
                y=tier_slice["trained_headcount"],
                name=f"{tier} headcount",
                mode="lines",
                stackgroup="headcount",
            ),
            row=1,
            col=1,
        )
    for tier in TIERS:
        tier_slice = tier_df[tier_df["tier"] == tier]
        fig.add_trace(
            go.Scatter(
                x=tier_slice["time_years"],
                y=tier_slice["potential_capacity"],
                name=f"{tier} effective capacity",
                mode="lines",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        title="Workforce stock and effective capacity by tier",
        template="plotly_white",
        height=620,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Trained headcount", row=1, col=1)
    fig.update_yaxes(title_text="Effective FTE", row=2, col=1)
    fig.update_xaxes(title_text="Years from model start", row=2, col=1)
    return fig


def make_age_profile(age_df: pd.DataFrame, selected_time: float) -> go.Figure:
    nearest_time = age_df.iloc[(age_df["time_years"] - selected_time).abs().argsort()[:1]]["time_years"].iloc[0]
    selected = age_df[np.isclose(age_df["time_years"], nearest_time)]
    pivot = selected.pivot_table(
        index="age_band",
        columns="tier",
        values="trained_headcount",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(AGE_LABELS)

    fig = go.Figure()
    for tier in TIERS:
        fig.add_trace(
            go.Bar(
                y=pivot.index,
                x=pivot[tier] if tier in pivot.columns else np.zeros(len(pivot)),
                orientation="h",
                name=tier,
            )
        )
    fig.update_layout(
        title=f"Age distribution at year {nearest_time:.1f}",
        xaxis_title="Trained headcount",
        yaxis_title="Age band",
        barmode="stack",
        template="plotly_white",
        height=500,
    )
    return fig


def make_scenario_comparison(base_params: Dict[str, float], years: int, selected_scenarios) -> pd.DataFrame:
    comparison_records = []
    for scenario_name in selected_scenarios:
        scenario_params = apply_scenario(base_params, scenario_name)
        summary, _, _ = run_model(scenario_params, years=years)
        for _, row in summary.iterrows():
            comparison_records.append(
                {
                    "scenario": scenario_name,
                    "time_years": row["time_years"],
                    "total_demand": row["total_demand"],
                    "realized_capacity": row["realized_capacity"],
                    "gap_positive": row["gap_positive"],
                    "coverage_ratio": row["coverage_ratio"],
                }
            )
    return pd.DataFrame(comparison_records)


# -----------------------------------------------------------------------------
# Streamlit user interface
# -----------------------------------------------------------------------------

st.title("Kenya Field Epidemiology Workforce System Dynamics Model")
st.markdown(
    "This revised template models demand, trained workforce supply, effective capacity, "
    "training delays, attrition, policy delays, and outbreak shocks. Outputs are in "
    "advanced-equivalent effective FTE so trained headcount is not mistaken for deployable capacity."
)

with st.expander("What changed from the first draft", expanded=False):
    st.markdown(
        """
        - Uses a monthly stock-flow simulation rather than an ODE with inconsistent year/month labels.
        - Adds explicit demand modules: population benchmark, routine surveillance, outbreak response, readiness/PHEOC, and analytics.
        - Separates trained headcount from effective capacity using active share, protected epidemiology time, and tier skill weights.
        - Adds basic, intermediate, and advanced FETP tiers with training durations, completion rates, mentor constraints, and attrition.
        - Adds policy and fiscal delay by making funded effective capacity adjust gradually toward the target coverage level.
        - Replaces the artificial male/female age pyramid with a single age profile by tier.
        - Adds scenario comparison and exportable model outputs.
        """
    )

base = base_parameters()

with st.sidebar:
    st.header("Scenario")
    scenario = st.selectbox("Preset scenario", list(SCENARIOS.keys()), index=0)
    params = apply_scenario(base, scenario)

    years_to_run = st.slider("Projection horizon, years", 5, 30, 20, 1)

    st.divider()
    st.subheader("Country and benchmark")
    params["population_start"] = st.number_input(
        "Starting population",
        min_value=1_000_000.0,
        max_value=100_000_000.0,
        value=float(params["population_start"]),
        step=500_000.0,
        format="%.0f",
    )
    params["population_growth"] = st.slider(
        "Annual population growth", 0.0, 0.04, float(params["population_growth"]), 0.001, format="%.3f"
    )
    params["benchmark_people_per_fe"] = st.number_input(
        "Benchmark people per trained field epidemiologist",
        min_value=50_000.0,
        max_value=500_000.0,
        value=float(params["benchmark_people_per_fe"]),
        step=25_000.0,
        format="%.0f",
    )
    params["counties"] = st.number_input("Counties", 1.0, 100.0, float(params["counties"]), 1.0, format="%.0f")
    params["pheocs"] = st.number_input("PHEOCs / EOCs", 0.0, 60.0, float(params["pheocs"]), 1.0, format="%.0f")

    st.divider()
    st.subheader("Initial trained headcount")
    for tier in TIERS:
        key = TIER_KEYS[tier]
        params[f"initial_{key}"] = st.number_input(
            f"{tier} trained headcount",
            min_value=0.0,
            max_value=10_000.0,
            value=float(params[f"initial_{key}"]),
            step=10.0,
            format="%.0f",
        )

    st.divider()
    st.subheader("Training pipeline")
    for tier in TIERS:
        key = TIER_KEYS[tier]
        params[f"seats_{key}"] = st.slider(
            f"Annual {tier.lower()} seats",
            min_value=0.0,
            max_value=1_000.0 if tier != "Advanced" else 150.0,
            value=float(params[f"seats_{key}"]),
            step=5.0,
        )
    params["training_response_to_gap"] = st.slider(
        "Training response to workforce gap", 0.0, 2.0, float(params["training_response_to_gap"]), 0.05
    )

    st.divider()
    st.subheader("Demand workload")
    params["reporting_units"] = st.number_input(
        "Reporting units", 100.0, 30_000.0, float(params["reporting_units"]), 100.0, format="%.0f"
    )
    params["annual_alerts"] = st.slider("Annual alerts requiring verification", 0.0, 5_000.0, float(params["annual_alerts"]), 50.0)
    params["annual_outbreak_events"] = st.slider("Annual outbreaks/events", 0.0, 100.0, float(params["annual_outbreak_events"]), 1.0)
    params["deployment_days"] = st.slider("Average deployment days per outbreak", 1.0, 45.0, float(params["deployment_days"]), 1.0)
    params["response_team_epi"] = st.slider("Field epidemiologists per response team", 1.0, 12.0, float(params["response_team_epi"]), 1.0)

    st.divider()
    st.subheader("Policy and retention")
    params["budget_coverage_target"] = st.slider(
        "Target funded coverage of demand", 0.50, 1.30, float(params["budget_coverage_target"]), 0.01
    )
    params["policy_delay"] = st.slider("Policy/budget response delay, years", 0.25, 5.0, float(params["policy_delay"]), 0.25)
    params["burnout_threshold_workload"] = st.slider(
        "Burnout threshold: demand / capacity", 0.8, 2.0, float(params["burnout_threshold_workload"]), 0.05
    )
    params["burnout_sensitivity"] = st.slider("Burnout sensitivity", 0.0, 0.20, float(params["burnout_sensitivity"]), 0.005)

    with st.expander("Advanced tier assumptions"):
        for tier in TIERS:
            key = TIER_KEYS[tier]
            st.caption(tier)
            params[f"active_share_{key}"] = st.slider(
                f"Active share - {tier}", 0.0, 1.0, float(params[f"active_share_{key}"]), 0.01
            )
            params[f"protected_time_{key}"] = st.slider(
                f"Protected epi time - {tier}", 0.0, 1.0, float(params[f"protected_time_{key}"]), 0.01
            )
            params[f"attrition_{key}"] = st.slider(
                f"Annual non-retirement attrition - {tier}", 0.0, 0.15, float(params[f"attrition_{key}"]), 0.005
            )
            params[f"emigration_{key}"] = st.slider(
                f"Annual external migration pressure - {tier}", 0.0, 0.15, float(params[f"emigration_{key}"]), 0.005
            )

    with st.expander("Outbreak shock settings"):
        params["shock_start_year"] = st.slider("Shock start year", 0.0, float(years_to_run), float(params["shock_start_year"]), 0.5)
        params["shock_duration_years"] = st.slider("Shock duration, years", 0.0, 10.0, float(params["shock_duration_years"]), 0.5)
        params["shock_multiplier"] = st.slider("Shock multiplier", 1.0, 5.0, float(params["shock_multiplier"]), 0.1)

summary_df, tier_df, age_df = run_model(params, years=years_to_run)

start = summary_df.iloc[0]
final = summary_df.iloc[-1]

metric_cols = st.columns(5)
with metric_cols[0]:
    st.metric(
        "Final demand",
        f"{final['total_demand']:.0f}",
        f"{final['total_demand'] - start['total_demand']:+.0f}",
    )
with metric_cols[1]:
    st.metric(
        "Final effective capacity",
        f"{final['realized_capacity']:.0f}",
        f"{final['realized_capacity'] - start['realized_capacity']:+.0f}",
    )
with metric_cols[2]:
    st.metric(
        "Final positive gap",
        f"{final['gap_positive']:.0f}",
        f"{final['gap_positive'] - start['gap_positive']:+.0f}",
    )
with metric_cols[3]:
    st.metric(
        "Coverage ratio",
        f"{final['coverage_ratio']:.2f}",
        "capacity / demand",
    )
with metric_cols[4]:
    st.metric(
        "7-1-7 probability",
        f"{100 * final['probability_meet_717']:.0f}%",
        "heuristic",
    )

if final["gap_positive"] > 0:
    st.warning(
        "The selected assumptions end with an effective capacity gap. Try retention, protected time, funded coverage, or tiered training scale-up scenarios."
    )
else:
    st.success("The selected assumptions close the effective capacity gap by the end of the projection.")

tabs = st.tabs([
    "Overview",
    "Demand components",
    "Supply and pipeline",
    "Age profile",
    "Scenario comparison",
    "Uncertainty",
    "Export and assumptions",
])

with tabs[0]:
    st.plotly_chart(make_overview_figure(summary_df, params), use_container_width=True)
    st.dataframe(
        summary_df[[
            "time_years",
            "trained_headcount",
            "potential_capacity",
            "funded_capacity",
            "realized_capacity",
            "total_demand",
            "gap_positive",
            "coverage_ratio",
            "expected_detection_days",
            "expected_notify_investigate_days",
            "expected_response_start_days",
        ]].tail(13).round(2),
        use_container_width=True,
    )

with tabs[1]:
    st.plotly_chart(make_demand_stack(summary_df, params), use_container_width=True)
    st.markdown(
        "Demand is the larger of the population benchmark and the workload-based estimate. "
        "The workload estimate includes routine surveillance, outbreak response, readiness/PHEOC coverage, and analytics/evaluation."
    )

with tabs[2]:
    st.plotly_chart(make_tier_figure(tier_df), use_container_width=True)
    latest_tier = tier_df[np.isclose(tier_df["time_years"], tier_df["time_years"].max())]
    st.dataframe(latest_tier.round(2), use_container_width=True)

with tabs[3]:
    selected_year = st.slider("Age profile year", 0.0, float(years_to_run), float(years_to_run), 0.5)
    st.plotly_chart(make_age_profile(age_df, selected_year), use_container_width=True)

with tabs[4]:
    selected_scenarios = st.multiselect(
        "Scenarios to compare",
        list(SCENARIOS.keys()),
        default=["Status quo", "Training scale-up", "Retention package"],
    )
    if selected_scenarios:
        comp = make_scenario_comparison(params, years_to_run, selected_scenarios)
        fig_comp = go.Figure()
        for scenario_name in selected_scenarios:
            s = comp[comp["scenario"] == scenario_name]
            fig_comp.add_trace(
                go.Scatter(
                    x=s["time_years"],
                    y=s["gap_positive"],
                    name=scenario_name,
                    mode="lines",
                    line=dict(width=3),
                )
            )
        fig_comp.update_layout(
            title="Positive effective-capacity gap by scenario",
            xaxis_title="Years from model start",
            yaxis_title="Advanced-equivalent effective FTE gap",
            template="plotly_white",
            hovermode="x unified",
            height=470,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        final_comp = comp.loc[comp.groupby("scenario")["time_years"].idxmax()].copy()
        st.dataframe(final_comp.round(3), use_container_width=True)

with tabs[5]:
    if streamlit_uncertainty_panel is None:
        st.warning("The uncertainty module could not be imported. Keep uncertainty_tools_improved.py in the same directory as this app.")
    else:
        streamlit_uncertainty_panel(params.copy(), run_model, years_to_run)

with tabs[6]:
    st.subheader("Download results")
    output_csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download monthly summary CSV",
        data=output_csv,
        file_name=f"kenya_field_epi_sysdyn_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    tier_csv = tier_df.to_csv(index=False)
    st.download_button(
        label="Download tier results CSV",
        data=tier_csv,
        file_name=f"kenya_field_epi_sysdyn_tiers_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    age_csv = age_df.to_csv(index=False)
    st.download_button(
        label="Download age results CSV",
        data=age_csv,
        file_name=f"kenya_field_epi_sysdyn_age_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    params_json = json.dumps(params, indent=2)
    st.download_button(
        label="Download parameter JSON",
        data=params_json,
        file_name=f"kenya_field_epi_sysdyn_parameters_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
    )

    st.subheader("Parameter set")
    st.json(params)

st.markdown(
    """
    ---
    **Interpretation note.** This is a transparent template, not a calibrated forecast. Replace defaults with county-level HRH, K-FELTP alumni tracking, PHEOC, IDSR, DHIS2, outbreak-response, payroll, and budget data. The 7-1-7 outputs are heuristic indicators to support scenario comparison, not validated probabilities.
    """
)
