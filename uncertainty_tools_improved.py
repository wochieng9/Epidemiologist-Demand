"""
Uncertainty-analysis utilities for the Kenya Field Epidemiology Workforce
System Dynamics app.

This module adds:
1. bounded parameter uncertainty specifications,
2. reproducible Monte Carlo ensemble runs,
3. percentile uncertainty bands,
4. one-way sensitivity/tornado analysis, and
5. a Streamlit panel that can be imported into sysdyn_improved.py.

The functions are designed to work with the run_model(params, years) interface in
sysdyn_improved.py, which returns summary_df, tier_df, age_df.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RunModelFunc = Callable[[Dict[str, float], int], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]

# Bounds are deliberately transparent and editable. The sampling functions center
# each distribution on the CURRENT app parameter value, not only on these defaults.
PARAMETER_UNCERTAINTY: Dict[str, Dict[str, object]] = {
    # Demand drivers
    "reporting_units": {
        "base": 11_000.0,
        "min": 9_500.0,
        "max": 12_500.0,
        "distribution": "normal",
        "description": "Health facility reporting units; could vary with scope and reporting-unit definitions.",
    },
    "annual_outbreak_events": {
        "base": 25.0,
        "min": 12.0,
        "max": 45.0,
        "distribution": "lognormal",
        "description": "Outbreak/event frequency; varies with climate, mobility, detection sensitivity and case definitions.",
    },
    "annual_alerts": {
        "base": 1_000.0,
        "min": 500.0,
        "max": 2_000.0,
        "distribution": "lognormal",
        "description": "Alerts requiring verification; depends strongly on IDSR reporting quality and surveillance sensitivity.",
    },
    "deployment_days": {
        "base": 10.0,
        "min": 6.0,
        "max": 18.0,
        "distribution": "normal",
        "description": "Average field days per outbreak response.",
    },
    "response_team_epi": {
        "base": 4.0,
        "min": 2.0,
        "max": 7.0,
        "distribution": "normal",
        "description": "Field epidemiologists per response team.",
    },
    "productive_hours_per_fte": {
        "base": 1_500.0,
        "min": 1_250.0,
        "max": 1_700.0,
        "distribution": "normal",
        "description": "Annual productive epidemiology hours per effective FTE.",
    },
    "hours_per_reporting_unit_week": {
        "base": 0.20,
        "min": 0.10,
        "max": 0.35,
        "distribution": "lognormal",
        "description": "Routine weekly surveillance review hours per reporting unit.",
    },
    "hours_per_alert": {
        "base": 4.0,
        "min": 2.0,
        "max": 8.0,
        "distribution": "lognormal",
        "description": "Average verification/investigation hours per alert.",
    },
    # Workforce availability and effective capacity
    "active_share_frontline": {
        "base": 0.75,
        "min": 0.60,
        "max": 0.85,
        "distribution": "beta",
        "description": "Fraction of trained frontline staff still active in relevant public-health work.",
    },
    "active_share_intermediate": {
        "base": 0.75,
        "min": 0.60,
        "max": 0.85,
        "distribution": "beta",
        "description": "Fraction of trained intermediate staff still active in relevant public-health work.",
    },
    "active_share_advanced": {
        "base": 0.75,
        "min": 0.65,
        "max": 0.90,
        "distribution": "beta",
        "description": "Fraction of trained advanced staff still active in relevant public-health work.",
    },
    "protected_time_frontline": {
        "base": 0.15,
        "min": 0.10,
        "max": 0.25,
        "distribution": "beta",
        "description": "Share of active frontline staff time protected for epidemiology work.",
    },
    "protected_time_intermediate": {
        "base": 0.35,
        "min": 0.25,
        "max": 0.50,
        "distribution": "beta",
        "description": "Share of active intermediate staff time protected for epidemiology work.",
    },
    "protected_time_advanced": {
        "base": 0.65,
        "min": 0.50,
        "max": 0.80,
        "distribution": "beta",
        "description": "Share of active advanced staff time protected for epidemiology work.",
    },
    # Attrition and emigration
    "attrition_frontline": {
        "base": 0.035,
        "min": 0.015,
        "max": 0.070,
        "distribution": "lognormal",
        "description": "Annual non-retirement attrition from frontline field epidemiology roles.",
    },
    "attrition_intermediate": {
        "base": 0.030,
        "min": 0.012,
        "max": 0.065,
        "distribution": "lognormal",
        "description": "Annual non-retirement attrition from intermediate field epidemiology roles.",
    },
    "attrition_advanced": {
        "base": 0.025,
        "min": 0.010,
        "max": 0.055,
        "distribution": "lognormal",
        "description": "Annual non-retirement attrition from advanced field epidemiology roles.",
    },
    "emigration_frontline": {
        "base": 0.010,
        "min": 0.002,
        "max": 0.030,
        "distribution": "lognormal",
        "description": "Annual migration or external job-pull pressure for frontline staff.",
    },
    "emigration_intermediate": {
        "base": 0.020,
        "min": 0.005,
        "max": 0.060,
        "distribution": "lognormal",
        "description": "Annual migration or external job-pull pressure for intermediate staff.",
    },
    "emigration_advanced": {
        "base": 0.030,
        "min": 0.010,
        "max": 0.080,
        "distribution": "lognormal",
        "description": "Annual migration or external job-pull pressure for advanced staff.",
    },
    # Training and pipeline
    "seats_frontline": {
        "base": 300.0,
        "min": 200.0,
        "max": 450.0,
        "distribution": "normal",
        "description": "Annual K-FELTP/frontline-equivalent seats.",
    },
    "seats_intermediate": {
        "base": 90.0,
        "min": 60.0,
        "max": 140.0,
        "distribution": "normal",
        "description": "Annual intermediate training seats.",
    },
    "seats_advanced": {
        "base": 24.0,
        "min": 12.0,
        "max": 40.0,
        "distribution": "normal",
        "description": "Annual advanced FETP seats before mentor constraints.",
    },
    "completion_frontline": {
        "base": 0.95,
        "min": 0.85,
        "max": 0.99,
        "distribution": "beta",
        "description": "Frontline training completion rate.",
    },
    "completion_intermediate": {
        "base": 0.90,
        "min": 0.75,
        "max": 0.97,
        "distribution": "beta",
        "description": "Intermediate training completion rate.",
    },
    "completion_advanced": {
        "base": 0.85,
        "min": 0.70,
        "max": 0.95,
        "distribution": "beta",
        "description": "Advanced FETP completion rate.",
    },
    # Policy, budget and burnout
    "policy_delay": {
        "base": 2.0,
        "min": 1.0,
        "max": 4.0,
        "distribution": "normal",
        "description": "Years for policy/budget response to workforce gap.",
    },
    "budget_coverage_target": {
        "base": 0.90,
        "min": 0.70,
        "max": 1.10,
        "distribution": "normal",
        "description": "Target funded coverage of computed effective-FTE demand.",
    },
    "burnout_threshold_workload": {
        "base": 1.15,
        "min": 0.95,
        "max": 1.40,
        "distribution": "normal",
        "description": "Demand/capacity ratio where burnout attrition begins to accelerate.",
    },
    "burnout_sensitivity": {
        "base": 0.06,
        "min": 0.02,
        "max": 0.12,
        "distribution": "normal",
        "description": "Strength of burnout effect on attrition once threshold is exceeded.",
    },
}

DEFAULT_METRICS: Tuple[str, ...] = (
    "total_demand",
    "realized_capacity",
    "potential_capacity",
    "gap_positive",
    "coverage_ratio",
    "probability_meet_717",
)


def available_uncertain_parameters(base_params: Dict[str, float]) -> List[str]:
    """Return uncertain parameters that exist in the current app parameter set."""
    return sorted([key for key in PARAMETER_UNCERTAINTY if key in base_params])


def _bounded_current_value(param_name: str, current_value: Optional[float]) -> float:
    spec = PARAMETER_UNCERTAINTY[param_name]
    base = float(spec["base"])
    min_val = float(spec["min"])
    max_val = float(spec["max"])
    value = base if current_value is None else float(current_value)
    return float(np.clip(value, min_val, max_val))


def sample_parameter(
    param_name: str,
    current_value: Optional[float] = None,
    n_samples: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw bounded samples for a parameter.

    Unlike the original add-on version, this function uses the current Streamlit
    value as the center/median of the uncertainty distribution whenever that
    value is supplied. This keeps uncertainty analysis aligned with the selected
    scenario and sidebar sliders.
    """
    if param_name not in PARAMETER_UNCERTAINTY:
        raise KeyError(f"Unknown uncertainty parameter: {param_name}")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    rng = rng or np.random.default_rng()
    spec = PARAMETER_UNCERTAINTY[param_name]
    min_val = float(spec["min"])
    max_val = float(spec["max"])
    center = _bounded_current_value(param_name, current_value)
    dist = str(spec["distribution"]).lower()

    if max_val <= min_val:
        return np.full(n_samples, min_val, dtype=float)

    if dist == "normal":
        # Treat bounds as roughly a 95% practical uncertainty interval, then clip.
        sigma = max((max_val - min_val) / (2.0 * 1.96), 1e-12)
        samples = rng.normal(center, sigma, n_samples)

    elif dist == "lognormal":
        # Treat center as median. Use the more conservative side of the interval
        # to define a broad 90% practical interval, then clip to explicit bounds.
        safe_center = max(center, 1e-12)
        upper_sigma = abs(np.log(max_val / safe_center)) / 1.645 if max_val > safe_center else 0.0
        lower_sigma = abs(np.log(safe_center / max(min_val, 1e-12))) / 1.645 if safe_center > min_val else 0.0
        sigma = max(upper_sigma, lower_sigma, 1e-6)
        samples = rng.lognormal(mean=np.log(safe_center), sigma=sigma, size=n_samples)

    elif dist == "beta":
        # Scaled beta with mean centered on the current parameter value.
        normalized_mean = (center - min_val) / (max_val - min_val)
        normalized_mean = float(np.clip(normalized_mean, 0.02, 0.98))
        concentration = float(spec.get("concentration", 24.0))
        alpha = max(normalized_mean * concentration, 0.5)
        beta_param = max((1.0 - normalized_mean) * concentration, 0.5)
        samples01 = rng.beta(alpha, beta_param, n_samples)
        samples = min_val + samples01 * (max_val - min_val)

    elif dist == "uniform":
        samples = rng.uniform(min_val, max_val, n_samples)

    else:
        raise ValueError(f"Unknown distribution for {param_name}: {dist}")

    return np.clip(samples, min_val, max_val).astype(float)


def generate_parameter_ensemble(
    base_params: Dict[str, float],
    n_samples: int = 100,
    param_subset: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate an ensemble of sampled parameter sets.

    Returns a DataFrame with one row per run. Each row contains all base
    parameters plus sampled values for the varied parameters and a run_id.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    if param_subset is None:
        varied = available_uncertain_parameters(base_params)
    else:
        varied = [p for p in param_subset if p in PARAMETER_UNCERTAINTY and p in base_params]

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for run_id in range(n_samples):
        params = dict(base_params)
        for param_name in varied:
            params[param_name] = float(
                sample_parameter(
                    param_name,
                    current_value=float(base_params[param_name]),
                    n_samples=1,
                    rng=rng,
                )[0]
            )
        params["run_id"] = float(run_id)
        rows.append(params)

    return pd.DataFrame(rows)


def summarize_ensemble(
    combined_df: pd.DataFrame,
    metrics: Iterable[str] = DEFAULT_METRICS,
    quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
) -> pd.DataFrame:
    """Compute percentile bands by time for selected model outputs."""
    if "time_years" not in combined_df.columns:
        raise KeyError("combined_df must include a time_years column")

    available_metrics = [metric for metric in metrics if metric in combined_df.columns]
    if not available_metrics:
        raise ValueError("None of the requested metrics are present in combined_df")

    grouped = combined_df.groupby("time_years", sort=True)[available_metrics]
    percentile_df = grouped.quantile(list(quantiles)).unstack(level=-1)
    percentile_df.columns = [f"{metric}_p{int(round(q * 100)):02d}" for metric, q in percentile_df.columns]
    percentile_df = percentile_df.reset_index()
    return percentile_df


def run_ensemble(
    base_params: Dict[str, float],
    run_model_func: RunModelFunc,
    n_samples: int = 100,
    years: int = 20,
    param_subset: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the model for sampled parameter sets.

    Returns:
        percentile_df: time series percentiles across ensemble runs
        ensemble_params_df: sampled parameters used by each run
        combined_df: all monthly model outputs with run_id attached
        final_by_run_df: final-year outputs, one row per run
    """
    ensemble_params_df = generate_parameter_ensemble(
        base_params=base_params,
        n_samples=n_samples,
        param_subset=param_subset,
        seed=seed,
    )

    all_summaries: List[pd.DataFrame] = []
    param_cols_to_drop = ["run_id"]
    for i, row in ensemble_params_df.iterrows():
        params = row.drop(labels=param_cols_to_drop).to_dict()
        summary_df, _, _ = run_model_func(params, years=years)
        summary_df = summary_df.copy()
        summary_df["run_id"] = int(row["run_id"])
        all_summaries.append(summary_df)
        if progress_callback is not None:
            progress_callback(i + 1, n_samples)

    combined_df = pd.concat(all_summaries, ignore_index=True)
    percentile_df = summarize_ensemble(combined_df, metrics=DEFAULT_METRICS)
    final_idx = combined_df.groupby("run_id")["time_years"].idxmax()
    final_by_run_df = combined_df.loc[final_idx].copy().reset_index(drop=True)

    return percentile_df, ensemble_params_df, combined_df, final_by_run_df


def _band_trace(x: pd.Series, upper: pd.Series, lower: pd.Series, name: str) -> go.Scatter:
    return go.Scatter(
        x=x.tolist() + x.tolist()[::-1],
        y=upper.tolist() + lower.tolist()[::-1],
        fill="toself",
        line={"width": 0},
        name=name,
        hoverinfo="skip",
    )


def make_uncertainty_figure(percentile_df: pd.DataFrame) -> go.Figure:
    """Create demand/capacity percentile-band figure."""
    fig = go.Figure()
    x = percentile_df["time_years"]

    fig.add_trace(_band_trace(x, percentile_df["total_demand_p90"], percentile_df["total_demand_p10"], "Demand 80% uncertainty interval"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=percentile_df["total_demand_p50"],
            name="Demand median",
            mode="lines",
            line={"width": 3},
        )
    )

    fig.add_trace(_band_trace(x, percentile_df["realized_capacity_p90"], percentile_df["realized_capacity_p10"], "Capacity 80% uncertainty interval"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=percentile_df["realized_capacity_p50"],
            name="Capacity median",
            mode="lines",
            line={"width": 3, "dash": "dash"},
        )
    )

    fig.update_layout(
        title="Demand and realized capacity with uncertainty bands",
        xaxis_title="Years from model start",
        yaxis_title="Advanced-equivalent effective FTE",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    return fig


def make_gap_uncertainty_figure(percentile_df: pd.DataFrame) -> go.Figure:
    """Create positive-gap percentile-band figure."""
    fig = go.Figure()
    x = percentile_df["time_years"]

    fig.add_trace(_band_trace(x, percentile_df["gap_positive_p90"], percentile_df["gap_positive_p10"], "Gap 80% uncertainty interval"))
    fig.add_trace(_band_trace(x, percentile_df["gap_positive_p75"], percentile_df["gap_positive_p25"], "Gap 50% uncertainty interval"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=percentile_df["gap_positive_p50"],
            name="Gap median",
            mode="lines",
            line={"width": 3},
        )
    )

    fig.update_layout(
        title="Positive capacity gap with uncertainty bands",
        xaxis_title="Years from model start",
        yaxis_title="Gap, advanced-equivalent effective FTE",
        hovermode="x unified",
        template="plotly_white",
        height=470,
    )
    return fig


def make_tornado_chart(
    base_params: Dict[str, float],
    run_model_func: RunModelFunc,
    years: int = 20,
    outcome_year: Optional[float] = None,
    metric: str = "gap_positive",
    param_subset: Optional[Sequence[str]] = None,
    top_n: int = 18,
) -> Tuple[go.Figure, pd.DataFrame]:
    """Run one-way sensitivity analysis and return a tornado-style chart."""
    if outcome_year is None:
        outcome_year = float(years)

    candidate_params = param_subset if param_subset is not None else available_uncertain_parameters(base_params)
    varied = [p for p in candidate_params if p in PARAMETER_UNCERTAINTY and p in base_params]
    if not varied:
        raise ValueError("No valid uncertain parameters selected for sensitivity analysis")

    base_summary, _, _ = run_model_func(dict(base_params), years=years)
    if metric not in base_summary.columns:
        raise KeyError(f"Metric {metric!r} is not present in model outputs")
    base_row = base_summary.iloc[(base_summary["time_years"] - outcome_year).abs().argsort()[:1]]
    base_value = float(base_row.iloc[0][metric])

    rows: List[Dict[str, float]] = []
    for param_name in varied:
        spec = PARAMETER_UNCERTAINTY[param_name]
        min_val = float(spec["min"])
        max_val = float(spec["max"])

        params_min = dict(base_params)
        params_min[param_name] = min_val
        summary_min, _, _ = run_model_func(params_min, years=years)
        min_row = summary_min.iloc[(summary_min["time_years"] - outcome_year).abs().argsort()[:1]]
        value_at_min = float(min_row.iloc[0][metric])

        params_max = dict(base_params)
        params_max[param_name] = max_val
        summary_max, _, _ = run_model_func(params_max, years=years)
        max_row = summary_max.iloc[(summary_max["time_years"] - outcome_year).abs().argsort()[:1]]
        value_at_max = float(max_row.iloc[0][metric])

        rows.append(
            {
                "parameter": param_name,
                "base_value": base_value,
                "parameter_min": min_val,
                "parameter_max": max_val,
                "outcome_at_parameter_min": value_at_min,
                "outcome_at_parameter_max": value_at_max,
                "delta_at_parameter_min": value_at_min - base_value,
                "delta_at_parameter_max": value_at_max - base_value,
                "swing": abs(value_at_max - value_at_min),
                "description": str(spec.get("description", "")),
            }
        )

    sens_df = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    chart_df = sens_df.head(top_n).sort_values("swing", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=chart_df["parameter"],
            x=chart_df["delta_at_parameter_min"],
            orientation="h",
            name="Parameter at min",
        )
    )
    fig.add_trace(
        go.Bar(
            y=chart_df["parameter"],
            x=chart_df["delta_at_parameter_max"],
            orientation="h",
            name="Parameter at max",
        )
    )
    fig.add_vline(x=0, line_dash="dash", annotation_text="Base case")
    fig.update_layout(
        title=f"One-way sensitivity: change in {metric} near year {outcome_year:g}",
        xaxis_title=f"Change from base case in {metric}",
        yaxis_title="Parameter",
        barmode="overlay",
        template="plotly_white",
        height=max(450, 28 * len(chart_df) + 160),
    )
    return fig, sens_df


def uncertainty_spec_dataframe(param_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Return parameter uncertainty specifications as a table."""
    names = list(param_names) if param_names is not None else sorted(PARAMETER_UNCERTAINTY.keys())
    rows = []
    for name in names:
        if name not in PARAMETER_UNCERTAINTY:
            continue
        spec = PARAMETER_UNCERTAINTY[name]
        rows.append(
            {
                "parameter": name,
                "base": spec["base"],
                "min": spec["min"],
                "max": spec["max"],
                "distribution": spec["distribution"],
                "description": spec.get("description", ""),
            }
        )
    return pd.DataFrame(rows)


def streamlit_uncertainty_panel(base_params: Dict[str, float], run_model_func: RunModelFunc, years_to_run: int) -> None:
    """Render the uncertainty-analysis panel inside the Streamlit app."""
    st.subheader("Monte Carlo uncertainty and one-way sensitivity")
    st.markdown(
        "Uncertainty analysis samples selected parameters around the values currently set in the sidebar. "
        "The shaded bands are empirical percentiles across model runs, not calibrated statistical confidence intervals."
    )

    available_params = available_uncertain_parameters(base_params)
    default_selection = [
        p
        for p in [
            "annual_outbreak_events",
            "annual_alerts",
            "protected_time_intermediate",
            "protected_time_advanced",
            "emigration_advanced",
            "budget_coverage_target",
            "policy_delay",
            "burnout_sensitivity",
        ]
        if p in available_params
    ]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        n_samples = st.slider("Monte Carlo samples", 25, 500, 100, step=25)
    with col2:
        seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=12345, step=1)
    with col3:
        top_n = st.slider("Tornado parameters shown", 5, 25, 15, step=1)

    varied_params = st.multiselect(
        "Parameters to vary",
        available_params,
        default=default_selection,
        help="Use fewer parameters for faster exploratory runs; use more for a full uncertainty screen.",
    )

    with st.expander("Parameter uncertainty ranges", expanded=False):
        st.dataframe(uncertainty_spec_dataframe(varied_params), use_container_width=True)

    run_button = st.button("Run uncertainty analysis", type="primary", disabled=len(varied_params) == 0)

    result_key = "uncertainty_results_v2"
    if run_button:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_callback(current: int, total: int) -> None:
            progress_bar.progress(current / total)
            status_text.text(f"Completed {current} of {total} ensemble runs")

        percentile_df, ensemble_params_df, combined_df, final_by_run_df = run_ensemble(
            base_params=dict(base_params),
            run_model_func=run_model_func,
            n_samples=int(n_samples),
            years=int(years_to_run),
            param_subset=varied_params,
            seed=int(seed),
            progress_callback=progress_callback,
        )
        progress_bar.empty()
        status_text.empty()

        st.session_state[result_key] = {
            "percentile_df": percentile_df,
            "ensemble_params_df": ensemble_params_df,
            "combined_df": combined_df,
            "final_by_run_df": final_by_run_df,
            "varied_params": list(varied_params),
            "n_samples": int(n_samples),
            "seed": int(seed),
        }

    results = st.session_state.get(result_key)
    if not results:
        st.info("Choose parameters and run the analysis to display uncertainty bands and a tornado chart.")
        return

    percentile_df = results["percentile_df"]
    ensemble_params_df = results["ensemble_params_df"]
    combined_df = results["combined_df"]
    final_by_run_df = results["final_by_run_df"]
    selected_params = results["varied_params"]

    st.caption(
        f"Showing stored results from {results['n_samples']} runs, seed {results['seed']}, "
        f"varying {len(selected_params)} parameters. Rerun after changing sidebar assumptions."
    )

    st.plotly_chart(make_uncertainty_figure(percentile_df), use_container_width=True)
    st.plotly_chart(make_gap_uncertainty_figure(percentile_df), use_container_width=True)

    final_row = percentile_df.iloc[-1]
    prob_gap_closed = float((final_by_run_df["gap_positive"] < 5.0).mean())
    prob_coverage_90 = float((final_by_run_df["coverage_ratio"] >= 0.90).mean())

    st.subheader("Final-year uncertainty summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Demand median",
            f"{final_row['total_demand_p50']:.0f}",
            f"80% UI [{final_row['total_demand_p10']:.0f}, {final_row['total_demand_p90']:.0f}]",
        )
    with m2:
        st.metric(
            "Capacity median",
            f"{final_row['realized_capacity_p50']:.0f}",
            f"80% UI [{final_row['realized_capacity_p10']:.0f}, {final_row['realized_capacity_p90']:.0f}]",
        )
    with m3:
        st.metric(
            "Gap median",
            f"{final_row['gap_positive_p50']:.0f}",
            f"80% UI [{final_row['gap_positive_p10']:.0f}, {final_row['gap_positive_p90']:.0f}]",
        )
    with m4:
        st.metric("Pr(gap < 5 FTE)", f"{100.0 * prob_gap_closed:.0f}%", f"Pr(coverage >= 0.90): {100.0 * prob_coverage_90:.0f}%")

    st.divider()
    st.subheader("One-way sensitivity analysis")
    outcome_metric = st.selectbox(
        "Outcome for tornado chart",
        ["gap_positive", "realized_capacity", "total_demand", "coverage_ratio", "probability_meet_717"],
        index=0,
    )

    numeric_signature = tuple(
        sorted(
            (key, round(float(value), 10))
            for key, value in base_params.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        )
    )
    tornado_signature = (
        tuple(selected_params),
        int(years_to_run),
        str(outcome_metric),
        int(top_n),
        numeric_signature,
    )
    tornado_key = "uncertainty_tornado_results_v2"
    tornado_state = st.session_state.get(tornado_key)
    run_tornado = st.button("Run/update tornado chart")

    if run_tornado or not tornado_state or tornado_state.get("signature") != tornado_signature:
        with st.spinner("Computing one-way sensitivity..."):
            fig_tornado, sens_df = make_tornado_chart(
                base_params=dict(base_params),
                run_model_func=run_model_func,
                years=int(years_to_run),
                outcome_year=float(years_to_run),
                metric=outcome_metric,
                param_subset=selected_params,
                top_n=int(top_n),
            )
        st.session_state[tornado_key] = {
            "signature": tornado_signature,
            "fig": fig_tornado,
            "sens_df": sens_df,
        }
    else:
        fig_tornado = tornado_state["fig"]
        sens_df = tornado_state["sens_df"]

    st.plotly_chart(fig_tornado, use_container_width=True)
    st.dataframe(
        sens_df[
            [
                "parameter",
                "swing",
                "outcome_at_parameter_min",
                "outcome_at_parameter_max",
                "description",
            ]
        ].round(4),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Download uncertainty outputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download all ensemble runs",
            data=combined_df.to_csv(index=False),
            file_name="kenya_epi_uncertainty_ensemble_runs.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download percentile summary",
            data=percentile_df.to_csv(index=False),
            file_name="kenya_epi_uncertainty_percentiles.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "Download sampled parameters",
            data=ensemble_params_df.to_csv(index=False),
            file_name="kenya_epi_uncertainty_parameters.csv",
            mime="text/csv",
        )
