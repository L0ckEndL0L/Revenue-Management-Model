"""Output path and summary helpers for the RMS pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pricing import build_priority_lists, simulate_elasticity_pricing
from src.tailored import build_tailored_recommendations, build_tailored_summary
from src.utils import ensure_directory_exists, get_timestamp


def prepare_output_directory(output_base: str = "outputs") -> Path:
    """Create and return a timestamped run output directory."""
    output_root = ensure_directory_exists(output_base)
    return ensure_directory_exists(str(output_root / f"run_{get_timestamp()}"))


def write_recommendation_outputs(
    *,
    output_dir: Path,
    future_context: pd.DataFrame,
    baseline_reco_df: pd.DataFrame,
    pricing_config,
    elasticity: float,
    budget_summary: dict,
    monthly_budget_summary_df: pd.DataFrame | None,
    tailored_settings: dict | None,
    target_occ: float,
    comp_set_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path, Path]:
    """Write rate, tailored, and priority recommendation CSV outputs."""
    recommendation_frames: list[pd.DataFrame] = []
    dated_future = future_context.copy()
    dated_future["stay_date"] = pd.to_datetime(dated_future.get("stay_date"), errors="coerce")
    if monthly_budget_summary_df is not None and len(monthly_budget_summary_df) > 0:
        monthly_lookup = monthly_budget_summary_df.copy()
        monthly_lookup["year"] = pd.to_numeric(monthly_lookup["year"], errors="coerce")
        monthly_lookup["month"] = pd.to_numeric(monthly_lookup["month"], errors="coerce")
        for (year, month), month_future in dated_future.groupby(
            [dated_future["stay_date"].dt.year, dated_future["stay_date"].dt.month],
            sort=True,
        ):
            budget_row = monthly_lookup[
                (monthly_lookup["year"] == year) & (monthly_lookup["month"] == month)
            ]
            month_budget = budget_row.iloc[0].to_dict() if len(budget_row) else {}
            budget_gap = float(month_budget.get("remaining_budget", 0.0))
            required_adr = float(month_budget.get("required_adr_remaining", 0.0))
            month_recommendations, _ = simulate_elasticity_pricing(
                month_future,
                config=pricing_config,
                elasticity=elasticity,
                budget_gap=budget_gap,
                required_adr_remaining=required_adr,
            )
            month_recommendations["budget_gap"] = budget_gap
            month_recommendations["required_adr_remaining"] = required_adr
            recommendation_frames.append(month_recommendations)
    if recommendation_frames:
        recommendations_df = pd.concat(recommendation_frames, ignore_index=True).sort_values("stay_date").reset_index(drop=True)
    else:
        recommendations_df, _ = simulate_elasticity_pricing(
            future_context,
            config=pricing_config,
            elasticity=elasticity,
            budget_gap=float(budget_summary.get("remaining_budget", 0.0)),
            required_adr_remaining=float(budget_summary.get("required_adr_remaining", 0.0)),
        )

    recommendations_path = output_dir / "rate_recommendations.csv"
    if "recommended_rate" not in recommendations_df.columns and "new_policy_rate" in recommendations_df.columns:
        recommendations_df = recommendations_df.copy()
        recommendations_df["recommended_rate"] = recommendations_df["new_policy_rate"]
    recommendations_df.to_csv(recommendations_path, index=False)

    tailored_results_df = build_tailored_recommendations(
        future_context,
        baseline_reco_df,
        tailored_settings,
        comp_set_df=comp_set_df,
    )
    tailored_results_path = output_dir / "tailored_model_results.csv"
    tailored_results_df.to_csv(tailored_results_path, index=False)

    tailored_summary_df = build_tailored_summary(
        tailored_results_df,
        tailored_settings,
    )
    tailored_summary_path = output_dir / "tailored_model_summary.csv"
    tailored_summary_df.to_csv(tailored_summary_path, index=False)

    top_raise_df, top_rescue_df, top_monitor_df, priority_full_df = build_priority_lists(
        recommendations_df,
        budget_gap=float(budget_summary.get("remaining_budget", 0.0)),
        target_occ=target_occ,
    )

    top_raise_path = output_dir / "top_raise_opportunities.csv"
    top_rescue_path = output_dir / "top_rescue_dates.csv"
    top_monitor_path = output_dir / "top_monitor_dates.csv"
    top_raise_df.to_csv(top_raise_path, index=False)
    top_rescue_df.to_csv(top_rescue_path, index=False)
    top_monitor_df.to_csv(top_monitor_path, index=False)

    return (
        recommendations_df,
        tailored_summary_df,
        top_raise_df,
        top_rescue_df,
        priority_full_df,
        recommendations_path,
        tailored_results_path,
        tailored_summary_path,
        top_raise_path,
        top_rescue_path,
        top_monitor_path,
    )


def collect_output_paths(
    output_dir: Path,
    *,
    cleaned_path: Path,
    daily_metrics_path: Path,
    validation_path: Path,
    yoy_path: Path,
    baseline_reco_path: Path,
    forecast_path: Path,
    recommendations_path: Path,
    tailored_results_path: Path,
    tailored_summary_path: Path,
    top_raise_path: Path,
    top_rescue_path: Path,
    top_monitor_path: Path,
    evaluation_metrics_path: Path,
    baseline_vs_new_path: Path,
    forecast_vs_actual_csv_path: Path,
    model_comparison_path: Path,
    subgroup_metrics_path: Path,
    model_backtest_path: Path,
    rate_backtest_path: Path,
    rate_backtest_metrics_path: Path,
    rate_subgroup_metrics_path: Path,
    intraday_updates_path: Path | None = None,
    monthly_budget_summary_path: Path | None = None,
) -> dict[str, str]:
    """Return the public output path mapping used by CLI and Streamlit."""
    return {
        "output_dir": str(output_dir),
        "cleaned_data": str(cleaned_path),
        "daily_metrics": str(daily_metrics_path),
        "validation_report": str(validation_path),
        "yoy_comparison": str(yoy_path),
        "baseline_recommendations": str(baseline_reco_path),
        "forecast": str(forecast_path),
        "rate_recommendations": str(recommendations_path),
        "tailored_model_results": str(tailored_results_path),
        "tailored_model_summary": str(tailored_summary_path),
        "top_raise_opportunities": str(top_raise_path),
        "top_rescue_dates": str(top_rescue_path),
        "top_monitor_dates": str(top_monitor_path),
        "evaluation_metrics": str(evaluation_metrics_path),
        "baseline_vs_new_policy": str(baseline_vs_new_path),
        "forecast_vs_actual": str(forecast_vs_actual_csv_path),
        "baseline_vs_tailored_model_metrics": str(model_comparison_path),
        "subgroup_backtest_metrics": str(subgroup_metrics_path),
        "model_backtest_results": str(model_backtest_path),
        "rate_backtest_results": str(rate_backtest_path),
        "rate_backtest_metrics": str(rate_backtest_metrics_path),
        "rate_subgroup_backtest_metrics": str(rate_subgroup_metrics_path),
        "intraday_recommendation_changes": str(intraday_updates_path) if intraday_updates_path else "",
        "monthly_budget_summary": str(monthly_budget_summary_path) if monthly_budget_summary_path else "",
    }


def build_pipeline_summary(
    *,
    budget_summary: dict,
    forecast_metrics: dict,
    yoy_summary: dict,
    yoy_field_checks: dict,
    using_uploaded_comparison: bool,
    baseline_reco_df: pd.DataFrame,
    projected_uplift: float,
    top_raise_df: pd.DataFrame,
    tailored_summary_df: pd.DataFrame,
    model_comparison_df: pd.DataFrame | None = None,
    rate_backtest_metrics_df: pd.DataFrame | None = None,
    monthly_budget_summary_df: pd.DataFrame | None = None,
) -> dict:
    """Build the public summary payload returned by run_pipeline."""
    summary = {
        "budget_summary": budget_summary,
        "forecast_metrics": forecast_metrics,
        "yoy_summary": yoy_summary,
        "yoy_field_checks": yoy_field_checks,
        "using_uploaded_comparison": using_uploaded_comparison,
        "baseline_rows": int(len(baseline_reco_df)),
        "baseline_unavailable_rows": int((baseline_reco_df.get("baseline_status", pd.Series(dtype=str)) == "UNAVAILABLE").sum()),
        "projected_uplift_vs_baseline": projected_uplift,
        "heavy_need_days": int(len(top_raise_df)),
        "tailored_summary": tailored_summary_df.iloc[0].to_dict() if len(tailored_summary_df) else {},
    }
    if model_comparison_df is not None and len(model_comparison_df) > 0:
        summary["model_comparison"] = model_comparison_df.to_dict("records")
    if rate_backtest_metrics_df is not None and len(rate_backtest_metrics_df) > 0:
        summary["rate_backtest_metrics"] = rate_backtest_metrics_df.to_dict("records")
    if monthly_budget_summary_df is not None and len(monthly_budget_summary_df) > 0:
        summary["monthly_budget_summaries"] = monthly_budget_summary_df.to_dict("records")
    return summary
