"""
main.py
RMS entry point with forward-looking on-books pricing simulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.budget import current_month_context
from src.baseline import generate_baseline_pricing_recommendations
from src.events import apply_event_impacts, load_events
from src.forecast import build_future_forecast
from src.ingest import process_file
from src.intraday import process_intraday_updates
from src.metrics import calculate_daily_metrics, export_metrics
from src.pace import calculate_pace_analysis, load_historical_data
from src.pipeline_budget_forecast import (
    build_month_forecast_budget_context,
    build_monthly_forecast_budget_summaries,
)
from src.pipeline_config import build_pipeline_config
from src.pipeline_inputs import prepare_future_dataset, select_user_comparison_frames
from src.pipeline_outputs import (
    build_pipeline_summary,
    collect_output_paths,
    prepare_output_directory,
    write_recommendation_outputs,
)
from src.pipeline_reporting import write_chart_outputs, write_evaluation_outputs
from src.pricing import generate_rate_recommendations
from src.utils import print_error
from src.validate import (
    check_data_quality,
    save_validation_report,
    validate_data,
    validate_required_fields_for_yoy,
)
from src.yoy import YOY_REQUIRED_FIELDS, build_yoy_comparison, summarize_yoy


def run_pipeline(
    input_path: str,
    future_path: str | None = None,
    budget_path: str | None = None,
    events_path: str | None = None,
    config: dict | None = None,
):
    """Run end-to-end RMS pricing workflow and return output paths + summary payload."""
    config = config or {}

    output_dir = prepare_output_directory(config.get("output_dir", "outputs"))

    runtime = build_pipeline_config(config)
    allow_overbooking = runtime.allow_overbooking
    interactive = runtime.interactive
    elasticity = runtime.elasticity
    default_current_rate = runtime.default_current_rate
    manual_rooms_available = runtime.manual_rooms_available
    pricing_config = runtime.pricing_config
    baseline_config = runtime.baseline_config

    # Historical base dataset.
    historical_required_cols = ["stay_date", "rooms_sold", "room_revenue"]
    if manual_rooms_available is None:
        historical_required_cols.insert(1, "rooms_available")

    historical_df = process_file(
        input_path,
        interactive=interactive,
        user_mapping=config.get("column_mapping"),
        required_columns=historical_required_cols,
    )

    # Optional manual override for total inventory.
    if manual_rooms_available is not None:
        historical_df["rooms_available"] = int(manual_rooms_available)
        historical_df["rooms_available_derived_from_occupancy"] = False

    _, _, as_of_date = current_month_context()

    historical_df, historical_validation = validate_data(
        historical_df,
        allow_overbooking=allow_overbooking,
        as_of_date=as_of_date,
        default_current_rate=default_current_rate,
    )
    if len(historical_df) == 0:
        issue_counts: dict[str, int] = {}
        for issue in historical_validation.issues:
            issue_type = str(issue.get("issue_type", "UNKNOWN"))
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        top_issues = ", ".join(
            f"{k}={v}" for k, v in sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        )
        if not top_issues:
            top_issues = "no issue details captured"
        raise ValueError(
            "No valid historical rows after validation. "
            f"Top issue types: {top_issues}. "
            "Check historical column mapping (stay_date, rooms_sold, room_revenue, rooms_available/occupancy_percent)."
        )

    check_data_quality(historical_df)

    # Future on-books dataset.
    future_df = prepare_future_dataset(
        future_path=future_path,
        input_df=historical_df,
        interactive=interactive,
        mapping=config.get("future_column_mapping"),
        as_of_date=as_of_date,
        manual_rooms_available=manual_rooms_available,
    )
    if len(future_df) > 0:
        future_validation_as_of_date = as_of_date
        if future_path:
            future_dates = pd.to_datetime(future_df["stay_date"], errors="coerce").dropna()
            future_revenue = pd.to_numeric(future_df.get("room_revenue", pd.Series(dtype=float)), errors="coerce")
            if len(future_dates) and future_dates.max() <= as_of_date and future_revenue.isna().any():
                future_validation_as_of_date = future_dates.min() - pd.Timedelta(days=1)

        future_df, future_validation = validate_data(
            future_df,
            allow_overbooking=allow_overbooking,
            as_of_date=future_validation_as_of_date,
            default_current_rate=default_current_rate,
        )
    else:
        future_validation = historical_validation

    # Save validation report with both blocks merged.
    historical_validation.issues.extend(future_validation.issues)
    historical_validation.total_rows = int(historical_validation.total_rows + future_validation.total_rows)
    historical_validation.valid_rows = int(historical_validation.valid_rows + future_validation.valid_rows)
    historical_validation.invalid_rows = int(historical_validation.invalid_rows + future_validation.invalid_rows)

    validation_path = output_dir / "validation_report.txt"
    save_validation_report(historical_validation, str(validation_path))

    historical_metrics = calculate_daily_metrics(historical_df)
    daily_metrics_path = output_dir / "daily_metrics.csv"
    export_metrics(historical_metrics, str(daily_metrics_path))

    cleaned_path = output_dir / "cleaned_data.csv"
    historical_df.to_csv(cleaned_path, index=False)

    events_df = load_events(events_path)

    # Forecast future demand.
    forecast_df = build_future_forecast(
        historical_df=historical_metrics,
        future_df=future_df,
        events_df=events_df,
        blend_on_books_floor=True,
    )
    forecast_path = output_dir / "forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    # Pace/event context.
    historical_dir = Path(__file__).parent / "data" / "historical"
    repo_stly_df = load_historical_data(str(historical_dir))
    yoy_current_df, yoy_prior_df, stly_df, using_uploaded_comparison = select_user_comparison_frames(
        historical_metrics,
        future_df,
        repo_stly_df,
    )

    yoy_field_checks = {
        "current": validate_required_fields_for_yoy(
            yoy_current_df,
            YOY_REQUIRED_FIELDS,
            dataset_label="current_year",
        ),
        "prior": validate_required_fields_for_yoy(
            yoy_prior_df,
            YOY_REQUIRED_FIELDS,
            dataset_label="prior_year",
        ),
    }

    # Week 5 YoY output: keep this independent from pace to preserve baseline flows.
    yoy_df = build_yoy_comparison(yoy_current_df, yoy_prior_df)
    yoy_summary = summarize_yoy(yoy_df)
    yoy_path = output_dir / "yoy_comparison.csv"
    yoy_df.to_csv(yoy_path, index=False)

    combined_for_pace = pd.concat([historical_df, future_df], ignore_index=True) if len(future_df) else historical_df.copy()
    pace_df = calculate_pace_analysis(combined_for_pace, stly_df)
    pace_df = apply_event_impacts(pace_df, events_df)

    merge_forecast_df = forecast_df[
        ["stay_date", "forecast_rooms_sold", "forecast_occ", "base_demand", "on_books"]
    ].copy()
    merge_forecast_df["stay_date"] = pd.to_datetime(merge_forecast_df["stay_date"], errors="coerce")

    future_df = future_df.copy()
    future_df["stay_date"] = pd.to_datetime(future_df["stay_date"], errors="coerce")

    future_context = future_df.merge(
        merge_forecast_df,
        on="stay_date",
        how="left",
    )
    future_context = future_context.merge(
        pace_df[["stay_date", "stly_occupancy", "pace_variance", "impact_level", "event_pct"]],
        on="stay_date",
        how="left",
    )
    future_context["forecast_occ"] = future_context["forecast_occ"].fillna(0.0)

    baseline_input_df = future_context.copy()
    baseline_input_df["occupancy"] = pd.to_numeric(
        baseline_input_df.get("occupancy", baseline_input_df.get("forecast_occ", pd.Series(dtype=float))),
        errors="coerce",
    )
    baseline_reco_df = generate_baseline_pricing_recommendations(
        baseline_input_df,
        historical_df=yoy_prior_df,
        config=baseline_config,
    )
    baseline_reco_path = output_dir / "baseline_recommendations.csv"
    baseline_reco_df.to_csv(baseline_reco_path, index=False)

    # Extracted month forecast/budget block.
    # Previously-created values: anchor_date, target_year, target_month,
    # month_future, forecast_rate_series, actual_revenue_to_date,
    # forecast_revenue_remaining, month_end_forecast, monthly_room_capacity,
    # on_books_rooms_total, and budget_summary. Only budget_summary is consumed
    # later in run_pipeline; the context object keeps the other intermediates
    # available for focused tests and future maintenance.
    month_budget_context = build_month_forecast_budget_context(
        future_context=future_context,
        historical_df=historical_df,
        historical_metrics=historical_metrics,
        stly_df=stly_df,
        as_of_date=as_of_date,
        default_current_rate=default_current_rate,
        budget_path=budget_path,
        config=config,
    )
    budget_summary = month_budget_context.budget_summary
    monthly_budget_summary_df = build_monthly_forecast_budget_summaries(
        future_context=future_context,
        historical_df=historical_df,
        historical_metrics=historical_metrics,
        stly_df=stly_df,
        as_of_date=as_of_date,
        default_current_rate=default_current_rate,
        budget_path=budget_path,
        config=config,
    )
    monthly_budget_summary_path = output_dir / "monthly_budget_summary.csv"
    monthly_budget_summary_df.to_csv(monthly_budget_summary_path, index=False)
    # Baseline policy output for comparison.
    baseline_rules_df = generate_rate_recommendations(
        pace_df.rename(columns={"current_adr": "current_adr"}),
        pricing_config,
    ) if len(pace_df) else pd.DataFrame(columns=["stay_date", "recommended_adr"])

    (
        recommendations_df,
        tailored_summary_df,
        top_raise_df,
        _top_rescue_df,
        priority_full_df,
        recommendations_path,
        tailored_results_path,
        tailored_summary_path,
        top_raise_path,
        top_rescue_path,
        top_monitor_path,
    ) = write_recommendation_outputs(
        output_dir=output_dir,
        future_context=future_context,
        baseline_reco_df=baseline_reco_df,
        pricing_config=pricing_config,
        elasticity=elasticity,
        budget_summary=budget_summary,
        monthly_budget_summary_df=monthly_budget_summary_df,
        tailored_settings=config.get("tailored_settings"),
        target_occ=float(config.get("target_occ", 0.80)),
        comp_set_df=config.get("comp_set_df"),
    )

    (
        forecast_metrics,
        _baseline_vs_new_df,
        projected_uplift,
        full_forecast_vs_actual_df,
        evaluation_metrics_path,
        baseline_vs_new_path,
        forecast_vs_actual_csv_path,
        model_comparison_path,
        subgroup_metrics_path,
        model_backtest_path,
        rate_backtest_path,
        rate_backtest_metrics_path,
        rate_subgroup_metrics_path,
    ) = write_evaluation_outputs(
        output_dir=output_dir,
        historical_metrics=historical_metrics,
        events_df=events_df,
        stly_df=stly_df,
        as_of_date=as_of_date,
        baseline_rules_df=baseline_rules_df,
        elasticity=elasticity,
        tailored_settings=config.get("tailored_settings"),
    )

    intraday_changes_df, intraday_warnings = process_intraday_updates(
        future_context,
        baseline_reco_df,
        config.get("tailored_settings"),
        config.get("intraday_updates_df"),
        comp_set_df=config.get("comp_set_df"),
    )
    intraday_changes_path = output_dir / "intraday_recommendation_changes.csv"
    intraday_changes_df.to_csv(intraday_changes_path, index=False)

    write_chart_outputs(
        output_dir=output_dir,
        recommendations_df=recommendations_df,
        priority_full_df=priority_full_df,
        forecast_vs_actual_df=full_forecast_vs_actual_df,
        model_comparison_df=pd.read_csv(model_comparison_path),
        subgroup_metrics_df=pd.read_csv(subgroup_metrics_path),
    )

    output_paths = collect_output_paths(
        output_dir,
        cleaned_path=cleaned_path,
        daily_metrics_path=daily_metrics_path,
        validation_path=validation_path,
        yoy_path=yoy_path,
        baseline_reco_path=baseline_reco_path,
        forecast_path=forecast_path,
        recommendations_path=recommendations_path,
        tailored_results_path=tailored_results_path,
        tailored_summary_path=tailored_summary_path,
        top_raise_path=top_raise_path,
        top_rescue_path=top_rescue_path,
        top_monitor_path=top_monitor_path,
        evaluation_metrics_path=evaluation_metrics_path,
        baseline_vs_new_path=baseline_vs_new_path,
        forecast_vs_actual_csv_path=forecast_vs_actual_csv_path,
        model_comparison_path=model_comparison_path,
        subgroup_metrics_path=subgroup_metrics_path,
        model_backtest_path=model_backtest_path,
        rate_backtest_path=rate_backtest_path,
        rate_backtest_metrics_path=rate_backtest_metrics_path,
        rate_subgroup_metrics_path=rate_subgroup_metrics_path,
        intraday_updates_path=intraday_changes_path,
        monthly_budget_summary_path=monthly_budget_summary_path,
    )

    summary = build_pipeline_summary(
        budget_summary=budget_summary,
        forecast_metrics=forecast_metrics,
        yoy_summary=yoy_summary,
        yoy_field_checks=yoy_field_checks,
        using_uploaded_comparison=using_uploaded_comparison,
        baseline_reco_df=baseline_reco_df,
        projected_uplift=projected_uplift,
        top_raise_df=top_raise_df,
        tailored_summary_df=tailored_summary_df,
        model_comparison_df=pd.read_csv(model_comparison_path),
        rate_backtest_metrics_df=pd.read_csv(rate_backtest_metrics_path),
        monthly_budget_summary_df=monthly_budget_summary_df,
    )
    if intraday_warnings:
        summary["intraday_update_warnings"] = intraday_warnings

    return output_paths, summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hotel RMS - Forward-looking pricing simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Historical PMS report (CSV/XLSX)")
    parser.add_argument("--future", default=None, help="Future on-the-books report (CSV/XLSX)")
    parser.add_argument("--events", default=None, help="Optional events CSV")
    parser.add_argument("--budget", default=None, help="Optional budget file")
    parser.add_argument("--output", "-o", default="outputs", help="Output base directory")

    parser.add_argument("--rate_floor", type=float, default=99.0)
    parser.add_argument("--rate_ceiling", type=float, default=399.0)
    parser.add_argument("--max_change_pct", type=float, default=0.10)
    parser.add_argument("--elasticity", type=float, default=1.2)
    parser.add_argument("--manual_rooms_available", type=int, default=None, help="Optional manual override for rooms_available across all rows")

    parser.add_argument("--allow-overbooking", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")

    return parser.parse_args()


def _print_summary(summary: dict) -> None:
    budget = summary.get("budget_summary", {})
    forecast = summary.get("forecast_metrics", {})
    print("\n" + "=" * 60)
    print("RMS SUMMARY")
    print("=" * 60)
    print(f"Month-end forecast: ${budget.get('month_end_forecast', 0.0):,.2f}")
    print(f"Budget variance: ${budget.get('variance_to_budget_abs', 0.0):,.2f} ({budget.get('variance_to_budget_pct', 0.0):.2f}%)")
    print(f"Required ADR remaining: ${budget.get('required_adr_remaining', 0.0):,.2f}")
    print(f"Forecast MAE: {forecast.get('mae', np.nan):.3f}")
    print(f"Forecast MAPE: {forecast.get('mape', np.nan):.2f}%")
    print(f"Projected uplift vs baseline: ${summary.get('projected_uplift_vs_baseline', 0.0):,.2f}")
    print("=" * 60)


def main() -> int:
    args = parse_arguments()
    try:
        output_paths, summary = run_pipeline(
            input_path=args.input,
            future_path=args.future,
            events_path=args.events,
            budget_path=args.budget,
            config={
                "output_dir": args.output,
                "rate_floor": args.rate_floor,
                "rate_ceiling": args.rate_ceiling,
                "max_change_pct": args.max_change_pct,
                "elasticity": args.elasticity,
                "manual_rooms_available": args.manual_rooms_available,
                "allow_overbooking": args.allow_overbooking,
                "interactive": not args.no_interactive,
            },
        )
        _print_summary(summary)
        print(f"All files saved to: {Path(output_paths['output_dir']).absolute()}")
        return 0
    except FileNotFoundError as exc:
        print_error(f"File not found: {exc}")
        return 1
    except ValueError as exc:
        print_error(f"Validation error: {exc}")
        return 1
    except Exception as exc:
        print_error(f"Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
