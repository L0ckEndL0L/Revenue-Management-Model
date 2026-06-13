"""Evaluation and chart reporting helpers for the RMS pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.elasticity import expected_rooms_sold
from src.evaluation import (
    build_policy_evaluation_metrics,
    calculate_forecast_metrics,
    plot_current_vs_recommended_rate,
    plot_expected_revenue_uplift,
    plot_forecast_vs_actual,
    plot_priority_score_by_date,
)
from src.forecast import baseline_forecast, enhanced_forecast, evaluate_backtest, prepare_forecast_frame
from src.pricing import PricingConfig, simulate_elasticity_pricing


def build_baseline_vs_new_policy(
    historical_metrics: pd.DataFrame,
    baseline_rules_df: pd.DataFrame,
    elasticity: float,
) -> pd.DataFrame:
    """Per-date backtest style comparison of baseline policy vs new simulation policy."""
    if len(historical_metrics) == 0:
        return pd.DataFrame(
            columns=[
                "stay_date",
                "current_rate",
                "baseline_rate",
                "new_policy_rate",
                "forecast_rooms_sold",
                "baseline_expected_revenue",
                "new_policy_expected_revenue",
                "expected_revenue_uplift",
            ]
        )

    eval_df = historical_metrics.copy().sort_values("stay_date")
    eval_df["dow"] = eval_df["stay_date"].dt.dayofweek
    dow_avg = eval_df.groupby("dow")["rooms_sold"].transform("mean")
    eval_df["forecast_rooms_sold"] = dow_avg.fillna(eval_df["rooms_sold"].mean())
    eval_df["current_rate"] = np.where(eval_df["rooms_sold"] > 0, eval_df["room_revenue"] / eval_df["rooms_sold"], eval_df.get("adr", 0.0))
    eval_df["on_books_eval"] = np.minimum(eval_df["rooms_sold"] * 0.70, eval_df["forecast_rooms_sold"])

    merged = eval_df.merge(
        baseline_rules_df[["stay_date", "recommended_adr"]].rename(columns={"recommended_adr": "baseline_rate"}),
        on="stay_date",
        how="left",
    )

    merged["baseline_rate"] = merged["baseline_rate"].fillna(merged["current_rate"])

    sim_input = merged[
        ["stay_date", "rooms_available", "on_books_eval", "current_rate", "forecast_rooms_sold", "occupancy"]
    ].copy()
    sim_input = sim_input.rename(columns={"on_books_eval": "rooms_sold", "occupancy": "forecast_occ"})
    sim_input["pace_variance"] = np.nan
    sim_input["impact_level"] = pd.NA
    sim_input["event_pct"] = 0.0

    new_policy_df, _ = simulate_elasticity_pricing(
        sim_input,
        config=PricingConfig(),
        elasticity=elasticity,
        budget_gap=0.0,
        required_adr_remaining=0.0,
    )

    merged = merged.merge(
        new_policy_df[["stay_date", "recommended_rate"]].rename(columns={"recommended_rate": "new_policy_rate"}),
        on="stay_date",
        how="left",
    )
    merged["new_policy_rate"] = merged["new_policy_rate"].fillna(merged["current_rate"])

    merged["baseline_expected_revenue"] = merged.apply(
        lambda row: row["baseline_rate"]
        * expected_rooms_sold(
            base_demand=row["forecast_rooms_sold"],
            candidate_rate=row["baseline_rate"],
            current_rate=row["current_rate"],
            elasticity=elasticity,
            on_books=row["on_books_eval"],
            rooms_available=row["rooms_available"],
        ),
        axis=1,
    )
    merged["new_policy_expected_revenue"] = merged.apply(
        lambda row: row["new_policy_rate"]
        * expected_rooms_sold(
            base_demand=row["forecast_rooms_sold"],
            candidate_rate=row["new_policy_rate"],
            current_rate=row["current_rate"],
            elasticity=elasticity,
            on_books=row["on_books_eval"],
            rooms_available=row["rooms_available"],
        ),
        axis=1,
    )
    merged["expected_revenue_uplift"] = merged["new_policy_expected_revenue"] - merged["baseline_expected_revenue"]

    return merged[
        [
            "stay_date",
            "current_rate",
            "baseline_rate",
            "new_policy_rate",
            "forecast_rooms_sold",
            "baseline_expected_revenue",
            "new_policy_expected_revenue",
            "expected_revenue_uplift",
        ]
    ].sort_values("stay_date").reset_index(drop=True)


def write_evaluation_outputs(
    *,
    output_dir: Path,
    historical_metrics: pd.DataFrame,
    events_df: pd.DataFrame,
    stly_df: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    baseline_rules_df: pd.DataFrame,
    elasticity: float,
) -> tuple[dict, pd.DataFrame, float, pd.DataFrame, Path, Path]:
    """Write evaluation CSV outputs and return chart-ready forecast data."""
    model_df = prepare_forecast_frame(daily_df=historical_metrics, events_df=events_df, stly_df=stly_df)
    backtest_df = evaluate_backtest(model_df=model_df, as_of_date=as_of_date)
    forecast_metrics = calculate_forecast_metrics(
        actual=backtest_df.get("actual_rooms_sold", pd.Series(dtype=float)),
        predicted=backtest_df.get("baseline_rooms_sold", pd.Series(dtype=float)),
    )

    baseline_vs_new_df = build_baseline_vs_new_policy(historical_metrics, baseline_rules_df, elasticity=elasticity)
    baseline_vs_new_path = output_dir / "baseline_vs_new_policy.csv"
    baseline_vs_new_df.to_csv(baseline_vs_new_path, index=False)

    projected_uplift = float(baseline_vs_new_df.get("expected_revenue_uplift", pd.Series(dtype=float)).sum())
    evaluation_metrics_df = build_policy_evaluation_metrics(forecast_metrics, projected_uplift_vs_baseline=projected_uplift)
    evaluation_metrics_path = output_dir / "evaluation_metrics.csv"
    evaluation_metrics_df.to_csv(evaluation_metrics_path, index=False)

    full_baseline_pred = baseline_forecast(train_df=model_df, target_df=model_df)
    full_enhanced_pred = enhanced_forecast(train_df=model_df, target_df=model_df)
    full_forecast_vs_actual_df = pd.DataFrame(
        {
            "stay_date": model_df["stay_date"],
            "actual_rooms_sold": model_df["rooms_sold"],
            "baseline_rooms_sold": full_baseline_pred["forecast_rooms_sold"],
            "enhanced_rooms_sold": full_enhanced_pred["forecast_rooms_sold"],
            "actual_revenue": model_df["room_revenue"],
            "baseline_revenue": full_baseline_pred["forecast_revenue"],
            "enhanced_revenue": full_enhanced_pred["forecast_revenue"],
        }
    ).sort_values("stay_date").reset_index(drop=True)

    forecast_vs_actual_csv_path = output_dir / "forecast_vs_actual.csv"
    full_forecast_vs_actual_df.to_csv(forecast_vs_actual_csv_path, index=False)

    return (
        forecast_metrics,
        baseline_vs_new_df,
        projected_uplift,
        full_forecast_vs_actual_df,
        evaluation_metrics_path,
        baseline_vs_new_path,
        forecast_vs_actual_csv_path,
    )


def write_chart_outputs(
    *,
    output_dir: Path,
    recommendations_df: pd.DataFrame,
    priority_full_df: pd.DataFrame,
    forecast_vs_actual_df: pd.DataFrame,
) -> None:
    """Write the PNG chart outputs expected by the app and CLI."""
    plot_current_vs_recommended_rate(recommendations_df, str(output_dir / "current_vs_recommended_rate.png"))
    plot_expected_revenue_uplift(recommendations_df, str(output_dir / "expected_revenue_uplift.png"))
    plot_priority_score_by_date(priority_full_df, str(output_dir / "priority_score_by_date.png"))
    plot_forecast_vs_actual(forecast_vs_actual_df, str(output_dir / "forecast_vs_actual.png"))
