"""
evaluation.py
Forecast evaluation, scenario comparison, and Week 4 chart helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_forecast_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and directional accuracy."""
    actual_values = pd.to_numeric(actual, errors="coerce").astype(float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").astype(float)

    mask = actual_values.notna() & predicted_values.notna()
    actual_values = actual_values[mask]
    predicted_values = predicted_values[mask]

    if len(actual_values) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "directional_accuracy": np.nan,
        }

    errors = predicted_values - actual_values
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    denom = actual_values.replace(0, np.nan)
    mape = float((np.abs(errors) / denom).mean() * 100) if denom.notna().any() else np.nan

    if len(actual_values) >= 2:
        actual_direction = np.sign(np.diff(actual_values.to_numpy()))
        predicted_direction = np.sign(np.diff(predicted_values.to_numpy()))
        directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)
    else:
        directional_accuracy = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }


def plot_forecast_vs_actual(forecast_df: pd.DataFrame, output_path: str) -> None:
    """Plot actual rooms sold against baseline and enhanced predictions."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if len(forecast_df) == 0:
        plt.figure(figsize=(12, 6))
        plt.title("Forecast vs Actual (Rooms Sold)")
        plt.text(0.5, 0.5, "No backtest data available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return

    chart_df = forecast_df.sort_values("stay_date").copy()
    plt.figure(figsize=(12, 6))
    plt.plot(chart_df["stay_date"], chart_df["actual_rooms_sold"], label="Actual", linewidth=2)
    if "baseline_rooms_sold" in chart_df.columns:
        plt.plot(chart_df["stay_date"], chart_df["baseline_rooms_sold"], label="Baseline", linestyle="--")
    if "enhanced_rooms_sold" in chart_df.columns:
        plt.plot(chart_df["stay_date"], chart_df["enhanced_rooms_sold"], label="Enhanced", linestyle="-.")
    plt.title("Forecast vs Actual (Rooms Sold)")
    plt.xlabel("Stay Date")
    plt.ylabel("Rooms Sold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_policy_evaluation_metrics(
    forecast_metrics: Dict[str, float],
    projected_uplift_vs_baseline: float,
) -> pd.DataFrame:
    """Build compact evaluation output for forecast quality and pricing uplift."""
    return pd.DataFrame(
        [
            {
                "metric": "forecast_mae",
                "value": forecast_metrics.get("mae", np.nan),
            },
            {
                "metric": "forecast_mape",
                "value": forecast_metrics.get("mape", np.nan),
            },
            {
                "metric": "projected_revenue_uplift_vs_baseline",
                "value": projected_uplift_vs_baseline,
            },
        ]
    )


def plot_current_vs_recommended_rate(rate_df: pd.DataFrame, output_path: str) -> None:
    """Plot current rate versus recommended rate by stay date."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    if len(rate_df) == 0:
        plt.title("Current vs Recommended Rate")
        plt.text(0.5, 0.5, "No recommendation data available", ha="center", va="center")
        plt.axis("off")
    else:
        chart_df = rate_df.sort_values("stay_date")
        plt.plot(chart_df["stay_date"], chart_df["current_rate"], label="Current Rate", linewidth=2)
        plt.plot(chart_df["stay_date"], chart_df["recommended_rate"], label="Recommended Rate", linestyle="--")
        plt.title("Current vs Recommended Rate")
        plt.xlabel("Stay Date")
        plt.ylabel("Rate")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_expected_revenue_uplift(rate_df: pd.DataFrame, output_path: str) -> None:
    """Plot expected revenue uplift versus current rate scenario."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    if len(rate_df) == 0:
        plt.title("Expected Revenue Uplift")
        plt.text(0.5, 0.5, "No recommendation data available", ha="center", va="center")
        plt.axis("off")
    else:
        chart_df = rate_df.sort_values("stay_date")
        plt.bar(chart_df["stay_date"].astype(str), chart_df["uplift_vs_current"], color="#4c78a8")
        plt.title("Expected Revenue Uplift vs Current Rate")
        plt.xlabel("Stay Date")
        plt.ylabel("Expected Uplift")
        plt.xticks(rotation=60)
        plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_priority_score_by_date(priority_df: pd.DataFrame, output_path: str) -> None:
    """Plot priority score by stay date."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    if len(priority_df) == 0:
        plt.title("Priority Score by Date")
        plt.text(0.5, 0.5, "No priority data available", ha="center", va="center")
        plt.axis("off")
    else:
        chart_df = priority_df.sort_values("stay_date")
        plt.plot(chart_df["stay_date"], chart_df["priority_score"], marker="o")
        plt.title("Priority Score by Stay Date")
        plt.xlabel("Stay Date")
        plt.ylabel("Priority Score")
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
