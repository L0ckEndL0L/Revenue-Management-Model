"""
evaluation.py
Forecast evaluation, scenario comparison, and Week 4 chart helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
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


def detect_prediction_identity_warning(actual: pd.Series, predicted: pd.Series) -> str:
    """Return a warning when predictions are suspiciously identical to actuals."""
    actual_values = pd.to_numeric(actual, errors="coerce").astype(float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").astype(float)
    mask = actual_values.notna() & predicted_values.notna()
    actual_values = actual_values[mask]
    predicted_values = predicted_values[mask]

    if len(actual_values) < 7:
        return ""

    max_abs_error = float(np.max(np.abs(predicted_values.to_numpy() - actual_values.to_numpy())))
    if max_abs_error <= 1e-6:
        return "Potential leakage: predictions are identical to actual rooms sold across the backtest window"
    return ""


def build_model_comparison_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline and tailored backtest accuracy using MAE and RMSE."""
    rows = []
    model_columns = [
        ("Baseline Model", "baseline_rooms_sold"),
        ("Tailored Model", "enhanced_rooms_sold"),
    ]
    actual = backtest_df.get("actual_rooms_sold", pd.Series(dtype=float))

    for model_name, prediction_col in model_columns:
        predicted = backtest_df.get(prediction_col, pd.Series(dtype=float))
        metrics = calculate_forecast_metrics(
            actual=actual,
            predicted=predicted,
        )
        rows.append(
            {
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "backtest_rows": int(len(backtest_df)),
                "validation_warning": detect_prediction_identity_warning(actual, predicted),
            }
        )

    return pd.DataFrame(rows)


def build_subgroup_backtest_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Build MAE/RMSE subgroup comparison by property type, event period, month, and day type."""
    columns = [
        "property_type",
        "event_period",
        "month",
        "day_type",
        "model",
        "mae",
        "rmse",
        "backtest_rows",
        "validation_warning",
    ]
    if len(backtest_df) == 0:
        return pd.DataFrame(columns=columns)

    df = backtest_df.copy()
    if "property_type" not in df.columns:
        df["property_type"] = "Unspecified"
    df["property_type"] = df["property_type"].fillna("Unspecified").astype(str)

    if "event_period" not in df.columns:
        event_flag = pd.to_numeric(df.get("event_flag", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        df["event_period"] = np.where(event_flag > 0, "Event period", "Non-event period")
    df["event_period"] = df["event_period"].fillna("Non-event period").astype(str)

    if "month" not in df.columns:
        if "stay_date" in df.columns:
            df["month"] = pd.to_datetime(df["stay_date"], errors="coerce").dt.month_name()
        else:
            df["month"] = "Unspecified"
    df["month"] = df["month"].fillna("Unspecified").astype(str)

    if "day_type" not in df.columns:
        if "is_weekend" in df.columns:
            is_weekend = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0)
            df["day_type"] = np.where(is_weekend == 1, "Weekend", "Weekday")
        elif "stay_date" in df.columns:
            stay_dates = pd.to_datetime(df["stay_date"], errors="coerce")
            df["day_type"] = np.where(stay_dates.dt.dayofweek.isin([4, 5]), "Weekend", "Weekday")
        else:
            df["day_type"] = "Unspecified"
    df["day_type"] = df["day_type"].fillna("Unspecified").astype(str)

    rows = []
    group_cols = ["property_type", "event_period", "month", "day_type"]
    for group_key, group in df.groupby(group_cols, dropna=False):
        property_type, event_period, month, day_type = group_key
        for model_name, prediction_col in [
            ("Baseline Model", "baseline_rooms_sold"),
            ("Tailored Model", "enhanced_rooms_sold"),
        ]:
            actual = group.get("actual_rooms_sold", pd.Series(dtype=float))
            predicted = group.get(prediction_col, pd.Series(dtype=float))
            metrics = calculate_forecast_metrics(
                actual=actual,
                predicted=predicted,
            )
            rows.append(
                {
                    "property_type": property_type,
                    "event_period": event_period,
                    "month": month,
                    "day_type": day_type,
                    "model": model_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "backtest_rows": int(len(group)),
                    "validation_warning": detect_prediction_identity_warning(actual, predicted),
                }
            )

    return pd.DataFrame(rows, columns=columns)


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
        plt.plot(chart_df["stay_date"], chart_df["enhanced_rooms_sold"], label="Tailored", linestyle="-.")
    plt.title("Forecast vs Actual (Rooms Sold)")
    plt.xlabel("Stay Date")
    plt.ylabel("Rooms Sold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_comparison_metrics(metrics_df: pd.DataFrame, output_path: str) -> None:
    """Plot MAE and RMSE side by side for baseline and tailored models."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    if len(metrics_df) == 0:
        plt.title("Baseline vs Tailored Model Accuracy")
        plt.text(0.5, 0.5, "No backtest metrics available", ha="center", va="center")
        plt.axis("off")
    else:
        chart_df = metrics_df.melt(
            id_vars=["model"],
            value_vars=["mae", "rmse"],
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])
        if len(chart_df) == 0:
            plt.title("Baseline vs Tailored Model Accuracy")
            plt.text(0.5, 0.5, "No valid backtest metrics available", ha="center", va="center")
            plt.axis("off")
        else:
            pivot = chart_df.pivot(index="model", columns="metric", values="value")
            for metric in ["mae", "rmse"]:
                if metric not in pivot.columns:
                    pivot[metric] = np.nan
            pivot[["mae", "rmse"]].plot(kind="bar", ax=plt.gca(), color=["#4c78a8", "#f58518"])
            plt.title("Baseline vs Tailored Model Accuracy")
            plt.xlabel("Model")
            plt.ylabel("Error (Rooms Sold)")
            plt.xticks(rotation=0)
            plt.grid(axis="y", alpha=0.3)
            plt.legend(["MAE", "RMSE"], title="Metric")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_subgroup_backtest_metrics(subgroup_df: pd.DataFrame, output_path: str) -> None:
    """Plot subgroup RMSE by property type, event period, month, and day type."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    if len(subgroup_df) == 0:
        plt.title("Subgroup Backtest RMSE")
        plt.text(0.5, 0.5, "No subgroup metrics available", ha="center", va="center")
        plt.axis("off")
    else:
        chart_df = subgroup_df.copy()
        for col in ["property_type", "event_period", "month", "day_type"]:
            if col not in chart_df.columns:
                chart_df[col] = "Unspecified"
        chart_df["subgroup"] = (
            chart_df["property_type"].astype(str)
            + " | "
            + chart_df["event_period"].astype(str)
            + " | "
            + chart_df["month"].astype(str)
            + " | "
            + chart_df["day_type"].astype(str)
        )
        chart_df = chart_df.dropna(subset=["rmse"])
        if len(chart_df) == 0:
            plt.title("Subgroup Backtest RMSE")
            plt.text(0.5, 0.5, "No valid subgroup metrics available", ha="center", va="center")
            plt.axis("off")
        else:
            pivot = chart_df.pivot_table(index="subgroup", columns="model", values="rmse", aggfunc="mean")
            pivot.plot(kind="bar", ax=plt.gca(), color=["#4c78a8", "#54a24b"])
            plt.title("Subgroup Backtest RMSE")
            plt.xlabel("Property Type | Event Period | Month | Day Type")
            plt.ylabel("RMSE (Rooms Sold)")
            plt.xticks(rotation=35, ha="right")
            plt.grid(axis="y", alpha=0.3)
            plt.legend(title="Model")
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
