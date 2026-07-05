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

from src.baseline import generate_baseline_pricing_recommendations
from src.tailored import build_tailored_recommendations


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


def _derive_actual_rate(df: pd.DataFrame) -> pd.Series:
    actual = pd.Series(np.nan, index=df.index, dtype=float)
    for column in ["actual_adr", "adr", "current_rate"]:
        if column in df.columns:
            actual = actual.combine_first(pd.to_numeric(df[column], errors="coerce"))
    if {"room_revenue", "rooms_sold"}.issubset(df.columns):
        revenue = pd.to_numeric(df["room_revenue"], errors="coerce")
        sold = pd.to_numeric(df["rooms_sold"], errors="coerce").replace(0, np.nan)
        actual = actual.combine_first(revenue / sold)
    return actual


def build_rate_backtest_frame(
    historical_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    tailored_settings: dict | None = None,
) -> pd.DataFrame:
    """Build historical ADR/rate rows with baseline and RateAnchor recommendations."""
    columns = [
        "stay_date",
        "actual_adr",
        "baseline_recommendation",
        "rateanchor_recommendation",
        "baseline_error",
        "rateanchor_error",
        "property_type",
        "event_period",
    ]
    if historical_df is None or len(historical_df) == 0:
        return pd.DataFrame(columns=columns)

    df = historical_df.copy()
    df["stay_date"] = pd.to_datetime(df.get("stay_date"), errors="coerce")
    df = df.dropna(subset=["stay_date"]).sort_values("stay_date").reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame(columns=columns)

    df["actual_adr"] = _derive_actual_rate(df)
    baseline_df = generate_baseline_pricing_recommendations(df, historical_df=df)
    tailored_df = build_tailored_recommendations(df, baseline_df, tailored_settings)

    out = df[["stay_date", "actual_adr"]].copy()
    out = out.merge(
        baseline_df[["stay_date", "baseline_recommended_rate"]],
        on="stay_date",
        how="left",
    ).rename(columns={"baseline_recommended_rate": "baseline_recommendation"})
    out = out.merge(
        tailored_df[["stay_date", "tailored_recommendation", "property_type"]],
        on="stay_date",
        how="left",
    ).rename(columns={"tailored_recommendation": "rateanchor_recommendation"})

    if "property_type" not in out.columns:
        out["property_type"] = str((tailored_settings or {}).get("property_type", "Unspecified")).strip() or "Unspecified"
    out["property_type"] = out["property_type"].fillna("Unspecified").astype(str)

    out["event_period"] = "Non-event period"
    if events_df is not None and len(events_df) > 0 and "stay_date" in events_df.columns:
        event_dates = pd.to_datetime(events_df["stay_date"], errors="coerce").dropna().dt.normalize()
        out["event_period"] = np.where(
            pd.to_datetime(out["stay_date"], errors="coerce").dt.normalize().isin(set(event_dates)),
            "Event period",
            "Non-event period",
        )
    elif "event_flag" in df.columns:
        out["event_period"] = np.where(pd.to_numeric(df["event_flag"], errors="coerce").fillna(0) > 0, "Event period", "Non-event period")

    out["baseline_error"] = pd.to_numeric(out["baseline_recommendation"], errors="coerce") - pd.to_numeric(out["actual_adr"], errors="coerce")
    out["rateanchor_error"] = pd.to_numeric(out["rateanchor_recommendation"], errors="coerce") - pd.to_numeric(out["actual_adr"], errors="coerce")
    return out[columns].sort_values("stay_date").reset_index(drop=True)


def build_rate_backtest_metrics(rate_backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Compare actual ADR/rate against baseline and RateAnchor recommendations."""
    rows = []
    actual = rate_backtest_df.get("actual_adr", pd.Series(dtype=float))
    model_columns = [
        ("Baseline Model", "baseline_recommendation"),
        ("RateAnchor Tailored Model", "rateanchor_recommendation"),
    ]
    baseline_mae = np.nan
    baseline_rmse = np.nan
    for model_name, prediction_col in model_columns:
        metrics = calculate_forecast_metrics(actual=actual, predicted=rate_backtest_df.get(prediction_col, pd.Series(dtype=float)))
        if model_name == "Baseline Model":
            baseline_mae = metrics["mae"]
            baseline_rmse = metrics["rmse"]
        mae_delta = metrics["mae"] - baseline_mae if pd.notna(metrics["mae"]) and pd.notna(baseline_mae) else np.nan
        rmse_delta = metrics["rmse"] - baseline_rmse if pd.notna(metrics["rmse"]) and pd.notna(baseline_rmse) else np.nan
        rows.append(
            {
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mae_difference_vs_baseline": mae_delta,
                "rmse_difference_vs_baseline": rmse_delta,
                "mae_improvement_vs_baseline": -mae_delta if pd.notna(mae_delta) else np.nan,
                "rmse_improvement_vs_baseline": -rmse_delta if pd.notna(rmse_delta) else np.nan,
                "backtest_rows": int((pd.to_numeric(actual, errors="coerce").notna() & pd.to_numeric(rate_backtest_df.get(prediction_col, pd.Series(dtype=float)), errors="coerce").notna()).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_rate_subgroup_backtest_metrics(rate_backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Build ADR/rate MAE/RMSE subgroup comparison by property type and event period."""
    columns = ["property_type", "event_period", "model", "mae", "rmse", "backtest_rows"]
    if rate_backtest_df is None or len(rate_backtest_df) == 0:
        return pd.DataFrame(columns=columns)

    df = rate_backtest_df.copy()
    if "property_type" not in df.columns:
        df["property_type"] = "Unspecified"
    if "event_period" not in df.columns:
        df["event_period"] = "Non-event period"
    df["property_type"] = df["property_type"].fillna("Unspecified").astype(str)
    df["event_period"] = df["event_period"].fillna("Non-event period").astype(str)

    rows = []
    for (property_type, event_period), group in df.groupby(["property_type", "event_period"], dropna=False):
        for model_name, prediction_col in [
            ("Baseline Model", "baseline_recommendation"),
            ("RateAnchor Tailored Model", "rateanchor_recommendation"),
        ]:
            metrics = calculate_forecast_metrics(group.get("actual_adr", pd.Series(dtype=float)), group.get(prediction_col, pd.Series(dtype=float)))
            rows.append(
                {
                    "property_type": property_type,
                    "event_period": event_period,
                    "model": model_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "backtest_rows": int(len(group.dropna(subset=["actual_adr", prediction_col])) if prediction_col in group.columns else 0),
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
