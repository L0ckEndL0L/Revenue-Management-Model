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


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

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

    result = pd.DataFrame(rows, columns=columns)
    month_rank = {month: rank for rank, month in enumerate(MONTH_ORDER)}
    model_rank = {"Baseline Model": 0, "Tailored Model": 1}
    day_type_rank = {"Weekday": 0, "Weekend": 1}
    result["_month_rank"] = result["month"].map(month_rank).fillna(len(MONTH_ORDER))
    result["_model_rank"] = result["model"].map(model_rank).fillna(len(model_rank))
    result["_day_type_rank"] = result["day_type"].map(day_type_rank).fillna(len(day_type_rank))
    return (
        result.sort_values(
            ["_month_rank", "property_type", "event_period", "_day_type_rank", "_model_rank"],
            kind="stable",
        )
        .drop(columns=["_month_rank", "_model_rank", "_day_type_rank"])
        .reset_index(drop=True)
    )


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


def _derive_historical_occupancy(df: pd.DataFrame) -> pd.Series:
    occupancy = pd.to_numeric(df.get("occupancy", pd.Series(np.nan, index=df.index)), errors="coerce")
    if occupancy.notna().any() and float(occupancy.max()) > 1.5:
        occupancy = occupancy / 100.0
    if {"rooms_available", "rooms_sold"}.issubset(df.columns):
        available = pd.to_numeric(df["rooms_available"], errors="coerce").replace(0, np.nan)
        sold = pd.to_numeric(df["rooms_sold"], errors="coerce")
        occupancy = occupancy.combine_first(sold / available)
    return occupancy.clip(lower=0.0, upper=1.2)


def _rolling_rate_input(history: pd.DataFrame, target_row: pd.Series) -> pd.DataFrame:
    """Create one date-safe recommendation row using only earlier outcomes."""
    target_date = pd.Timestamp(target_row["stay_date"])
    hist = history.copy().sort_values("stay_date")
    hist_dates = pd.to_datetime(hist["stay_date"], errors="coerce")
    hist_adr = _derive_actual_rate(hist)
    hist_occupancy = _derive_historical_occupancy(hist)
    same_dow = hist_dates.dt.dayofweek == target_date.dayofweek
    recent_same_dow = hist.loc[same_dow].tail(8).index
    recent_all = hist.tail(28).index

    prior_adr = float(hist_adr.loc[recent_same_dow].median()) if len(recent_same_dow) else np.nan
    if not np.isfinite(prior_adr):
        prior_adr = float(hist_adr.loc[recent_all].median())
    prior_occupancy = float(hist_occupancy.loc[recent_same_dow].median()) if len(recent_same_dow) else np.nan
    if not np.isfinite(prior_occupancy):
        prior_occupancy = float(hist_occupancy.loc[recent_all].median())

    target_inventory = pd.to_numeric(pd.Series([target_row.get("rooms_available")]), errors="coerce").iloc[0]
    if pd.isna(target_inventory) or target_inventory <= 0:
        inventory_history = pd.to_numeric(hist.get("rooms_available", pd.Series(dtype=float)), errors="coerce")
        inventory_history = inventory_history[inventory_history > 0]
        target_inventory = float(inventory_history.tail(28).median()) if len(inventory_history) else np.nan

    projected_rooms = prior_occupancy * target_inventory if np.isfinite(prior_occupancy) and np.isfinite(target_inventory) else np.nan
    proxy = {
        "stay_date": target_date,
        "rooms_available": target_inventory,
        "rooms_sold": projected_rooms,
        "occupancy": prior_occupancy,
        "adr": prior_adr,
        "current_rate": prior_adr,
        "room_revenue": projected_rooms * prior_adr if np.isfinite(projected_rooms) and np.isfinite(prior_adr) else np.nan,
    }
    # Event context is knowable before arrival and is safe to retain. Outcome
    # fields such as same-day ADR, revenue, and occupancy are never copied.
    for column in ["event_flag", "event_pct", "impact_level"]:
        if column in target_row.index:
            proxy[column] = target_row.get(column)
    return pd.DataFrame([proxy])


def _rate_candidate_score(
    actual: pd.Series,
    candidate: pd.Series,
    baseline: pd.Series,
) -> tuple[float, float, float, float, float]:
    baseline_metrics = calculate_forecast_metrics(actual, baseline)
    candidate_metrics = calculate_forecast_metrics(actual, candidate)
    baseline_mae = baseline_metrics["mae"]
    baseline_rmse = baseline_metrics["rmse"]
    candidate_mae = candidate_metrics["mae"]
    candidate_rmse = candidate_metrics["rmse"]
    if not all(np.isfinite(value) for value in [baseline_mae, baseline_rmse, candidate_mae, candidate_rmse]):
        return float("inf"), candidate_mae, candidate_rmse, baseline_mae, baseline_rmse
    mae_scale = baseline_mae if baseline_mae > 0 else 1.0
    rmse_scale = baseline_rmse if baseline_rmse > 0 else 1.0
    score = 0.4 * (candidate_mae / mae_scale) + 0.6 * (candidate_rmse / rmse_scale)
    return float(score), candidate_mae, candidate_rmse, baseline_mae, baseline_rmse


def _rate_weekly_wins(actual: pd.Series, candidate: pd.Series, baseline: pd.Series) -> tuple[int, int]:
    wins = 0
    windows = 0
    for start in range(0, len(actual), 7):
        stop = min(start + 7, len(actual))
        score, _, _, _, _ = _rate_candidate_score(
            actual.iloc[start:stop], candidate.iloc[start:stop], baseline.iloc[start:stop]
        )
        if np.isfinite(score):
            windows += 1
            if score < 1.0:
                wins += 1
    return wins, windows


def _calibrated_rate_recommendation(
    prior_rows: list[dict],
    baseline_rate: float,
    raw_tailored_rate: float,
) -> tuple[float, str, int, float, float]:
    """Select a date-safe rate challenger from earlier backtest evidence."""
    calibration_rows = 42
    if len(prior_rows) < calibration_rows:
        return baseline_rate, "baseline_warmup", 0, 1.0, 0.0

    prior = pd.DataFrame(prior_rows[-calibration_rows:]).reset_index(drop=True)
    validation = prior.iloc[14:]
    actual = pd.to_numeric(validation["actual_adr"], errors="coerce").reset_index(drop=True)
    baseline = pd.to_numeric(validation["baseline_recommendation"], errors="coerce").reset_index(drop=True)
    raw_tailored = pd.to_numeric(validation["raw_rateanchor_recommendation"], errors="coerce").reset_index(drop=True)

    candidates: list[tuple[str, pd.Series, float]] = [
        ("raw_tailored", raw_tailored, raw_tailored_rate),
    ]
    for weight in [0.25, 0.50, 0.75]:
        candidates.append(
            (
                f"baseline_tailored_blend_{weight:.2f}",
                baseline * (1.0 - weight) + raw_tailored * weight,
                baseline_rate * (1.0 - weight) + raw_tailored_rate * weight,
            )
        )

    best_name = "baseline"
    best_prediction = baseline_rate
    best_score = 1.0
    for name, validation_prediction, target_prediction in candidates:
        score, candidate_mae, candidate_rmse, baseline_mae, baseline_rmse = _rate_candidate_score(
            actual, validation_prediction, baseline
        )
        wins, windows = _rate_weekly_wins(actual, validation_prediction, baseline)
        required_wins = int(np.ceil(windows * 0.75))
        passes = (
            windows == 4
            and wins >= required_wins
            and score <= 0.97
            and candidate_mae <= baseline_mae
            and candidate_rmse <= baseline_rmse
        )
        if passes and score < best_score:
            best_name = name
            best_prediction = float(target_prediction)
            best_score = float(score)

    if best_name == "baseline":
        return baseline_rate, best_name, calibration_rows, 1.0, 0.0

    improvement = max(0.0, 1.0 - best_score)
    history_cap = 0.50 if len(prior_rows) <= 120 else 0.75
    confidence = history_cap * float(np.clip(improvement / 0.15, 0.0, 1.0))
    final_rate = baseline_rate * (1.0 - confidence) + best_prediction * confidence
    max_deviation = 0.10 + 0.10 * min(confidence / 0.75, 1.0)
    final_rate = float(np.clip(final_rate, baseline_rate * (1.0 - max_deviation), baseline_rate * (1.0 + max_deviation)))
    return final_rate, best_name, calibration_rows, best_score, confidence


def build_rate_backtest_frame(
    historical_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    tailored_settings: dict | None = None,
    min_history_days: int = 28,
) -> pd.DataFrame:
    """Run a rolling rate backtest using only information available beforehand."""
    columns = [
        "stay_date",
        "actual_adr",
        "baseline_recommendation",
        "raw_rateanchor_recommendation",
        "rateanchor_recommendation",
        "baseline_error",
        "rateanchor_error",
        "property_type",
        "event_period",
        "month",
        "day_type",
        "history_rows",
        "rate_input_adr",
        "rate_input_occupancy",
        "selected_rate_model",
        "rate_calibration_rows",
        "rate_calibration_score",
        "rate_confidence_weight",
    ]
    if historical_df is None or len(historical_df) == 0:
        return pd.DataFrame(columns=columns)

    df = historical_df.copy()
    df["stay_date"] = pd.to_datetime(df.get("stay_date"), errors="coerce")
    df = df.dropna(subset=["stay_date"]).sort_values("stay_date").reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame(columns=columns)

    df["actual_adr"] = _derive_actual_rate(df)
    event_dates: set[pd.Timestamp] = set()
    if events_df is not None and len(events_df) > 0 and "stay_date" in events_df.columns:
        event_dates = set(pd.to_datetime(events_df["stay_date"], errors="coerce").dropna().dt.normalize())

    rows: list[dict] = []
    start = max(1, int(min_history_days))
    for target_index in range(start, len(df)):
        history = df.iloc[:target_index].copy()
        target_row = df.iloc[target_index]
        proxy_input = _rolling_rate_input(history, target_row)
        baseline_df = generate_baseline_pricing_recommendations(proxy_input, historical_df=history)
        tailored_df = build_tailored_recommendations(proxy_input, baseline_df, tailored_settings)

        stay_date = pd.Timestamp(target_row["stay_date"])
        event_flag = int(pd.to_numeric(pd.Series([target_row.get("event_flag", 0)]), errors="coerce").fillna(0).iloc[0])
        is_event = stay_date.normalize() in event_dates if event_dates else event_flag > 0
        actual_adr = pd.to_numeric(pd.Series([target_row.get("actual_adr")]), errors="coerce").iloc[0]
        baseline_rate = pd.to_numeric(baseline_df["baseline_recommended_rate"], errors="coerce").iloc[0]
        raw_tailored_rate = pd.to_numeric(tailored_df["tailored_recommendation"], errors="coerce").iloc[0]
        tailored_rate, selected_rate_model, rate_calibration_rows, rate_calibration_score, rate_confidence = (
            _calibrated_rate_recommendation(rows, baseline_rate, raw_tailored_rate)
        )
        property_type = tailored_df["property_type"].iloc[0] if "property_type" in tailored_df.columns else "Unspecified"
        rows.append(
            {
                "stay_date": stay_date,
                "actual_adr": actual_adr,
                "baseline_recommendation": baseline_rate,
                "raw_rateanchor_recommendation": raw_tailored_rate,
                "rateanchor_recommendation": tailored_rate,
                "baseline_error": baseline_rate - actual_adr,
                "rateanchor_error": tailored_rate - actual_adr,
                "property_type": str(property_type or "Unspecified"),
                "event_period": "Event period" if is_event else "Non-event period",
                "month": stay_date.month_name(),
                "day_type": "Weekend" if stay_date.dayofweek in {4, 5} else "Weekday",
                "history_rows": len(history),
                "rate_input_adr": proxy_input["adr"].iloc[0],
                "rate_input_occupancy": proxy_input["occupancy"].iloc[0],
                "selected_rate_model": selected_rate_model,
                "rate_calibration_rows": rate_calibration_rows,
                "rate_calibration_score": rate_calibration_score,
                "rate_confidence_weight": rate_confidence,
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values("stay_date").reset_index(drop=True)


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
    """Build rate-error comparisons by property, event, month, and day type."""
    columns = ["property_type", "event_period", "month", "day_type", "model", "mae", "rmse", "backtest_rows"]
    if rate_backtest_df is None or len(rate_backtest_df) == 0:
        return pd.DataFrame(columns=columns)

    df = rate_backtest_df.copy()
    if "property_type" not in df.columns:
        df["property_type"] = "Unspecified"
    if "event_period" not in df.columns:
        df["event_period"] = "Non-event period"
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df.get("stay_date"), errors="coerce").dt.month_name().fillna("Unspecified")
    if "day_type" not in df.columns:
        stay_dates = pd.to_datetime(df.get("stay_date"), errors="coerce")
        df["day_type"] = np.where(stay_dates.dt.dayofweek.isin([4, 5]), "Weekend", "Weekday")
    df["property_type"] = df["property_type"].fillna("Unspecified").astype(str)
    df["event_period"] = df["event_period"].fillna("Non-event period").astype(str)
    df["month"] = df["month"].fillna("Unspecified").astype(str)
    df["day_type"] = df["day_type"].fillna("Unspecified").astype(str)

    rows = []
    group_cols = ["property_type", "event_period", "month", "day_type"]
    for (property_type, event_period, month, day_type), group in df.groupby(group_cols, dropna=False):
        for model_name, prediction_col in [
            ("Baseline Model", "baseline_recommendation"),
            ("RateAnchor Tailored Model", "rateanchor_recommendation"),
        ]:
            metrics = calculate_forecast_metrics(group.get("actual_adr", pd.Series(dtype=float)), group.get(prediction_col, pd.Series(dtype=float)))
            rows.append(
                {
                    "property_type": property_type,
                    "event_period": event_period,
                    "month": month,
                    "day_type": day_type,
                    "model": model_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "backtest_rows": int(len(group.dropna(subset=["actual_adr", prediction_col])) if prediction_col in group.columns else 0),
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    month_rank = {month: rank for rank, month in enumerate(MONTH_ORDER)}
    model_rank = {"Baseline Model": 0, "RateAnchor Tailored Model": 1}
    day_type_rank = {"Weekday": 0, "Weekend": 1}
    result["_month_rank"] = result["month"].map(month_rank).fillna(len(MONTH_ORDER))
    result["_model_rank"] = result["model"].map(model_rank).fillna(len(model_rank))
    result["_day_type_rank"] = result["day_type"].map(day_type_rank).fillna(len(day_type_rank))
    return (
        result.sort_values(
            ["_month_rank", "property_type", "event_period", "_day_type_rank", "_model_rank"],
            kind="stable",
        )
        .drop(columns=["_month_rank", "_model_rank", "_day_type_rank"])
        .reset_index(drop=True)
    )


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
