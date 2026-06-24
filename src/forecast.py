"""
forecast.py
Baseline and enhanced forecasting models for the Hotel RMS pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestRegressor = None
    LinearRegression = None
    SKLEARN_AVAILABLE = False


def prepare_forecast_frame(
    daily_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    stly_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build feature-rich daily frame for forecasting."""
    df = daily_df.copy().sort_values("stay_date").reset_index(drop=True)
    df["stay_date"] = pd.to_datetime(df["stay_date"], errors="coerce")
    df = df.dropna(subset=["stay_date"])

    df["dow"] = df["stay_date"].dt.dayofweek
    df["month"] = df["stay_date"].dt.month
    df["is_weekend"] = df["dow"].isin([4, 5]).astype(int)
    df["day_type"] = np.where(df["is_weekend"] == 1, "Weekend", "Weekday")

    if "rooms_available_derived_from_occupancy" in df.columns:
        derived_mask = df["rooms_available_derived_from_occupancy"].fillna(False).astype(bool)
        if derived_mask.any() and "rooms_available" in df.columns:
            source_inventory = pd.to_numeric(df.loc[~derived_mask, "rooms_available"], errors="coerce")
            source_inventory = source_inventory[source_inventory > 0]
            all_inventory = pd.to_numeric(df["rooms_available"], errors="coerce")
            all_inventory = all_inventory[all_inventory > 0]
            if len(source_inventory) > 0:
                stable_inventory = float(source_inventory.median())
            elif len(all_inventory) > 0:
                stable_inventory = float(all_inventory.median())
            else:
                stable_inventory = np.nan
            if pd.notna(stable_inventory):
                df.loc[derived_mask, "rooms_available"] = int(round(stable_inventory))

    if events_df is not None and len(events_df) > 0:
        event_flags = events_df[["stay_date"]].copy()
        event_flags["stay_date"] = pd.to_datetime(event_flags["stay_date"], errors="coerce")
        event_flags = event_flags.dropna().drop_duplicates(subset=["stay_date"])
        event_flags["event_flag"] = 1
        df = df.merge(event_flags, on="stay_date", how="left")
        df["event_flag"] = df["event_flag"].fillna(0).astype(int)
    else:
        df["event_flag"] = 0

    if stly_df is not None and len(stly_df) > 0 and "stly_occupancy" in stly_df.columns:
        stly = stly_df[["stay_date", "stly_occupancy"]].copy()
        stly["stay_date"] = pd.to_datetime(stly["stay_date"], errors="coerce")
        stly = stly.dropna(subset=["stay_date"])
        stly["stly_year"] = stly["stay_date"].dt.year
        stly["calendar_key"] = stly["stay_date"].dt.strftime("%m-%d")

        df["calendar_key"] = df["stay_date"].dt.strftime("%m-%d")
        df["target_year"] = df["stay_date"].dt.year
        df = df.merge(stly[["calendar_key", "stly_year", "stly_occupancy"]], on="calendar_key", how="left")
        valid_prior_year = df["stly_year"].isna() | (df["stly_year"] < df["target_year"])
        df.loc[~valid_prior_year, "stly_occupancy"] = np.nan
        df.loc[~valid_prior_year, "stly_year"] = np.nan
        df["has_valid_stly"] = df["stly_occupancy"].notna().astype(int)
        df = (
            df.sort_values(["stay_date", "has_valid_stly", "stly_year"])
            .drop_duplicates(subset=["stay_date"], keep="last")
            .drop(columns=["calendar_key", "target_year", "stly_year", "has_valid_stly"])
        )
    else:
        df["stly_occupancy"] = np.nan

    if "occupancy" in df.columns:
        df["stly_occupancy"] = df["stly_occupancy"].fillna(df["occupancy"].median())
    else:
        df["stly_occupancy"] = df["stly_occupancy"].fillna(0.0)

    return df


def baseline_forecast(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Forecast rooms sold and revenue using a day-of-week average baseline."""
    if len(target_df) == 0:
        return target_df.assign(
            forecast_rooms_sold=np.nan,
            forecast_revenue=np.nan,
            forecast_occupancy=np.nan,
            forecast_adr=np.nan,
            model_name="baseline",
        )

    if len(train_df) == 0:
        # Safe fallback if no train history is available.
        global_rooms = 0.0
        global_adr = 0.0
        by_dow_rooms = pd.Series(dtype=float)
        by_dow_adr = pd.Series(dtype=float)
    else:
        by_dow_rooms = train_df.groupby("dow")["rooms_sold"].mean()
        by_dow_adr = train_df.groupby("dow")["adr"].mean()
        global_rooms = float(train_df["rooms_sold"].mean())
        global_adr = float(train_df["adr"].mean())

    out = target_df.copy()
    out["forecast_rooms_sold"] = out["dow"].map(by_dow_rooms).fillna(global_rooms).clip(lower=0.0)
    out["forecast_adr"] = out["dow"].map(by_dow_adr).fillna(global_adr).clip(lower=0.0)
    out["forecast_revenue"] = out["forecast_rooms_sold"] * out["forecast_adr"]
    out["forecast_occupancy"] = np.where(
        out["rooms_available"] > 0,
        out["forecast_rooms_sold"] / out["rooms_available"],
        0.0,
    )
    out["model_name"] = "baseline"
    return out


def _select_model(train_size: int):
    if not SKLEARN_AVAILABLE:
        return None
    if train_size >= 45:
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    return LinearRegression()


def enhanced_forecast(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Forecast rooms sold via regression using occupancy, date, and event features."""
    if len(target_df) == 0:
        return target_df.assign(
            forecast_rooms_sold=np.nan,
            forecast_revenue=np.nan,
            forecast_occupancy=np.nan,
            forecast_adr=np.nan,
            model_name="enhanced",
        )

    feature_cols = ["rooms_available", "dow", "month", "is_weekend", "event_flag", "stly_occupancy"]

    if not SKLEARN_AVAILABLE:
        # Fallback path if scikit-learn is unavailable in the active environment.
        return baseline_forecast(train_df=train_df, target_df=target_df).assign(
            model_name="enhanced_fallback_no_sklearn"
        )

    if len(train_df) < 10:
        # Avoid unstable ML fits on very short history.
        return baseline_forecast(train_df=train_df, target_df=target_df).assign(model_name="enhanced_fallback")

    x_train = train_df[feature_cols].fillna(0.0)
    y_rooms = train_df["rooms_sold"].fillna(0.0)
    y_adr = train_df["adr"].fillna(0.0)

    rooms_model = _select_model(len(train_df))
    adr_model = _select_model(len(train_df))
    rooms_model.fit(x_train, y_rooms)
    adr_model.fit(x_train, y_adr)

    out = target_df.copy()
    x_target = out[feature_cols].fillna(0.0)
    # Clip predictions at zero to keep business-valid outputs.
    out["forecast_rooms_sold"] = np.maximum(rooms_model.predict(x_target), 0.0)
    out["forecast_adr"] = np.maximum(adr_model.predict(x_target), 0.0)
    out["forecast_revenue"] = out["forecast_rooms_sold"] * out["forecast_adr"]
    out["forecast_occupancy"] = np.where(
        out["rooms_available"] > 0,
        out["forecast_rooms_sold"] / out["rooms_available"],
        0.0,
    )
    out["model_name"] = "enhanced"
    return out


def _error_mae(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce").astype(float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").astype(float)
    mask = actual_values.notna() & predicted_values.notna()
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(predicted_values[mask] - actual_values[mask])))


def _error_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce").astype(float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").astype(float)
    mask = actual_values.notna() & predicted_values.notna()
    if not mask.any():
        return float("inf")
    errors = predicted_values[mask] - actual_values[mask]
    return float(np.sqrt(np.mean(np.square(errors))))


def _smoothed_group_bias(
    calibration: pd.DataFrame,
    target: pd.DataFrame,
    *,
    group_cols: list[str],
    baseline_values: pd.Series,
    baseline_target_values: pd.Series,
    actual_values: pd.Series,
    global_bias: float,
    min_group_rows: int = 2,
    smoothing_prior: float = 3.0,
) -> tuple[pd.Series, pd.Series] | None:
    """Build conservative residual correction by subgroup."""
    if not set(group_cols).issubset(calibration.columns) or not set(group_cols).issubset(target.columns):
        return None

    cal_context = calibration.reset_index(drop=True).copy()
    target_context = target.reset_index(drop=True).copy()
    cal_context["_actual"] = actual_values.reset_index(drop=True)
    cal_context["_baseline"] = baseline_values.reset_index(drop=True)
    cal_context["_residual"] = cal_context["_actual"] - cal_context["_baseline"]
    grouped = cal_context.groupby(group_cols, dropna=False)["_residual"].agg(["sum", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= min_group_rows].copy()
    if len(grouped) == 0:
        return None

    grouped["_bias"] = (grouped["sum"] + global_bias * smoothing_prior) / (grouped["count"] + smoothing_prior)
    selected_rows = []
    for _, row in grouped.iterrows():
        mask = pd.Series(True, index=cal_context.index)
        for col in group_cols:
            mask &= cal_context[col].eq(row[col])
        group_actual = cal_context.loc[mask, "_actual"]
        group_baseline = cal_context.loc[mask, "_baseline"]
        group_adjusted = group_baseline + float(row["_bias"])
        base_mae = _error_mae(group_actual, group_baseline)
        base_rmse = _error_rmse(group_actual, group_baseline)
        adj_mae = _error_mae(group_actual, group_adjusted)
        adj_rmse = _error_rmse(group_actual, group_adjusted)
        mae_scale = base_mae if np.isfinite(base_mae) and base_mae > 0 else 1.0
        rmse_scale = base_rmse if np.isfinite(base_rmse) and base_rmse > 0 else 1.0
        score = 0.3 * (adj_mae / mae_scale) + 0.7 * (adj_rmse / rmse_scale)
        if np.isfinite(score) and score < 1.0:
            selected_rows.append(row)

    if not selected_rows:
        return None

    selected = pd.DataFrame(selected_rows)[group_cols + ["_bias"]]
    cal_bias = cal_context.merge(selected, on=group_cols, how="left")["_bias"].fillna(0.0)
    target_bias = target_context.merge(selected, on=group_cols, how="left")["_bias"].fillna(0.0)

    return (
        baseline_values.reset_index(drop=True) + cal_bias,
        baseline_target_values.reset_index(drop=True) + target_bias,
    )


def calibrated_tailored_forecast(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast with validation-gated tailored logic.

    The raw enhanced model can overfit when history is short. This wrapper tests
    baseline, enhanced, blended, and bias-corrected candidates on the most recent
    training slice, then applies the best validated choice to the target period.
    """
    baseline_target = baseline_forecast(train_df=train_df, target_df=target_df)
    if len(target_df) == 0:
        return baseline_target.assign(model_name="tailored_calibrated")

    if len(train_df) < 14:
        return baseline_target.assign(model_name="tailored_calibrated_baseline_short_history")

    calibration_size = min(7, max(3, int(round(len(train_df) * 0.30))))
    if len(train_df) - calibration_size < 5:
        return baseline_target.assign(model_name="tailored_calibrated_baseline_short_history")

    core_train = train_df.iloc[:-calibration_size].copy()
    calibration = train_df.iloc[-calibration_size:].copy()

    baseline_cal = baseline_forecast(train_df=core_train, target_df=calibration)
    actual_cal = calibration["rooms_sold"]

    baseline_values = baseline_cal["forecast_rooms_sold"].reset_index(drop=True)
    actual_values = actual_cal.reset_index(drop=True)
    bias_adjustment = float((actual_values - baseline_values).mean())

    candidate_predictions: list[tuple[str, pd.Series, pd.Series]] = [
        (
            "baseline",
            baseline_values,
            baseline_target["forecast_rooms_sold"].reset_index(drop=True),
        )
    ]

    for name, group_cols in [
        ("baseline_month_bias_corrected", ["month"]),
        ("baseline_day_type_bias_corrected", ["day_type"]),
        ("baseline_month_day_type_bias_corrected", ["month", "day_type"]),
    ]:
        group_candidate = _smoothed_group_bias(
            calibration=calibration,
            target=target_df,
            group_cols=group_cols,
            baseline_values=baseline_values,
            baseline_target_values=baseline_target["forecast_rooms_sold"].reset_index(drop=True),
            actual_values=actual_values,
            global_bias=bias_adjustment,
        )
        if group_candidate is not None:
            candidate_predictions.append((name, group_candidate[0], group_candidate[1]))

    if len(core_train) >= 45:
        enhanced_cal = enhanced_forecast(train_df=core_train, target_df=calibration)
        enhanced_target = enhanced_forecast(train_df=train_df, target_df=target_df)
        enhanced_values = enhanced_cal["forecast_rooms_sold"].reset_index(drop=True)
        enhanced_target_values = enhanced_target["forecast_rooms_sold"].reset_index(drop=True)
        candidate_predictions.append(("enhanced", enhanced_values, enhanced_target_values))
        for weight in [0.25, 0.50, 0.75]:
            candidate_predictions.append(
                (
                    f"baseline_enhanced_blend_{weight:.2f}",
                    baseline_values * (1.0 - weight) + enhanced_values * weight,
                    baseline_target["forecast_rooms_sold"].reset_index(drop=True) * (1.0 - weight)
                    + enhanced_target_values * weight,
                )
            )

    if len(core_train) >= 45 and {"stly_occupancy", "rooms_available"}.issubset(calibration.columns) and {"stly_occupancy", "rooms_available"}.issubset(target_df.columns):
        stly_cal_values = (
            pd.to_numeric(calibration["stly_occupancy"], errors="coerce")
            * pd.to_numeric(calibration["rooms_available"], errors="coerce")
        ).reset_index(drop=True)
        stly_target_values = (
            pd.to_numeric(target_df["stly_occupancy"], errors="coerce")
            * pd.to_numeric(target_df["rooms_available"], errors="coerce")
        ).reset_index(drop=True)
        if stly_cal_values.notna().any() and stly_target_values.notna().any():
            stly_bias = float((actual_values - stly_cal_values).dropna().mean())
            candidate_predictions.append(("stly_prior_year", stly_cal_values, stly_target_values))
            candidate_predictions.append(("stly_prior_year_bias_corrected", stly_cal_values + stly_bias, stly_target_values + stly_bias))
            for weight in [0.25, 0.50, 0.75]:
                candidate_predictions.append(
                    (
                        f"baseline_stly_blend_{weight:.2f}",
                        baseline_values * (1.0 - weight) + stly_cal_values * weight,
                        baseline_target["forecast_rooms_sold"].reset_index(drop=True) * (1.0 - weight)
                        + stly_target_values * weight,
                    )
                )

    best_name = "baseline"
    best_target_prediction = baseline_target["forecast_rooms_sold"].reset_index(drop=True)
    baseline_mae = _error_mae(actual_values, baseline_values)
    baseline_rmse = _error_rmse(actual_values, baseline_values)
    mae_scale = baseline_mae if np.isfinite(baseline_mae) and baseline_mae > 0 else 1.0
    rmse_scale = baseline_rmse if np.isfinite(baseline_rmse) and baseline_rmse > 0 else 1.0
    best_score = 1.0
    for name, calibration_prediction, target_prediction in candidate_predictions:
        mae = _error_mae(actual_values, calibration_prediction)
        rmse = _error_rmse(actual_values, calibration_prediction)
        if not np.isfinite(mae) or not np.isfinite(rmse):
            continue
        score = 0.3 * (mae / mae_scale) + 0.7 * (rmse / rmse_scale)
        if score < best_score:
            best_name = name
            best_target_prediction = target_prediction
            best_score = score

    out = baseline_target.copy()
    tailored_prediction = best_target_prediction.fillna(baseline_target["forecast_rooms_sold"].reset_index(drop=True))
    out["forecast_rooms_sold"] = tailored_prediction.clip(lower=0.0).to_numpy()
    if "rooms_available" in out.columns:
        out["forecast_rooms_sold"] = np.minimum(
            out["forecast_rooms_sold"],
            pd.to_numeric(out["rooms_available"], errors="coerce").fillna(np.inf),
        )
    out["forecast_revenue"] = out["forecast_rooms_sold"] * out["forecast_adr"]
    out["forecast_occupancy"] = np.where(
        out["rooms_available"] > 0,
        out["forecast_rooms_sold"] / out["rooms_available"],
        0.0,
    )
    out["model_name"] = f"tailored_calibrated_{best_name}"
    return out


def build_backtest_sets(
    model_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    min_test_days: int = 21,
    max_test_days: int = 30,
    min_train_days: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological train/test split for forecast evaluation."""
    history = model_df[model_df["stay_date"] <= as_of_date].copy().sort_values("stay_date")

    if len(history) <= min_train_days:
        return history.iloc[0:0].copy(), history.copy()

    available_test_days = max(1, len(history) - min_train_days)
    target_test_days = max(min_test_days, int(round(len(history) * 0.2)))
    test_size = min(max_test_days, available_test_days, target_test_days)
    test = history.tail(test_size).copy()
    train = history.iloc[:-test_size].copy()
    return train, test


def _evaluate_backtest_window(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Evaluate one train/test backtest window."""
    baseline_pred = baseline_forecast(train_df=train, target_df=test)
    enhanced_pred = calibrated_tailored_forecast(train_df=train, target_df=test)

    out = test[["stay_date", "rooms_sold", "room_revenue"]].copy()
    out = out.rename(columns={"rooms_sold": "actual_rooms_sold", "room_revenue": "actual_revenue"})
    out["baseline_rooms_sold"] = baseline_pred["forecast_rooms_sold"].to_numpy()
    out["enhanced_rooms_sold"] = enhanced_pred["forecast_rooms_sold"].to_numpy()
    out["baseline_revenue"] = baseline_pred["forecast_revenue"].to_numpy()
    out["enhanced_revenue"] = enhanced_pred["forecast_revenue"].to_numpy()
    out["month"] = test["stay_date"].dt.month_name().to_numpy()
    if "day_type" in test.columns:
        out["day_type"] = test["day_type"].to_numpy()
    elif "is_weekend" in test.columns:
        out["day_type"] = np.where(pd.to_numeric(test["is_weekend"], errors="coerce").fillna(0) == 1, "Weekend", "Weekday")
    else:
        out["day_type"] = np.where(test["stay_date"].dt.dayofweek.isin([4, 5]), "Weekend", "Weekday")
    if "event_flag" in test.columns:
        out["event_flag"] = test["event_flag"].to_numpy()
        out["event_period"] = np.where(out["event_flag"] > 0, "Event period", "Non-event period")
    if "property_type" in test.columns:
        out["property_type"] = test["property_type"].to_numpy()
    return out


def evaluate_backtest(
    model_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    min_test_days: int = 21,
    max_test_days: int = 30,
    min_train_days: int = 10,
    rolling_window_days: int = 7,
) -> pd.DataFrame:
    """Run baseline and enhanced predictions on historical holdout data."""
    history = model_df[model_df["stay_date"] <= as_of_date].copy().sort_values("stay_date")
    empty_columns = [
        "stay_date",
        "actual_rooms_sold",
        "baseline_rooms_sold",
        "enhanced_rooms_sold",
        "actual_revenue",
        "baseline_revenue",
        "enhanced_revenue",
        "month",
        "day_type",
    ]

    if len(history) == 0:
        return pd.DataFrame(
            columns=empty_columns
        )

    initial_train_days = max(min_train_days, min_test_days)
    if len(history) > initial_train_days + max(1, rolling_window_days):
        windows = []
        start = initial_train_days
        while start < len(history):
            end = min(start + rolling_window_days, len(history))
            train = history.iloc[:start].copy()
            test = history.iloc[start:end].copy()
            if len(train) >= min_train_days and len(test) > 0:
                windows.append(_evaluate_backtest_window(train, test))
            start = end
        if windows:
            return pd.concat(windows, ignore_index=True).sort_values("stay_date").reset_index(drop=True)

    train, test = build_backtest_sets(
        model_df=history,
        as_of_date=as_of_date,
        min_test_days=min_test_days,
        max_test_days=max_test_days,
        min_train_days=min_train_days,
    )
    if len(test) == 0:
        return pd.DataFrame(columns=empty_columns)
    return _evaluate_backtest_window(train, test)


def build_future_forecast(
    historical_df: pd.DataFrame,
    future_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    blend_on_books_floor: bool = True,
) -> pd.DataFrame:
    """
    Forecast future demand using explainable day-of-week/month seasonality.

    Output columns include:
      stay_date, rooms_available, on_books, forecast_rooms_sold, forecast_occ
    """
    if len(future_df) == 0:
        return pd.DataFrame(
            columns=[
                "stay_date",
                "rooms_available",
                "on_books",
                "forecast_rooms_sold",
                "forecast_occ",
                "base_demand",
                "history_depth_days",
                "forecast_method",
            ]
        )

    hist = historical_df.copy()
    hist["stay_date"] = pd.to_datetime(hist["stay_date"], errors="coerce")
    hist = hist.dropna(subset=["stay_date"]).sort_values("stay_date")
    hist["dow"] = hist["stay_date"].dt.dayofweek
    hist["month"] = hist["stay_date"].dt.month

    fut = future_df.copy()
    fut["stay_date"] = pd.to_datetime(fut["stay_date"], errors="coerce")
    fut = fut.dropna(subset=["stay_date"]).sort_values("stay_date")
    fut["dow"] = fut["stay_date"].dt.dayofweek
    fut["month"] = fut["stay_date"].dt.month
    fut["on_books"] = pd.to_numeric(fut["rooms_sold"], errors="coerce").fillna(0.0)

    history_depth_days = int(hist["stay_date"].nunique()) if len(hist) else 0
    history_trust = float(np.clip(history_depth_days / 28.0, 0.0, 1.0))
    if history_depth_days == 0:
        forecast_method = "no_history_on_books_floor"
    elif history_depth_days < 14:
        forecast_method = "low_history_blended_fallback"
    elif history_depth_days < 28:
        forecast_method = "moderate_history_blended_seasonality"
    else:
        forecast_method = "strong_history_seasonality"

    dow_avg = hist.groupby("dow")["rooms_sold"].mean() if len(hist) else pd.Series(dtype=float)
    month_avg = hist.groupby("month")["rooms_sold"].mean() if len(hist) else pd.Series(dtype=float)
    global_avg = float(hist["rooms_sold"].mean()) if len(hist) else 0.0
    observed_occupancy = (
        hist["rooms_sold"] / hist["rooms_available"].replace(0, np.nan)
        if len(hist) and "rooms_available" in hist.columns
        else pd.Series(dtype=float)
    )
    global_occ = float(observed_occupancy.dropna().clip(lower=0.0, upper=1.0).median()) if observed_occupancy.notna().any() else np.nan

    base_from_dow = fut["dow"].map(dow_avg)
    base_from_month = fut["month"].map(month_avg)
    seasonal_base = pd.Series(
        np.nanmean(
            np.vstack([
                base_from_dow.fillna(global_avg).to_numpy(),
                base_from_month.fillna(global_avg).to_numpy(),
            ]),
            axis=0,
        ),
        index=fut.index,
    ).fillna(global_avg).clip(lower=0.0)

    if pd.notna(global_occ):
        inventory_fallback = pd.to_numeric(fut["rooms_available"], errors="coerce").fillna(0.0).clip(lower=0.0) * global_occ
    else:
        inventory_fallback = pd.Series(global_avg, index=fut.index, dtype=float)

    fallback_base = pd.concat(
        [
            pd.to_numeric(fut["on_books"], errors="coerce").fillna(0.0),
            inventory_fallback.fillna(global_avg),
            pd.Series(global_avg, index=fut.index, dtype=float),
        ],
        axis=1,
    ).max(axis=1)

    fut["base_demand"] = (
        seasonal_base * history_trust
        + fallback_base * (1.0 - history_trust)
    ).fillna(fallback_base).clip(lower=0.0)

    if history_depth_days < 14:
        # Sparse history can overreact to one unusual day. Keep the model close
        # to the on-books/global fallback until more property history exists.
        lower_bound = pd.to_numeric(fut["on_books"], errors="coerce").fillna(0.0)
        upper_bound = pd.concat(
            [
                fallback_base * 1.20,
                lower_bound,
            ],
            axis=1,
        ).max(axis=1)
        fut["base_demand"] = fut["base_demand"].clip(lower=lower_bound, upper=upper_bound)

    fut["history_depth_days"] = history_depth_days
    fut["forecast_method"] = forecast_method

    if events_df is not None and len(events_df) > 0 and "stay_date" in events_df.columns:
        events = events_df[["stay_date", "impact_level"]].copy()
        events["stay_date"] = pd.to_datetime(events["stay_date"], errors="coerce")
        events = events.dropna(subset=["stay_date"])
        events["event_factor"] = events.get("impact_level", pd.Series(index=events.index, dtype=object)).astype(str).str.lower().map(
            {"low": 1.02, "medium": 1.05, "high": 1.09}
        ).fillna(1.0)
        fut = fut.merge(events[["stay_date", "event_factor"]], on="stay_date", how="left")
        fut["event_factor"] = fut["event_factor"].fillna(1.0)
    else:
        fut["event_factor"] = 1.0

    model_forecast = fut["base_demand"] * fut["event_factor"]
    if blend_on_books_floor:
        fut["forecast_rooms_sold"] = np.maximum(fut["on_books"], model_forecast)
    else:
        fut["forecast_rooms_sold"] = model_forecast

    fut["forecast_rooms_sold"] = np.minimum(fut["forecast_rooms_sold"], fut["rooms_available"].clip(lower=0.0))
    fut["forecast_occ"] = np.where(
        fut["rooms_available"] > 0,
        fut["forecast_rooms_sold"] / fut["rooms_available"],
        0.0,
    )

    return fut[
        [
            "stay_date",
            "rooms_available",
            "on_books",
            "base_demand",
            "forecast_rooms_sold",
            "forecast_occ",
            "history_depth_days",
            "forecast_method",
        ]
    ].copy()
