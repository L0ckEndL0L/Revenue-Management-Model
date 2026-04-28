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
        stly["calendar_key"] = stly["stay_date"].dt.strftime("%m-%d")

        df["calendar_key"] = df["stay_date"].dt.strftime("%m-%d")
        df = df.merge(stly[["calendar_key", "stly_occupancy"]], on="calendar_key", how="left")
        df = df.drop(columns=["calendar_key"])
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


def build_backtest_sets(
    model_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    min_test_days: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological train/test split for forecast evaluation."""
    history = model_df[model_df["stay_date"] <= as_of_date].copy().sort_values("stay_date")

    if len(history) < (min_test_days + 10):
        return history.iloc[0:0].copy(), history.copy()

    test_size = max(min_test_days, int(round(len(history) * 0.2)))
    test = history.tail(test_size).copy()
    train = history.iloc[:-test_size].copy()
    return train, test


def evaluate_backtest(
    model_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Run baseline and enhanced predictions on holdout data for evaluation."""
    train, test = build_backtest_sets(model_df=model_df, as_of_date=as_of_date)
    if len(test) == 0:
        return pd.DataFrame(
            columns=[
                "stay_date",
                "actual_rooms_sold",
                "baseline_rooms_sold",
                "enhanced_rooms_sold",
                "actual_revenue",
                "baseline_revenue",
                "enhanced_revenue",
            ]
        )

    baseline_pred = baseline_forecast(train_df=train, target_df=test)
    enhanced_pred = enhanced_forecast(train_df=train, target_df=test)

    out = test[["stay_date", "rooms_sold", "room_revenue"]].copy()
    out = out.rename(columns={"rooms_sold": "actual_rooms_sold", "room_revenue": "actual_revenue"})
    out["baseline_rooms_sold"] = baseline_pred["forecast_rooms_sold"].to_numpy()
    out["enhanced_rooms_sold"] = enhanced_pred["forecast_rooms_sold"].to_numpy()
    out["baseline_revenue"] = baseline_pred["forecast_revenue"].to_numpy()
    out["enhanced_revenue"] = enhanced_pred["forecast_revenue"].to_numpy()
    return out


def forecast_remaining_month(
    model_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast remaining dates for baseline and enhanced models."""
    train_df = model_df[model_df["stay_date"] <= as_of_date].copy()
    target_df = model_df[model_df["stay_date"] > as_of_date].copy()

    baseline_future = baseline_forecast(train_df=train_df, target_df=target_df)
    enhanced_future = enhanced_forecast(train_df=train_df, target_df=target_df)
    return baseline_future, enhanced_future


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

    dow_avg = hist.groupby("dow")["rooms_sold"].mean() if len(hist) else pd.Series(dtype=float)
    month_avg = hist.groupby("month")["rooms_sold"].mean() if len(hist) else pd.Series(dtype=float)
    global_avg = float(hist["rooms_sold"].mean()) if len(hist) else 0.0

    base_from_dow = fut["dow"].map(dow_avg)
    base_from_month = fut["month"].map(month_avg)
    fut["base_demand"] = np.nanmean(
        np.vstack([
            base_from_dow.fillna(global_avg).to_numpy(),
            base_from_month.fillna(global_avg).to_numpy(),
        ]),
        axis=0,
    )
    fut["base_demand"] = pd.Series(fut["base_demand"], index=fut.index).fillna(global_avg).clip(lower=0.0)

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
        ]
    ].copy()
