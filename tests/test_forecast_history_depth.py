from __future__ import annotations

import pandas as pd

from src.forecast import build_future_forecast


def test_low_history_forecast_stays_close_to_on_books_fallback() -> None:
    historical_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]),
            "rooms_available": [100, 100, 100],
            "rooms_sold": [95, 10, 12],
            "room_revenue": [14250, 1200, 1440],
            "adr": [150, 120, 120],
            "occupancy": [0.95, 0.10, 0.12],
        }
    )
    future_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "rooms_available": [100, 100],
            "rooms_sold": [60, 65],
        }
    )

    out = build_future_forecast(historical_df, future_df)

    assert set(out["forecast_method"]) == {"low_history_blended_fallback"}
    assert set(out["history_depth_days"]) == {3}
    assert (out["forecast_rooms_sold"] >= out["on_books"]).all()
    assert (out["base_demand"] <= out["on_books"] * 1.20).all()


def test_strong_history_forecast_uses_day_of_week_depth() -> None:
    historical_dates = pd.date_range("2026-06-01", periods=56, freq="D")
    historical_df = pd.DataFrame(
        {
            "stay_date": historical_dates,
            "rooms_available": [100] * len(historical_dates),
            "rooms_sold": [90 if date.dayofweek == 4 else 45 for date in historical_dates],
            "room_revenue": [13500 if date.dayofweek == 4 else 5400 for date in historical_dates],
            "adr": [150 if date.dayofweek == 4 else 120 for date in historical_dates],
            "occupancy": [0.90 if date.dayofweek == 4 else 0.45 for date in historical_dates],
        }
    )
    future_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-08-07", "2026-08-08"]),
            "rooms_available": [100, 100],
            "rooms_sold": [20, 20],
        }
    )

    out = build_future_forecast(historical_df, future_df)
    friday = out.loc[out["stay_date"] == pd.Timestamp("2026-08-07")].iloc[0]
    saturday = out.loc[out["stay_date"] == pd.Timestamp("2026-08-08")].iloc[0]

    assert set(out["forecast_method"]) == {"strong_history_seasonality"}
    assert set(out["history_depth_days"]) == {56}
    assert friday["base_demand"] > saturday["base_demand"]
    assert friday["forecast_rooms_sold"] > saturday["forecast_rooms_sold"]
