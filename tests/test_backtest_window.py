from __future__ import annotations

import pandas as pd

from src.forecast import (
    baseline_forecast,
    build_backtest_sets,
    calibrated_tailored_forecast,
    evaluate_backtest,
    prepare_forecast_frame,
)


def _model_frame(days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_date": pd.date_range("2026-01-01", periods=days, freq="D"),
            "rooms_available": [100] * days,
            "rooms_sold": [60] * days,
            "room_revenue": [9000.0] * days,
            "adr": [150.0] * days,
            "dow": pd.date_range("2026-01-01", periods=days, freq="D").dayofweek,
            "month": pd.date_range("2026-01-01", periods=days, freq="D").month,
            "is_weekend": [0] * days,
            "event_flag": [0] * days,
            "stly_occupancy": [0.60] * days,
        }
    )


def test_build_backtest_sets_targets_larger_holdout_window() -> None:
    train, test = build_backtest_sets(
        _model_frame(90),
        as_of_date=pd.Timestamp("2026-03-31"),
    )

    assert len(train) == 69
    assert len(test) == 21


def test_build_backtest_sets_caps_holdout_at_thirty_days() -> None:
    train, test = build_backtest_sets(
        _model_frame(180),
        as_of_date=pd.Timestamp("2026-06-29"),
    )

    assert len(train) == 150
    assert len(test) == 30


def test_build_backtest_sets_preserves_minimum_training_window() -> None:
    train, test = build_backtest_sets(
        _model_frame(28),
        as_of_date=pd.Timestamp("2026-01-28"),
    )

    assert len(train) == 10
    assert len(test) == 18


def test_prepare_forecast_frame_does_not_merge_same_year_stly_occupancy() -> None:
    daily_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-07-01", "2025-07-02"]),
            "rooms_available": [99, 99],
            "rooms_sold": [83, 70],
            "room_revenue": [10772.48, 11782.29],
            "occupancy": [83 / 99, 70 / 99],
            "adr": [129.79, 168.32],
        }
    )
    same_year_stly = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-07-01", "2025-07-02"]),
            "stly_occupancy": [83 / 99, 70 / 99],
        }
    )

    out = prepare_forecast_frame(daily_df, stly_df=same_year_stly)

    assert not out["stly_occupancy"].equals(out["occupancy"])


def test_prepare_forecast_frame_uses_prior_year_stly_occupancy() -> None:
    daily_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-07-01"]),
            "rooms_available": [99],
            "rooms_sold": [83],
            "room_revenue": [10772.48],
            "occupancy": [83 / 99],
            "adr": [129.79],
        }
    )
    prior_year_stly = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2024-07-01"]),
            "stly_occupancy": [0.72],
        }
    )

    out = prepare_forecast_frame(daily_df, stly_df=prior_year_stly)

    assert out.loc[0, "stly_occupancy"] == 0.72


def test_prepare_forecast_frame_prefers_prior_year_over_same_year_stly() -> None:
    daily_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-07-01"]),
            "rooms_available": [99],
            "rooms_sold": [83],
            "room_revenue": [10772.48],
            "occupancy": [83 / 99],
            "adr": [129.79],
        }
    )
    mixed_stly = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2024-07-01", "2025-07-01"]),
            "stly_occupancy": [0.72, 83 / 99],
        }
    )

    out = prepare_forecast_frame(daily_df, stly_df=mixed_stly)

    assert out.loc[0, "stly_occupancy"] == 0.72


def test_calibrated_tailored_forecast_falls_back_on_short_history() -> None:
    frame = _model_frame(7)
    target = _model_frame(3)

    baseline = baseline_forecast(frame, target)
    tailored = calibrated_tailored_forecast(frame, target)

    assert tailored["model_name"].eq("tailored_calibrated_baseline_short_history").all()
    assert tailored["forecast_rooms_sold"].equals(baseline["forecast_rooms_sold"])


def test_evaluate_backtest_uses_rolling_windows_for_month_specific_results() -> None:
    frame = _model_frame(92)
    frame["stay_date"] = pd.date_range("2026-07-01", periods=92, freq="D")
    frame["dow"] = frame["stay_date"].dt.dayofweek
    frame["month"] = frame["stay_date"].dt.month
    frame["is_weekend"] = frame["dow"].isin([4, 5]).astype(int)
    frame["day_type"] = frame["is_weekend"].map({1: "Weekend", 0: "Weekday"})

    out = evaluate_backtest(frame, as_of_date=pd.Timestamp("2026-09-30"))

    assert len(out) > 30
    assert {"July", "August", "September"}.issubset(set(out["month"]))
    assert {"Weekday", "Weekend"}.issubset(set(out["day_type"]))
