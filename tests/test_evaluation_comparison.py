from __future__ import annotations

import math

import pandas as pd

from src.evaluation import (
    build_rate_backtest_frame,
    build_rate_backtest_metrics,
    build_rate_subgroup_backtest_metrics,
    build_model_comparison_metrics,
    build_subgroup_backtest_metrics,
    detect_prediction_identity_warning,
)


def test_build_model_comparison_metrics_reports_mae_and_rmse() -> None:
    backtest_df = pd.DataFrame(
        {
            "actual_rooms_sold": [10, 20, 30],
            "baseline_rooms_sold": [12, 18, 33],
            "enhanced_rooms_sold": [11, 21, 29],
        }
    )

    out = build_model_comparison_metrics(backtest_df)

    assert list(out["model"]) == ["Baseline Model", "Tailored Model"]
    baseline = out[out["model"] == "Baseline Model"].iloc[0]
    tailored = out[out["model"] == "Tailored Model"].iloc[0]
    assert baseline["mae"] == (2 + 2 + 3) / 3
    assert math.isclose(baseline["rmse"], math.sqrt((4 + 4 + 9) / 3))
    assert tailored["mae"] == 1.0
    assert tailored["rmse"] == 1.0
    assert baseline["backtest_rows"] == 3


def test_build_subgroup_backtest_metrics_splits_property_event_month_and_day_type() -> None:
    backtest_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-03", "2026-07-06", "2026-08-07", "2026-08-10"]),
            "actual_rooms_sold": [10, 20, 30, 40],
            "baseline_rooms_sold": [12, 18, 33, 39],
            "enhanced_rooms_sold": [11, 19, 31, 41],
            "property_type": ["Resort", "Resort", "Boutique", "Boutique"],
            "event_flag": [1, 0, 1, 0],
            "day_type": ["Weekend", "Weekday", "Weekend", "Weekday"],
        }
    )

    out = build_subgroup_backtest_metrics(backtest_df)

    assert {"property_type", "event_period", "month", "day_type", "model", "mae", "rmse", "backtest_rows"}.issubset(out.columns)
    assert set(out["property_type"]) == {"Resort", "Boutique"}
    assert set(out["event_period"]) == {"Event period", "Non-event period"}
    assert set(out["month"]) == {"July", "August"}
    assert set(out["day_type"]) == {"Weekend", "Weekday"}
    assert set(out["model"]) == {"Baseline Model", "Tailored Model"}
    assert len(out) == 8
    assert (out["backtest_rows"] == 1).all()


def test_detect_prediction_identity_warning_flags_suspicious_perfect_fit() -> None:
    actual = pd.Series([10, 12, 14, 16, 18, 20, 22])
    predicted = pd.Series([10, 12, 14, 16, 18, 20, 22])

    warning = detect_prediction_identity_warning(actual, predicted)

    assert "Potential leakage" in warning


def test_rate_backtest_metrics_compare_actual_adr_to_rate_recommendations() -> None:
    rate_backtest_df = pd.DataFrame(
        {
            "actual_adr": [100.0, 120.0, 140.0],
            "baseline_recommendation": [105.0, 110.0, 155.0],
            "rateanchor_recommendation": [102.0, 121.0, 136.0],
        }
    )

    out = build_rate_backtest_metrics(rate_backtest_df)

    baseline = out[out["model"] == "Baseline Model"].iloc[0]
    rateanchor = out[out["model"] == "RateAnchor Tailored Model"].iloc[0]
    assert baseline["mae"] == 10.0
    assert rateanchor["mae"] < baseline["mae"]
    assert rateanchor["mae_difference_vs_baseline"] < 0
    assert rateanchor["mae_improvement_vs_baseline"] > 0


def test_rate_subgroup_backtest_metrics_split_property_type_and_event_period() -> None:
    rate_backtest_df = pd.DataFrame(
        {
            "actual_adr": [100.0, 120.0, 140.0, None],
            "baseline_recommendation": [105.0, 110.0, 155.0, 130.0],
            "rateanchor_recommendation": [102.0, 121.0, 136.0, 128.0],
            "property_type": ["Resort", "Resort", "Boutique", "Boutique"],
            "event_period": ["Event period", "Non-event period", "Event period", "Non-event period"],
        }
    )

    out = build_rate_subgroup_backtest_metrics(rate_backtest_df)

    assert {"property_type", "event_period", "model", "mae", "rmse", "backtest_rows"}.issubset(out.columns)
    assert set(out["property_type"]) == {"Resort", "Boutique"}
    assert set(out["event_period"]) == {"Event period", "Non-event period"}
    assert set(out["model"]) == {"Baseline Model", "RateAnchor Tailored Model"}


def test_rate_backtest_frame_handles_missing_actual_rate_values() -> None:
    historical_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
            "rooms_available": [100, 100, 100],
            "rooms_sold": [80, 70, 0],
            "room_revenue": [9600.0, 9100.0, None],
            "current_rate": [120.0, 130.0, None],
            "occupancy": [0.80, 0.70, 0.20],
        }
    )

    out = build_rate_backtest_frame(
        historical_df,
        tailored_settings={
            "global_median_rate_fallback": 125.0,
            "minimum_acceptable_rate": 80.0,
            "maximum_recommended_rate": 250.0,
        },
    )
    metrics = build_rate_backtest_metrics(out)

    assert len(out) == 3
    assert out["actual_adr"].notna().sum() == 2
    assert metrics["mae"].notna().all()
