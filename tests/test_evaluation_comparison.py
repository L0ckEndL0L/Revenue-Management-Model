from __future__ import annotations

import math

import pandas as pd

from src.evaluation import (
    _calibrated_rate_recommendation,
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


def test_build_subgroup_backtest_metrics_sorts_months_chronologically() -> None:
    backtest_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-12-01", "2026-02-01", "2026-01-01"]),
            "actual_rooms_sold": [30, 20, 10],
            "baseline_rooms_sold": [29, 19, 9],
            "enhanced_rooms_sold": [30, 20, 10],
            "property_type": ["Full Service"] * 3,
            "event_flag": [0, 0, 0],
            "day_type": ["Weekday"] * 3,
        }
    )

    out = build_subgroup_backtest_metrics(backtest_df)

    assert list(out.drop_duplicates("month")["month"]) == ["January", "February", "December"]
    for month in ["January", "February", "December"]:
        assert list(out[out["month"] == month]["model"]) == ["Baseline Model", "Tailored Model"]


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


def test_rate_challenger_selects_tailored_only_with_repeatable_prior_improvement() -> None:
    improving_history = [
        {
            "actual_adr": 100.0,
            "baseline_recommendation": 110.0,
            "raw_rateanchor_recommendation": 101.0,
        }
        for _ in range(42)
    ]
    final_rate, selected, calibration_rows, score, confidence = _calibrated_rate_recommendation(
        improving_history, baseline_rate=110.0, raw_tailored_rate=101.0
    )

    assert selected == "raw_tailored"
    assert calibration_rows == 42
    assert score < 0.97
    assert confidence > 0
    assert abs(final_rate - 100.0) < abs(110.0 - 100.0)

    worsening_history = [
        {
            "actual_adr": 100.0,
            "baseline_recommendation": 101.0,
            "raw_rateanchor_recommendation": 110.0,
        }
        for _ in range(42)
    ]
    final_rate, selected, _, _, confidence = _calibrated_rate_recommendation(
        worsening_history, baseline_rate=101.0, raw_tailored_rate=110.0
    )

    assert selected == "baseline"
    assert final_rate == 101.0
    assert confidence == 0.0


def test_rate_subgroup_backtest_metrics_split_property_event_month_and_day_type() -> None:
    rate_backtest_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-02-02", "2026-01-03", "2026-02-06", "2026-01-05"]),
            "actual_adr": [100.0, 120.0, 140.0, None],
            "baseline_recommendation": [105.0, 110.0, 155.0, 130.0],
            "rateanchor_recommendation": [102.0, 121.0, 136.0, 128.0],
            "property_type": ["Resort", "Resort", "Boutique", "Boutique"],
            "event_period": ["Event period", "Non-event period", "Event period", "Non-event period"],
        }
    )

    out = build_rate_subgroup_backtest_metrics(rate_backtest_df)

    assert {"property_type", "event_period", "month", "day_type", "model", "mae", "rmse", "backtest_rows"}.issubset(out.columns)
    assert set(out["property_type"]) == {"Resort", "Boutique"}
    assert set(out["event_period"]) == {"Event period", "Non-event period"}
    assert set(out["month"]) == {"January", "February"}
    assert set(out["day_type"]) == {"Weekday", "Weekend"}
    assert set(out["model"]) == {"Baseline Model", "RateAnchor Tailored Model"}
    assert list(out.drop_duplicates("month")["month"]) == ["January", "February"]


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
        min_history_days=1,
        tailored_settings={
            "global_median_rate_fallback": 125.0,
            "minimum_acceptable_rate": 80.0,
            "maximum_recommended_rate": 250.0,
        },
    )
    metrics = build_rate_backtest_metrics(out)

    assert len(out) == 2
    assert out["actual_adr"].notna().sum() == 1
    assert metrics["mae"].notna().all()


def test_rate_backtest_does_not_use_target_day_actual_adr_as_input() -> None:
    dates = pd.date_range("2026-01-01", periods=35, freq="D")
    historical_df = pd.DataFrame(
        {
            "stay_date": dates,
            "rooms_available": [100] * 35,
            "rooms_sold": [70] * 35,
            "room_revenue": [7000.0] * 34 + [21000.0],
            "adr": [100.0] * 34 + [300.0],
            "occupancy": [0.70] * 35,
        }
    )

    out = build_rate_backtest_frame(
        historical_df,
        tailored_settings={
            "property_type": "Full Service",
            "segment_focus": "Balanced",
            "minimum_acceptable_rate": 40.0,
            "maximum_recommended_rate": 400.0,
        },
    )

    final_row = out.iloc[-1]
    assert len(out) == 7
    assert final_row["actual_adr"] == 300.0
    assert final_row["rate_input_adr"] == 100.0
    assert final_row["baseline_recommendation"] != final_row["actual_adr"]
    assert final_row["history_rows"] == 34
    assert {
        "raw_rateanchor_recommendation",
        "selected_rate_model",
        "rate_calibration_rows",
        "rate_calibration_score",
        "rate_confidence_weight",
    }.issubset(out.columns)
    assert set(out["selected_rate_model"]).issubset(
        {
            "baseline_warmup",
            "baseline",
            "raw_tailored",
            "baseline_tailored_blend_0.25",
            "baseline_tailored_blend_0.50",
            "baseline_tailored_blend_0.75",
        }
    )
