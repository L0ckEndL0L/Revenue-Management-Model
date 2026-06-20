from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tailored import (
    DATASET_DERIVED_SOURCE,
    DAILY_COMP_RATE_MODE,
    GLOBAL_FALLBACK_SOURCE,
    MANUAL_DAILY_SOURCE,
    MISSING_MEDIAN_SOURCE,
    MONTHLY_COMP_RATE_MODE,
    build_daily_median_rate_table,
    build_tailored_recommendations,
    build_tailored_summary,
    default_tailored_settings,
    infer_median_rate_from_dataset,
    infer_median_rate_from_comp_set,
    is_median_rate_stale,
    update_daily_median_rates,
    update_median_rate,
    validate_tailored_settings,
)


def _sample_future_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-03"]),
            "rooms_available": [100, 100, 100, 100],
            "rooms_sold": [92, 88, 45, 20],
            "current_rate": [145.0, 155.0, 130.0, None],
            "room_revenue": [13340.0, 13640.0, 5850.0, None],
            "forecast_occ": [0.92, 0.88, 0.45, 0.20],
            "pace_variance": [0.12, 0.08, -0.08, -0.15],
            "impact_level": ["high", "medium", "low", "low"],
            "event_pct": [0.03, 0.01, 0.0, 0.0],
        }
    )


def _sample_baseline_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
            "baseline_recommended_rate": [150.0, 125.0, 110.0],
            "baseline_status": ["OK", "OK", "OK"],
            "baseline_reason": ["baseline high occ", "baseline low occ", "baseline very low occ"],
        }
    )


def test_infer_median_rate_from_dataset_returns_date_level_medians() -> None:
    medians = infer_median_rate_from_dataset(_sample_future_df(), _sample_baseline_df())

    assert len(medians) == 2
    assert round(float(medians.loc[medians["stay_date"] == "2026-07-01", "suggested_dataset_median_rate"].iloc[0]), 2) == 150.0
    assert round(float(medians.loc[medians["stay_date"] == "2026-07-02", "suggested_dataset_median_rate"].iloc[0]), 2) == 130.0


def test_infer_median_rate_from_comp_set_returns_date_level_medians() -> None:
    comp_set_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-02"]),
            "competitor_name": ["Comp A", "Comp B", "Comp A"],
            "rate": [170.0, 190.0, 150.0],
        }
    )

    medians = infer_median_rate_from_comp_set(comp_set_df)

    assert round(float(medians.loc[medians["stay_date"] == "2026-07-01", "suggested_dataset_median_rate"].iloc[0]), 2) == 180.0
    assert round(float(medians.loc[medians["stay_date"] == "2026-07-02", "suggested_dataset_median_rate"].iloc[0]), 2) == 150.0


def test_comp_set_medians_take_priority_over_future_rate_in_daily_table() -> None:
    comp_set_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "competitor_name": ["Comp A", "Comp B"],
            "rate": [170.0, 190.0],
        }
    )

    table = build_daily_median_rate_table(
        _sample_future_df(),
        default_tailored_settings(),
        baseline_df=_sample_baseline_df(),
        comp_set_df=comp_set_df,
    )
    july_1 = table.loc[table["stay_date"] == "2026-07-01"].iloc[0]

    assert july_1["suggested_dataset_median_rate"] == 180.0
    assert july_1["median_rate_source"] == DATASET_DERIVED_SOURCE


def test_manual_daily_median_overrides_dataset_derived_by_date() -> None:
    settings = default_tailored_settings()
    settings.update(
        {
            "global_median_rate_fallback": 140.0,
            "median_rate_last_updated": datetime(2026, 7, 1, 9, 0, 0).isoformat(),
            "median_rate_update_frequency": "Every 2 hours",
            "maximum_recommended_rate": 250.0,
            "daily_median_rates": [
                {
                    "stay_date": "2026-07-01",
                    "manual_daily_median_rate": 165.0,
                    "last_median_update_timestamp": datetime(2026, 7, 1, 9, 5, 0).isoformat(),
                }
            ],
        }
    )

    out = build_tailored_recommendations(
        _sample_future_df(),
        _sample_baseline_df(),
        settings,
        reference_time=datetime(2026, 7, 1, 10, 0, 0),
    )

    july_1 = out.loc[out["stay_date"] == "2026-07-01"].iloc[0]
    july_2 = out.loc[out["stay_date"] == "2026-07-02"].iloc[0]

    assert july_1["median_rate_used"] == 165.0
    assert july_1["median_rate_source"] == MANUAL_DAILY_SOURCE
    assert july_2["median_rate_source"] == DATASET_DERIVED_SOURCE


def test_dataset_derived_daily_median_is_used_when_manual_is_blank() -> None:
    settings = default_tailored_settings()
    settings["global_median_rate_fallback"] = None
    settings["daily_median_rates"] = [
        {"stay_date": "2026-07-02", "manual_daily_median_rate": None, "last_median_update_timestamp": None}
    ]

    out = build_tailored_recommendations(
        _sample_future_df(),
        _sample_baseline_df(),
        settings,
        reference_time=datetime(2026, 7, 1, 10, 0, 0),
    )

    july_2 = out.loc[out["stay_date"] == "2026-07-02"].iloc[0]

    assert july_2["median_rate_used"] == 130.0
    assert july_2["median_rate_source"] == DATASET_DERIVED_SOURCE


def test_monthly_comp_rate_mode_uses_global_rate_for_each_date() -> None:
    settings = default_tailored_settings()
    settings.update(
        {
            "comp_rate_input_mode": MONTHLY_COMP_RATE_MODE,
            "global_median_rate_fallback": 140.0,
            "daily_median_rates": [
                {"stay_date": "2026-07-01", "manual_daily_median_rate": 180.0}
            ],
        }
    )

    table = build_daily_median_rate_table(_sample_future_df(), settings, baseline_df=_sample_baseline_df())

    assert set(table["median_rate_source"]) == {GLOBAL_FALLBACK_SOURCE}
    assert set(table["final_median_rate_used"]) == {140.0}


def test_daily_comp_rate_mode_can_override_monthly_rate_by_date() -> None:
    settings = default_tailored_settings()
    settings.update(
        {
            "comp_rate_input_mode": DAILY_COMP_RATE_MODE,
            "global_median_rate_fallback": 140.0,
            "daily_median_rates": [
                {"stay_date": "2026-07-01", "manual_daily_median_rate": 180.0}
            ],
        }
    )

    table = build_daily_median_rate_table(_sample_future_df(), settings, baseline_df=_sample_baseline_df())
    july_1 = table.loc[table["stay_date"] == "2026-07-01"].iloc[0]

    assert july_1["median_rate_source"] == MANUAL_DAILY_SOURCE
    assert july_1["final_median_rate_used"] == 180.0


def test_segment_focus_changes_tailored_recommendation_posture() -> None:
    base_settings = default_tailored_settings()
    base_settings.update(
        {
            "global_median_rate_fallback": 145.0,
            "maximum_recommended_rate": 300.0,
        }
    )

    balanced_settings = {**base_settings, "segment_focus": "Balanced"}
    premium_settings = {**base_settings, "segment_focus": "Premium"}
    group_settings = {**base_settings, "segment_focus": "Group"}

    balanced = build_tailored_recommendations(_sample_future_df(), _sample_baseline_df(), balanced_settings)
    premium = build_tailored_recommendations(_sample_future_df(), _sample_baseline_df(), premium_settings)
    group = build_tailored_recommendations(_sample_future_df(), _sample_baseline_df(), group_settings)

    assert premium["tailored_recommendation"].mean() > balanced["tailored_recommendation"].mean()
    assert group["tailored_recommendation"].mean() < balanced["tailored_recommendation"].mean()
    assert "premium posture" in premium.loc[0, "reasoning_notes"]
    assert "group-focused posture" in group.loc[0, "reasoning_notes"]


def test_revenue_segment_prices_above_occupancy_segment() -> None:
    base_settings = default_tailored_settings()
    base_settings.update(
        {
            "global_median_rate_fallback": 145.0,
            "maximum_recommended_rate": 300.0,
        }
    )

    revenue = build_tailored_recommendations(
        _sample_future_df(),
        _sample_baseline_df(),
        {**base_settings, "segment_focus": "Revenue"},
    )
    occupancy = build_tailored_recommendations(
        _sample_future_df(),
        _sample_baseline_df(),
        {**base_settings, "segment_focus": "Occupancy"},
    )

    assert revenue["tailored_recommendation"].mean() > occupancy["tailored_recommendation"].mean()


def test_property_type_changes_tailored_recommendation_posture() -> None:
    base_settings = default_tailored_settings()
    base_settings.update(
        {
            "global_median_rate_fallback": 145.0,
            "maximum_recommended_rate": 300.0,
            "segment_focus": "Balanced",
        }
    )

    full_service = build_tailored_recommendations(
        _sample_future_df(),
        _sample_baseline_df(),
        {**base_settings, "property_type": "Full Service"},
    )
    luxury = build_tailored_recommendations(
        _sample_future_df(),
        _sample_baseline_df(),
        {**base_settings, "property_type": "Luxury"},
    )
    economy = build_tailored_recommendations(
        _sample_future_df(),
        _sample_baseline_df(),
        {**base_settings, "property_type": "Economy"},
    )

    assert luxury["tailored_recommendation"].mean() > full_service["tailored_recommendation"].mean()
    assert economy["tailored_recommendation"].mean() < full_service["tailored_recommendation"].mean()
    assert "luxury posture" in luxury.loc[0, "reasoning_notes"]
    assert "economy posture" in economy.loc[0, "reasoning_notes"]


def test_resort_property_reacts_more_to_event_demand_than_extended_stay() -> None:
    future_df = _sample_future_df().copy()
    future_df["impact_level"] = ["high", "high", "high", "high"]
    future_df["event_pct"] = [0.08, 0.08, 0.08, 0.08]

    base_settings = default_tailored_settings()
    base_settings.update(
        {
            "global_median_rate_fallback": 145.0,
            "maximum_recommended_rate": 300.0,
            "segment_focus": "Balanced",
        }
    )

    resort = build_tailored_recommendations(
        future_df,
        _sample_baseline_df(),
        {**base_settings, "property_type": "Resort"},
    )
    extended_stay = build_tailored_recommendations(
        future_df,
        _sample_baseline_df(),
        {**base_settings, "property_type": "Extended Stay"},
    )

    assert resort["tailored_recommendation"].mean() > extended_stay["tailored_recommendation"].mean()
    assert "resort posture" in resort.loc[0, "reasoning_notes"]
    assert "extended-stay posture" in extended_stay.loc[0, "reasoning_notes"]


def test_global_fallback_is_used_only_when_no_daily_median_exists() -> None:
    settings = default_tailored_settings()
    settings.update(
        {
            "global_median_rate_fallback": 120.0,
            "daily_median_rates": [
                {"stay_date": "2026-07-01", "manual_daily_median_rate": 165.0, "last_median_update_timestamp": datetime(2026, 7, 1, 9, 5, 0).isoformat()}
            ],
        }
    )

    out = build_tailored_recommendations(_sample_future_df(), _sample_baseline_df(), settings)

    july_3 = out.loc[out["stay_date"] == "2026-07-03"].iloc[0]
    assert july_3["median_rate_used"] == 120.0
    assert july_3["median_rate_source"] == GLOBAL_FALLBACK_SOURCE


def test_missing_median_produces_warning_but_does_not_crash() -> None:
    future_df = _sample_future_df().copy()
    future_df.loc[future_df["stay_date"] == pd.Timestamp("2026-07-03"), ["current_rate", "room_revenue"]] = None

    settings = default_tailored_settings()
    out = build_tailored_recommendations(future_df, _sample_baseline_df(), settings)

    july_3 = out.loc[out["stay_date"] == "2026-07-03"].iloc[0]
    assert july_3["median_rate_source"] == MISSING_MEDIAN_SOURCE
    assert str(july_3["model_status"]).startswith("WARNING")


def test_updating_one_dates_manual_median_changes_that_dates_recommendation() -> None:
    base_settings = default_tailored_settings()
    base_settings = update_median_rate(base_settings, 120.0, updated_at=datetime(2026, 7, 1, 8, 0, 0))
    base_settings = update_daily_median_rates(
        base_settings,
        [{"stay_date": "2026-07-01", "manual_daily_median_rate": 150.0}],
        updated_at=datetime(2026, 7, 1, 8, 5, 0),
    )

    updated_settings = update_daily_median_rates(
        base_settings,
        [{"stay_date": "2026-07-01", "manual_daily_median_rate": 180.0}],
        updated_at=datetime(2026, 7, 1, 11, 0, 0),
    )

    early = build_tailored_recommendations(_sample_future_df(), _sample_baseline_df(), base_settings)
    late = build_tailored_recommendations(_sample_future_df(), _sample_baseline_df(), updated_settings)

    assert early.loc[0, "tailored_recommendation"] != late.loc[0, "tailored_recommendation"]
    assert late.loc[0, "last_median_update_timestamp"] == datetime(2026, 7, 1, 11, 0, 0).isoformat()


def test_validate_tailored_settings_rejects_invalid_inputs_cleanly() -> None:
    settings = {
        "global_median_rate_fallback": -10,
        "minimum_acceptable_rate": 200,
        "maximum_recommended_rate": 150,
        "baseline_occupancy_sensitivity": 3.0,
        "median_rate_update_frequency": "Every 7 minutes",
        "daily_median_rates": [
            {"stay_date": "2026-07-02", "manual_daily_median_rate": -5},
        ],
    }

    _, errors = validate_tailored_settings(settings)

    assert "global median fallback must be greater than 0" in errors
    assert "maximum recommended rate must be greater than minimum acceptable rate" in errors
    assert "baseline occupancy sensitivity must be between 0.0 and 2.0" in errors
    assert "update frequency must be one of: Every 30 minutes, Every hour, Every 2 hours, Daily, Manual only" in errors
    assert "manual daily median rate for 2026-07-02 must be greater than 0" in errors


def test_thirty_minute_comp_rate_review_cadence_is_supported() -> None:
    settings = update_median_rate(default_tailored_settings(), 150.0, updated_at=datetime(2026, 7, 1, 8, 0, 0))
    settings["median_rate_update_frequency"] = "Every 30 minutes"

    assert is_median_rate_stale(settings, reference_time=datetime(2026, 7, 1, 8, 31, 0)) is True
    assert is_median_rate_stale(settings, reference_time=datetime(2026, 7, 1, 8, 20, 0)) is False


def test_is_median_rate_stale_honors_review_cadence() -> None:
    settings = update_median_rate(default_tailored_settings(), 150.0, updated_at=datetime(2026, 7, 1, 8, 0, 0))
    settings["median_rate_update_frequency"] = "Every hour"

    assert is_median_rate_stale(settings, reference_time=datetime(2026, 7, 1, 9, 30, 0)) is True
    assert is_median_rate_stale(settings, reference_time=datetime(2026, 7, 1, 8, 45, 0)) is False


def test_tailored_summary_contains_expected_fields() -> None:
    settings = update_median_rate(default_tailored_settings(), 140.0, updated_at=datetime(2026, 7, 1, 9, 0, 0))
    settings = update_daily_median_rates(
        settings,
        [{"stay_date": "2026-07-01", "manual_daily_median_rate": 165.0}],
        updated_at=datetime(2026, 7, 1, 9, 5, 0),
    )
    out = build_tailored_recommendations(_sample_future_df(), _sample_baseline_df(), settings)
    summary = build_tailored_summary(out, settings)

    assert {"global_median_rate_fallback", "avg_tailored_recommendation", "warning_rows", "manual_daily_median_dates", "dataset_derived_daily_median_dates", "global_fallback_median_dates", "missing_median_dates"}.issubset(summary.columns)


def test_build_daily_median_rate_table_contains_date_level_sources() -> None:
    settings = update_median_rate(default_tailored_settings(), 118.0, updated_at=datetime(2026, 7, 1, 8, 0, 0))
    settings = update_daily_median_rates(
        settings,
        [{"stay_date": "2026-07-01", "manual_daily_median_rate": 164.0}],
        updated_at=datetime(2026, 7, 1, 8, 30, 0),
    )

    table = build_daily_median_rate_table(_sample_future_df(), settings, baseline_df=_sample_baseline_df())

    july_1 = table.loc[table["stay_date"] == "2026-07-01"].iloc[0]
    july_2 = table.loc[table["stay_date"] == "2026-07-02"].iloc[0]
    july_3 = table.loc[table["stay_date"] == "2026-07-03"].iloc[0]

    assert july_1["median_rate_source"] == MANUAL_DAILY_SOURCE
    assert july_2["median_rate_source"] == DATASET_DERIVED_SOURCE
    assert july_3["median_rate_source"] == GLOBAL_FALLBACK_SOURCE
