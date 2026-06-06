from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import (
    BASELINE_MODEL_TYPE,
    BaselinePricingConfig,
    generate_baseline_pricing_recommendations,
    validate_baseline_inputs,
)


def test_baseline_high_low_moderate_hand_checked_examples() -> None:
    config = BaselinePricingConfig(
        high_occupancy_threshold=0.85,
        low_occupancy_threshold=0.55,
        high_occupancy_increase_pct=0.05,
        low_occupancy_decrease_pct=-0.05,
        moderate_occupancy_change_pct=0.00,
        rate_floor=50.0,
        rate_ceiling=500.0,
    )

    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]),
            "rooms_available": [100, 100, 100],
            "rooms_sold": [90, 40, 70],
            "room_revenue": [11700.0, 4400.0, 8400.0],
        }
    )

    out = generate_baseline_pricing_recommendations(df, historical_df=None, config=config)

    high = out.iloc[0]
    low = out.iloc[1]
    moderate = out.iloc[2]

    # Current ADRs derived from revenue/sold: 130, 110, 120
    assert high["current_ADR"] == 130.0
    assert low["current_ADR"] == 110.0
    assert moderate["current_ADR"] == 120.0

    # Occupancy-driven adjustments
    assert round(high["baseline_adjustment_percent"], 2) == 5.00
    assert round(low["baseline_adjustment_percent"], 2) == -5.00
    assert round(moderate["baseline_adjustment_percent"], 2) == 0.00

    # Recommended rates
    assert round(high["baseline_recommended_rate"], 2) == 136.50
    assert round(low["baseline_recommended_rate"], 2) == 104.50
    assert round(moderate["baseline_recommended_rate"], 2) == 120.00

    assert high["baseline_status"] == "OK"
    assert low["baseline_status"] == "OK"
    assert moderate["baseline_status"] == "OK"


def test_baseline_missing_incomplete_data_safe_handling() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "rooms_available": [100, 100],
            "rooms_sold": [80, 0],
            "room_revenue": [9600.0, None],
        }
    )

    out = generate_baseline_pricing_recommendations(df)

    ok_row = out.iloc[0]
    unavailable_row = out.iloc[1]

    assert ok_row["baseline_status"] == "OK"
    assert unavailable_row["baseline_status"] == "UNAVAILABLE"
    assert pd.isna(unavailable_row["current_ADR"])
    assert pd.isna(unavailable_row["baseline_recommended_rate"])
    assert pd.isna(unavailable_row["baseline_adjustment_amount"])
    assert pd.isna(unavailable_row["baseline_adjustment_percent"])
    assert "unavailable" in unavailable_row["baseline_reason"].lower()


def test_baseline_output_required_columns_present() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-06-01"]),
            "rooms_available": [100],
            "rooms_sold": [90],
            "room_revenue": [11700.0],
        }
    )

    out = generate_baseline_pricing_recommendations(df)

    required_cols = {
        "stay_date",
        "day_of_week",
        "current_occupancy",
        "current_ADR",
        "baseline_recommended_rate",
        "baseline_adjustment_amount",
        "baseline_adjustment_percent",
        "baseline_reason",
        "baseline_model_type",
    }
    assert required_cols.issubset(set(out.columns))
    assert out.loc[0, "baseline_model_type"] == BASELINE_MODEL_TYPE


def test_baseline_model_ignores_rms_only_features() -> None:
    base_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-06-01"]),
            "rooms_available": [100],
            "rooms_sold": [90],
            "room_revenue": [11700.0],
        }
    )
    rms_extra_df = base_df.copy()
    rms_extra_df["event_pct"] = [0.25]
    rms_extra_df["impact_level"] = ["high"]
    rms_extra_df["remaining_budget"] = [50000.0]
    rms_extra_df["required_adr_remaining"] = [180.0]

    base_out = generate_baseline_pricing_recommendations(base_df)
    extra_out = generate_baseline_pricing_recommendations(rms_extra_df)

    # Baseline output should not expose RMS-only features.
    assert "event_pct" not in extra_out.columns
    assert "remaining_budget" not in extra_out.columns
    assert "required_adr_remaining" not in extra_out.columns

    # Recommendations should be unchanged by RMS-only fields.
    assert base_out.loc[0, "baseline_recommended_rate"] == extra_out.loc[0, "baseline_recommended_rate"]
    assert base_out.loc[0, "baseline_adjustment_percent"] == extra_out.loc[0, "baseline_adjustment_percent"]


def test_validate_baseline_inputs_requires_stay_date_and_kpi_inputs() -> None:
    missing_date_df = pd.DataFrame({"rooms_sold": [10], "room_revenue": [1000.0]})
    check1 = validate_baseline_inputs(missing_date_df)
    assert check1["is_valid"] is False
    assert "stay_date" in check1["missing_fields"]

    valid_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-06-01"]),
            "rooms_available": [100],
            "rooms_sold": [80],
            "room_revenue": [9600.0],
        }
    )
    check2 = validate_baseline_inputs(valid_df)
    assert check2["is_valid"] is True
