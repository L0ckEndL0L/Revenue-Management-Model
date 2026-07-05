from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.intraday import process_intraday_updates, validate_intraday_updates
from src.tailored import default_tailored_settings


def _future_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "rooms_available": [100, 100],
            "rooms_sold": [92, 55],
            "current_rate": [150.0, 130.0],
            "room_revenue": [13800.0, 7150.0],
            "forecast_occ": [0.92, 0.55],
            "pace_variance": [0.10, -0.02],
            "event_pct": [0.03, 0.0],
            "impact_level": ["high", "low"],
        }
    )


def _baseline_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "baseline_recommended_rate": [155.0, 125.0],
            "baseline_status": ["OK", "OK"],
            "baseline_reason": ["high occ", "low occ"],
        }
    )


def test_intraday_updates_are_sorted_and_track_recommendation_changes() -> None:
    settings = default_tailored_settings()
    settings.update(
        {
            "global_median_rate_fallback": 140.0,
            "minimum_acceptable_rate": 80.0,
            "maximum_recommended_rate": 260.0,
        }
    )
    updates = pd.DataFrame(
        {
            "stay_date": ["2026-07-01", "2026-07-01"],
            "update_timestamp": [datetime(2026, 7, 1, 14, 0), datetime(2026, 7, 1, 9, 0)],
            "manual_daily_median_rate": [190.0, 170.0],
            "reason_summary": ["Afternoon shop", "Morning shop"],
        }
    )

    out, errors = process_intraday_updates(_future_df(), _baseline_df(), settings, updates)

    assert errors == []
    assert list(out["reason_summary"]) == ["Morning shop", "Afternoon shop"]
    assert list(out["updated_median_rate"]) == [170.0, 190.0]
    assert {"old_rate", "new_rate", "absolute_change", "percent_change"}.issubset(out.columns)
    assert out.iloc[1]["new_rate"] != out.iloc[0]["new_rate"]


def test_validate_intraday_updates_reports_missing_rate_column() -> None:
    validation = validate_intraday_updates(
        pd.DataFrame(
            {
                "stay_date": ["2026-07-01"],
                "update_timestamp": ["2026-07-01 09:00"],
            }
        )
    )

    assert validation["is_valid"] is False
    assert "manual_daily_median_rate" in validation["errors"][0]
