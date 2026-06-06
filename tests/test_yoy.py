from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.yoy import build_yoy_comparison, summarize_yoy


def test_build_yoy_comparison_hand_checked_complete_case() -> None:
    current_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-03-01"]),
            "rooms_available": [100],
            "rooms_sold": [80],
            "room_revenue": [9600.0],
        }
    )

    prior_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-03-01"]),
            "rooms_available": [100],
            "rooms_sold": [70],
            "room_revenue": [7700.0],
        }
    )

    out = build_yoy_comparison(current_df, prior_df)
    assert len(out) == 1

    row = out.iloc[0]
    assert row["yoy_status"] == "OK"
    assert row["yoy_alignment_method"] == "exact_prior_date"

    # Current KPIs
    assert row["current_occupancy_pct"] == 80.0
    assert row["current_adr"] == 120.0
    assert row["current_revpar"] == 96.0
    assert row["current_rooms_sold"] == 80
    assert row["current_room_revenue"] == 9600.0

    # Prior KPIs
    assert row["prior_year_occupancy_pct"] == 70.0
    assert row["prior_year_adr"] == 110.0
    assert row["prior_year_revpar"] == 77.0
    assert row["prior_year_rooms_sold"] == 70
    assert row["prior_year_room_revenue"] == 7700.0

    # Absolute variances
    assert row["occupancy_variance"] == 10.0
    assert row["adr_variance"] == 10.0
    assert row["revpar_variance"] == 19.0
    assert row["rooms_sold_variance"] == 10.0
    assert row["revenue_variance"] == 1900.0

    # Percentage variances
    assert abs(row["occupancy_variance_pct"] - 14.285714) < 1e-6
    assert abs(row["adr_variance_pct"] - 9.090909) < 1e-6
    assert abs(row["revpar_variance_pct"] - 24.675325) < 1e-6
    assert abs(row["rooms_sold_variance_pct"] - 14.285714) < 1e-6
    assert abs(row["revenue_variance_pct"] - 24.675325) < 1e-6


def test_build_yoy_comparison_missing_prior_data_case() -> None:
    current_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "rooms_available": [100, 100],
            "rooms_sold": [80, 90],
            "room_revenue": [9600.0, 11700.0],
        }
    )

    prior_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-03-01"]),
            "rooms_available": [100],
            "rooms_sold": [70],
            "room_revenue": [7700.0],
        }
    )

    out = build_yoy_comparison(current_df, prior_df)
    assert len(out) == 2

    matched = out[out["stay_date"] == pd.Timestamp("2026-03-01")].iloc[0]
    missing = out[out["stay_date"] == pd.Timestamp("2026-03-02")].iloc[0]

    assert matched["yoy_status"] == "OK"
    assert missing["yoy_status"] == "PRIOR_YEAR_UNAVAILABLE"
    assert missing["yoy_alignment_method"] == "no_prior_match"

    # Current-year KPIs still computed for missing prior-year rows.
    assert missing["current_occupancy_pct"] == 90.0
    assert missing["current_adr"] == 130.0
    assert missing["current_revpar"] == 117.0

    # Prior-year and variance fields are safely null when unavailable.
    assert pd.isna(missing["prior_year_occupancy_pct"])
    assert pd.isna(missing["prior_year_adr"])
    assert pd.isna(missing["prior_year_revpar"])
    assert pd.isna(missing["occupancy_variance_pct"])
    assert pd.isna(missing["adr_variance_pct"])
    assert pd.isna(missing["revpar_variance_pct"])
    assert pd.isna(missing["rooms_sold_variance_pct"])
    assert pd.isna(missing["revenue_variance_pct"])


def test_summarize_yoy_counts_missing_rows() -> None:
    current_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "rooms_available": [100, 100],
            "rooms_sold": [80, 90],
            "room_revenue": [9600.0, 11700.0],
        }
    )
    prior_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-03-01"]),
            "rooms_available": [100],
            "rooms_sold": [70],
            "room_revenue": [7700.0],
        }
    )

    yoy_df = build_yoy_comparison(current_df, prior_df)
    summary = summarize_yoy(yoy_df)

    assert summary["matched_rows"] == 1
    assert summary["missing_rows"] == 1
    assert summary["incomplete_rows"] == 0


def test_summarize_yoy_returns_nan_when_no_comparable_prior_data() -> None:
    current_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-03-01", "2025-03-02"]),
            "rooms_available": [100, 100],
            "rooms_sold": [80, 90],
            "room_revenue": [9600.0, 11700.0],
        }
    )
    prior_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2024-02-01"]),
            "rooms_available": [100],
            "rooms_sold": [70],
            "room_revenue": [7700.0],
        }
    )

    yoy_df = build_yoy_comparison(current_df, prior_df)
    summary = summarize_yoy(yoy_df)

    assert summary["matched_rows"] == 0
    assert summary["missing_rows"] == 2
    assert summary["has_comparable_data"] is False
    assert pd.isna(summary["avg_stly_adr"])
    assert pd.isna(summary["total_stly_revenue"])
    assert pd.isna(summary["adr_change_pct"])
    assert pd.isna(summary["revenue_change_pct"])
