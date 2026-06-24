from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validate import validate_data, validate_required_fields_for_yoy


def test_validate_data_blocks_overbooking_when_disabled() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-01-01"]),
            "rooms_available": [10],
            "rooms_sold": [12],
            "room_revenue": [1200.0],
            "current_rate": [100.0],
        }
    )

    cleaned, result = validate_data(df, allow_overbooking=False)

    assert len(cleaned) == 0
    assert result.invalid_rows == 1


def test_validate_data_derives_missing_current_rate_from_report_adr_without_warning() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-01-01"]),
            "rooms_available": [10],
            "rooms_sold": [5],
            "room_revenue": [500.0],
            "adr": [100.0],
            "current_rate": [None],
        }
    )

    cleaned, result = validate_data(df, allow_overbooking=True, default_current_rate=120.0)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["current_rate"] == 100.0
    assert not any(issue["issue_type"] == "CURRENT_RATE_FILLED" for issue in result.issues)


def test_validate_data_derives_missing_current_rate_from_revenue_per_room_without_warning() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-01-01"]),
            "rooms_available": [10],
            "rooms_sold": [5],
            "room_revenue": [550.0],
            "current_rate": [None],
        }
    )

    cleaned, result = validate_data(df, allow_overbooking=True, default_current_rate=120.0)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["current_rate"] == 110.0
    assert not any(issue["issue_type"] == "CURRENT_RATE_FILLED" for issue in result.issues)


def test_validate_data_fills_missing_current_rate_when_no_report_rate_available() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-01-01"]),
            "rooms_available": [10],
            "rooms_sold": [0],
            "room_revenue": [0.0],
            "current_rate": [None],
        }
    )

    cleaned, result = validate_data(df, allow_overbooking=True, default_current_rate=120.0)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["current_rate"] == 120.0
    assert any(issue["issue_type"] == "CURRENT_RATE_FILLED" for issue in result.issues)


def test_validate_required_fields_for_yoy_reports_missing_fields() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-01-01"]),
            "rooms_sold": [5],
            "room_revenue": [500.0],
        }
    )

    check = validate_required_fields_for_yoy(
        df,
        ["stay_date", "rooms_available", "rooms_sold", "room_revenue"],
        dataset_label="current_year",
    )

    assert check["available"] is True
    assert check["is_complete"] is False
    assert check["missing_fields"] == ["rooms_available"]
