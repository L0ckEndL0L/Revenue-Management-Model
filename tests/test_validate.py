from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validate import validate_data


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


def test_validate_data_fills_missing_current_rate() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-01-01"]),
            "rooms_available": [10],
            "rooms_sold": [5],
            "room_revenue": [500.0],
            "current_rate": [None],
        }
    )

    cleaned, _ = validate_data(df, allow_overbooking=True, default_current_rate=120.0)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["current_rate"] == 120.0
