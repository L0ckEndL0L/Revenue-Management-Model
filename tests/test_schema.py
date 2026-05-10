from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schema import auto_map_columns, get_missing_required_columns


def test_auto_map_columns_handles_common_aliases() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "All Room Types": [50],
            "Room Revenue": [5000.0],
            "Occupancy %": [50.0],
        }
    )

    mapping = auto_map_columns(df)

    assert mapping["stay_date"] == "Date"
    assert mapping["rooms_sold"] == "All Room Types"
    assert mapping["room_revenue"] == "Room Revenue"


def test_missing_required_columns_reports_rooms_available() -> None:
    mapping = {
        "stay_date": "Date",
        "rooms_sold": "All Room Types",
        "room_revenue": "Room Revenue",
    }

    missing = get_missing_required_columns(mapping)

    assert "rooms_available" in missing
