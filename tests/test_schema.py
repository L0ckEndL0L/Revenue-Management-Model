from pathlib import Path
import sys
import io

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import clean_report_dataframe, process_dataframe, read_table_source
from src.schema import auto_map_columns, apply_column_mapping, get_missing_required_columns


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


def test_apply_column_mapping_allows_same_source_for_optional_fields_without_duplicate_columns() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2026-03-01"],
            "Rooms": [10],
            "Rate": [150.0],
        }
    )

    mapped = apply_column_mapping(
        df,
        {
            "stay_date": "Date",
            "rooms_sold": "Rooms",
            "adr": "Rate",
            "current_rate": "Rate",
        },
    )

    assert list(mapped.columns) == ["stay_date", "rooms_sold", "adr", "current_rate"]
    assert mapped["adr"].iloc[0] == mapped["current_rate"].iloc[0]


def test_missing_required_columns_reports_rooms_available() -> None:
    mapping = {
        "stay_date": "Date",
        "rooms_sold": "All Room Types",
        "room_revenue": "Room Revenue",
    }

    missing = get_missing_required_columns(mapping)

    assert "rooms_available" in missing


def test_clean_report_dataframe_repairs_shifted_csv_room_revenue() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["07/01/25", "07/02/25"],
            "": ["", ""],
            "Occupancy %": ["84.7%", "84.4%"],
            "All Room Types": [83, 81],
            "A": [127, 121],
            "C": [7, 8],
            "Unnamed: 6": [0, 0],
            "I": ["", ""],
            "Unnamed: 8": ["$ 10,772.48", "$ 9,439.53"],
            "Room Revenue": ["", ""],
            "ADR": ["$ 129.79", "$ 116.54"],
        }
    )

    cleaned = clean_report_dataframe(raw)

    assert cleaned.loc[0, "Room Revenue"] == 10772.48
    assert cleaned.loc[1, "Room Revenue"] == 9439.53


def test_process_dataframe_preserves_shifted_csv_revenue_values() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["07/01/25", "07/02/25"],
            "": ["", ""],
            "Occupancy %": ["84.7%", "84.4%"],
            "All Room Types": [83, 81],
            "A": [127, 121],
            "C": [7, 8],
            "Unnamed: 6": [0, 0],
            "I": ["", ""],
            "Unnamed: 8": ["$ 10,772.48", "$ 9,439.53"],
            "Room Revenue": ["", ""],
            "ADR": ["$ 129.79", "$ 116.54"],
        }
    )

    processed = process_dataframe(
        clean_report_dataframe(raw),
        interactive=False,
        required_columns=["stay_date", "rooms_sold", "room_revenue"],
    )

    assert processed.loc[0, "room_revenue"] == 10772.48
    assert processed.loc[1, "adr"] == 116.54


def test_read_table_source_aligns_csv_rows_missing_blank_date_placeholder() -> None:
    csv_text = "\n".join(
        [
            "Start Date:,07/01/2025,,,,,,,,,",
            "End Date:,09/30/2025,,,,,,,,,",
            "Date,,Occupancy %,All Room Types,A,C,,I,,Room Revenue,ADR",
            '07/01/25,,84.7%,83,127,7,0,,"$ 10,772.48",,$ 129.79',
            '08/04/25,89.6%,86,129,9,0,,"$ 12,408.44",,$ 144.28',
        ]
    )

    df = read_table_source(io.StringIO(csv_text), filename="report.csv")

    assert df.loc[0, "Occupancy %"] == "84.7%"
    assert df.loc[0, "All Room Types"] == "83"
    assert df.loc[1, "Occupancy %"] == "89.6%"
    assert df.loc[1, "All Room Types"] == "86"
    assert df.loc[1, "Room Revenue"] == 12408.44
    assert df.loc[1, "ADR"] == "$ 144.28"
