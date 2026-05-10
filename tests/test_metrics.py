from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import calculate_daily_metrics


def test_calculate_daily_metrics_core_kpis() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "rooms_available": [100, 80],
            "rooms_sold": [75, 40],
            "room_revenue": [15000.0, 6000.0],
        }
    )

    out = calculate_daily_metrics(df)

    assert out.loc[0, "occupancy"] == 0.75
    assert out.loc[0, "adr"] == 200.0
    assert out.loc[0, "revpar"] == 150.0
    assert out.loc[1, "occupancy"] == 0.5


def test_calculate_daily_metrics_zero_division_handling() -> None:
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-01-01"]),
            "rooms_available": [0],
            "rooms_sold": [0],
            "room_revenue": [0.0],
        }
    )

    out = calculate_daily_metrics(df)

    assert out.loc[0, "occupancy"] == 0.0
    assert out.loc[0, "adr"] == 0.0
    assert out.loc[0, "revpar"] == 0.0
