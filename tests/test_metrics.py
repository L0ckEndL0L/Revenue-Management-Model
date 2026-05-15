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


def test_calculate_daily_metrics_hand_checked_examples() -> None:
    """
    Verify KPI calculations with hand-checked examples for Week 4 milestone.
    
    Examples:
    - 2026-03-01: 100 rooms available, 80 sold, $9600 revenue
    - 2026-03-02: 100 rooms available, 90 sold, $11700 revenue
    - 2026-03-03: 100 rooms available, 70 sold, $7700 revenue
    """
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
            "rooms_available": [100, 100, 100],
            "rooms_sold": [80, 90, 70],
            "room_revenue": [9600.0, 11700.0, 7700.0],
        }
    )

    out = calculate_daily_metrics(df)

    # Row 0: 2026-03-01
    # occupancy = 80 / 100 = 0.80
    # occupancy_pct = 0.80 * 100 = 80.00%
    # ADR = 9600 / 80 = 120.00
    # RevPAR = 9600 / 100 = 96.00
    assert out.loc[0, "occupancy"] == 0.80
    assert out.loc[0, "occupancy_pct"] == 80.00
    assert out.loc[0, "adr"] == 120.00
    assert out.loc[0, "revpar"] == 96.00
    
    # Row 1: 2026-03-02
    # occupancy = 90 / 100 = 0.90
    # occupancy_pct = 0.90 * 100 = 90.00%
    # ADR = 11700 / 90 = 130.00
    # RevPAR = 11700 / 100 = 117.00
    assert out.loc[1, "occupancy"] == 0.90
    assert out.loc[1, "occupancy_pct"] == 90.00
    assert out.loc[1, "adr"] == 130.00
    assert out.loc[1, "revpar"] == 117.00
    
    # Row 2: 2026-03-03
    # occupancy = 70 / 100 = 0.70
    # occupancy_pct = 0.70 * 100 = 70.00%
    # ADR = 7700 / 70 = 110.00
    # RevPAR = 7700 / 100 = 77.00
    assert out.loc[2, "occupancy"] == 0.70
    assert out.loc[2, "occupancy_pct"] == 70.00
    assert out.loc[2, "adr"] == 110.00
    assert out.loc[2, "revpar"] == 77.00


def test_calculate_daily_metrics_aggregated_totals() -> None:
    """
    Verify aggregated KPI calculations match hand-checked totals.
    
    Combined totals across three days:
    - total rooms_available = 300
    - total rooms_sold = 240
    - total room_revenue = 29000
    - overall occupancy = 240 / 300 = 0.80 or 80.00%
    - overall ADR = 29000 / 240 = 120.833...
    - overall RevPAR = 29000 / 300 = 96.667...
    """
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
            "rooms_available": [100, 100, 100],
            "rooms_sold": [80, 90, 70],
            "room_revenue": [9600.0, 11700.0, 7700.0],
        }
    )

    out = calculate_daily_metrics(df)

    # Test aggregated totals
    total_rooms_available = out["rooms_available"].sum()
    total_rooms_sold = out["rooms_sold"].sum()
    total_room_revenue = out["room_revenue"].sum()
    
    assert total_rooms_available == 300
    assert total_rooms_sold == 240
    assert total_room_revenue == 29000.0
    
    # Calculate aggregated metrics using formula
    overall_occupancy = total_rooms_sold / total_rooms_available
    overall_occupancy_pct = overall_occupancy * 100.0
    overall_adr = total_room_revenue / total_rooms_sold
    overall_revpar = total_room_revenue / total_rooms_available
    
    assert overall_occupancy == 0.80
    assert overall_occupancy_pct == 80.00
    assert abs(overall_adr - 120.83) < 0.01
    assert abs(overall_revpar - 96.67) < 0.01
    
    # Verify RevPAR cross-check: revpar should equal adr * occupancy
    calculated_revpar = overall_adr * overall_occupancy
    assert abs(calculated_revpar - overall_revpar) < 0.01
