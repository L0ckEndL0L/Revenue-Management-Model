from __future__ import annotations

import pandas as pd

from src.pipeline_inputs import select_user_comparison_frames


def test_uploaded_adjacent_year_uses_repo_stly_for_backtest_context() -> None:
    historical_metrics = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2025-07-01", "2025-07-02"]),
            "rooms_available": [99, 99],
            "rooms_sold": [83, 81],
            "room_revenue": [10772.48, 9439.53],
            "occupancy": [83 / 99, 81 / 99],
            "adr": [129.79, 116.54],
        }
    )
    future_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "rooms_available": [99, 99],
            "rooms_sold": [28, 20],
            "room_revenue": [3934.81, 2884.91],
        }
    )
    repo_stly_df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2024-07-01", "2024-07-02"]),
            "stly_occupancy": [0.72, 0.74],
        }
    )

    yoy_current, yoy_prior, backtest_stly, using_uploaded = select_user_comparison_frames(
        historical_metrics,
        future_df,
        repo_stly_df,
    )

    assert using_uploaded is True
    assert yoy_current["stay_date"].dt.year.eq(2026).all()
    assert yoy_prior["stay_date"].dt.year.eq(2025).all()
    assert backtest_stly is repo_stly_df
