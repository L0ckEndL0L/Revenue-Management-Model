from __future__ import annotations

import math

import pandas as pd

from src.metrics import calculate_daily_metrics
from src.pipeline_budget_forecast import (
    build_month_forecast_budget_context,
    build_monthly_forecast_budget_summaries,
)


def test_build_month_forecast_budget_context_returns_numeric_budget_fields(tmp_path) -> None:
    historical_df = pd.DataFrame(
        {
            "stay_date": pd.date_range("2026-07-01", periods=6, freq="D"),
            "rooms_available": [100, 100, 100, 100, 100, 100],
            "rooms_sold": [50, 55, 60, 58, 62, 64],
            "room_revenue": [6000, 6600, 7200, 6960, 7440, 7680],
        }
    )
    historical_metrics = calculate_daily_metrics(historical_df)

    future_context = pd.DataFrame(
        {
            "stay_date": pd.date_range("2026-07-07", periods=4, freq="D"),
            "rooms_available": [100, 100, 100, 100],
            "rooms_sold": [40, 42, 45, 44],
            "room_revenue": [5200, 5460, 5850, 5720],
            "current_rate": [130, 130, 130, 130],
            "forecast_rooms_sold": [55, 56, 58, 57],
            "forecast_occ": [0.55, 0.56, 0.58, 0.57],
        }
    )

    stly_df = pd.DataFrame(
        {
            "stay_date": pd.date_range("2025-07-01", periods=4, freq="D"),
            "stly_occupancy": [0.60, 0.62, 0.61, 0.63],
            "stly_rooms_sold": [60, 62, 61, 63],
            "stly_adr": [120, 121, 122, 123],
            "stly_revenue": [7200, 7502, 7442, 7749],
        }
    )

    budget_path = tmp_path / "budget.csv"
    pd.DataFrame(
        [{"year": 2026, "month": 7, "budget_revenue": 250000.0}]
    ).to_csv(budget_path, index=False)

    context = build_month_forecast_budget_context(
        future_context=future_context,
        historical_df=historical_df,
        historical_metrics=historical_metrics,
        stly_df=stly_df,
        as_of_date=pd.Timestamp("2026-07-10"),
        default_current_rate=130.0,
        budget_path=str(budget_path),
        config={},
    )

    assert context.target_year == 2026
    assert context.target_month == 7
    assert {"stay_date", "forecast_rooms_sold", "rooms_sold"}.issubset(context.month_future.columns)

    for key in [
        "month_end_forecast",
        "variance_to_budget_abs",
        "required_adr_remaining",
        "remaining_budget",
    ]:
        assert key in context.budget_summary
        assert isinstance(context.budget_summary[key], (int, float))
        assert math.isfinite(float(context.budget_summary[key]))

    assert isinstance(context.budget_gap, float)
    assert isinstance(context.remaining_budget_total, float)
    assert isinstance(context.required_adr_remaining, float)


def test_build_monthly_budget_summaries_uses_each_months_target(tmp_path) -> None:
    historical_df = pd.DataFrame(
        {
            "stay_date": pd.date_range("2025-01-01", periods=60, freq="D"),
            "rooms_available": [100] * 60,
            "rooms_sold": [60] * 60,
            "room_revenue": [7200.0] * 60,
        }
    )
    historical_metrics = calculate_daily_metrics(historical_df)
    future_context = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-01-15", "2026-02-15"]),
            "rooms_available": [100, 100],
            "rooms_sold": [40, 45],
            "room_revenue": [5200.0, 6075.0],
            "current_rate": [130.0, 135.0],
            "forecast_rooms_sold": [55.0, 60.0],
            "forecast_occ": [0.55, 0.60],
        }
    )
    budget_path = tmp_path / "annual_budget.csv"
    pd.DataFrame(
        [
            {"year": 2026, "month": 1, "budget_revenue": 220000.0},
            {"year": 2026, "month": 2, "budget_revenue": 250000.0},
        ]
    ).to_csv(budget_path, index=False)

    summaries = build_monthly_forecast_budget_summaries(
        future_context=future_context,
        historical_df=historical_df,
        historical_metrics=historical_metrics,
        stly_df=None,
        as_of_date=pd.Timestamp("2025-12-31"),
        default_current_rate=130.0,
        budget_path=str(budget_path),
        config={},
    )

    assert summaries[["month", "month_name"]].to_dict("records") == [
        {"month": 1, "month_name": "January"},
        {"month": 2, "month_name": "February"},
    ]
    assert summaries["monthly_budget"].round(2).tolist() == [220000.0, 250000.0]
