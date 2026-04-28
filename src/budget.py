"""
budget.py
Budget parsing, daily target expansion, and month progress calculations for Week 4.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

BudgetFormat = Literal["monthly", "daily"]


def _load_tabular_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Budget file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Budget file must be CSV or XLSX")


def detect_budget_format(budget_df: pd.DataFrame) -> BudgetFormat:
    """Detect whether budget data is monthly or daily format."""
    columns = {str(c).strip().lower() for c in budget_df.columns}

    has_monthly = {"year", "month", "budget_revenue"}.issubset(columns)
    has_daily = {"stay_date", "budget_revenue"}.issubset(columns)

    if has_daily:
        # Prefer explicit daily targets when both shapes are present.
        return "daily"
    if has_monthly:
        return "monthly"

    raise ValueError(
        "Unable to detect budget format. Expected monthly columns "
        "(year, month, budget_revenue) or daily columns (stay_date, budget_revenue)."
    )


def _normalize_budget_columns(budget_df: pd.DataFrame) -> pd.DataFrame:
    normalized = budget_df.copy()
    normalized.columns = [str(c).strip().lower() for c in normalized.columns]
    return normalized


def _build_dow_weights(historical_df: Optional[pd.DataFrame], target_month: int) -> Dict[int, float]:
    """Build day-of-week revenue weights from historical data if available."""
    if historical_df is None or len(historical_df) == 0:
        return {i: 1.0 / 7.0 for i in range(7)}

    hist = historical_df.copy()
    if "stay_date" not in hist.columns or "room_revenue" not in hist.columns:
        return {i: 1.0 / 7.0 for i in range(7)}

    hist["stay_date"] = pd.to_datetime(hist["stay_date"], errors="coerce")
    hist = hist.dropna(subset=["stay_date"])
    hist = hist[hist["stay_date"].dt.month == target_month]

    if len(hist) == 0:
        hist = historical_df.copy()
        hist["stay_date"] = pd.to_datetime(hist["stay_date"], errors="coerce")
        hist = hist.dropna(subset=["stay_date"])

    if len(hist) == 0:
        return {i: 1.0 / 7.0 for i in range(7)}

    hist["dow"] = hist["stay_date"].dt.dayofweek
    revenue_by_dow = hist.groupby("dow")["room_revenue"].sum().reindex(range(7), fill_value=0.0)
    total = float(revenue_by_dow.sum())
    if total <= 0:
        return {i: 1.0 / 7.0 for i in range(7)}

    return {int(i): float(v / total) for i, v in revenue_by_dow.items()}


def expand_monthly_budget_to_daily(
    monthly_budget: float,
    year: int,
    month: int,
    method: str = "dow_weighted",
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Expand a monthly budget target into per-day budget amounts."""
    days_in_month = monthrange(year, month)[1]
    start = pd.Timestamp(year=year, month=month, day=1)
    dates = pd.date_range(start=start, periods=days_in_month, freq="D")

    if method == "equal":
        per_day = monthly_budget / days_in_month if days_in_month > 0 else 0.0
        daily = pd.DataFrame({"stay_date": dates, "budget_revenue": per_day})
        return daily

    weights = _build_dow_weights(historical_df=historical_df, target_month=month)
    daily = pd.DataFrame({"stay_date": dates})
    daily["dow_weight"] = daily["stay_date"].dt.dayofweek.map(weights).fillna(1.0 / 7.0)

    if daily["dow_weight"].sum() <= 0:
        daily["budget_revenue"] = monthly_budget / days_in_month if days_in_month > 0 else 0.0
    else:
        normalized_weight = daily["dow_weight"] / daily["dow_weight"].sum()
        daily["budget_revenue"] = normalized_weight * monthly_budget

    return daily[["stay_date", "budget_revenue"]]


def prepare_monthly_budget_targets(
    budget_path: str,
    current_year: int,
    current_month: int,
    historical_df: Optional[pd.DataFrame] = None,
    daily_distribution_method: str = "dow_weighted",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load budget file, detect format, and return daily budget targets for current month."""
    raw = _load_tabular_file(budget_path)
    budget_df = _normalize_budget_columns(raw)
    budget_format = detect_budget_format(budget_df)

    if budget_format == "daily":
        daily = budget_df[["stay_date", "budget_revenue"]].copy()
        daily["stay_date"] = pd.to_datetime(daily["stay_date"], errors="coerce")
        daily["budget_revenue"] = pd.to_numeric(daily["budget_revenue"], errors="coerce").fillna(0.0)
        daily = daily.dropna(subset=["stay_date"])
        daily = daily[
            (daily["stay_date"].dt.year == current_year)
            & (daily["stay_date"].dt.month == current_month)
        ].copy()

        if len(daily) == 0:
            raise ValueError("Daily budget file has no records for the current month/year")

        monthly_budget = float(daily["budget_revenue"].sum())
        metadata = {
            "budget_format": "daily",
            "monthly_budget": monthly_budget,
        }
        return daily.sort_values("stay_date").reset_index(drop=True), metadata

    monthly = budget_df[["year", "month", "budget_revenue"]].copy()
    monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce").astype("Int64")
    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce").astype("Int64")
    monthly["budget_revenue"] = pd.to_numeric(monthly["budget_revenue"], errors="coerce").fillna(0.0)

    current_row = monthly[
        (monthly["year"] == current_year) & (monthly["month"] == current_month)
    ]

    if len(current_row) == 0:
        raise ValueError("Monthly budget file has no target for the current month/year")

    monthly_budget = float(current_row.iloc[0]["budget_revenue"])
    daily = expand_monthly_budget_to_daily(
        monthly_budget=monthly_budget,
        year=current_year,
        month=current_month,
        method=daily_distribution_method,
        historical_df=historical_df,
    )

    metadata = {
        "budget_format": "monthly",
        "monthly_budget": monthly_budget,
    }
    return daily.sort_values("stay_date").reset_index(drop=True), metadata


def calculate_required_adr_remaining(
    remaining_budget: float,
    remaining_rooms_available: float,
    forecast_remaining_occ: float,
) -> float:
    """Compute required ADR for remaining days to achieve budget."""
    # If target already met or no remaining sellable capacity, required ADR is zero.
    if remaining_budget <= 0:
        return 0.0
    if remaining_rooms_available <= 0:
        return 0.0
    if forecast_remaining_occ <= 0:
        return 0.0

    denominator = remaining_rooms_available * forecast_remaining_occ
    if denominator <= 0:
        return 0.0
    return float(remaining_budget / denominator)


def calculate_budget_progress(
    month_actual_df: pd.DataFrame,
    forecast_remaining_df: pd.DataFrame,
    daily_budget_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> Dict[str, float]:
    """Calculate month-to-date budget progress and month-end projection."""
    df = month_actual_df.copy()
    df["stay_date"] = pd.to_datetime(df["stay_date"], errors="coerce")

    actual_to_date = float(df.loc[df["stay_date"] <= as_of_date, "room_revenue"].sum())

    remaining_rooms_available = float(forecast_remaining_df.get("rooms_available", pd.Series(dtype=float)).sum())
    forecast_remaining_rooms_sold = float(forecast_remaining_df.get("forecast_rooms_sold", pd.Series(dtype=float)).sum())
    forecast_remaining_revenue = float(forecast_remaining_df.get("forecast_revenue", pd.Series(dtype=float)).sum())

    monthly_budget = float(daily_budget_df["budget_revenue"].sum())
    month_end_forecast = actual_to_date + forecast_remaining_revenue

    remaining_budget = monthly_budget - actual_to_date
    forecast_remaining_occ = (
        forecast_remaining_rooms_sold / remaining_rooms_available
        if remaining_rooms_available > 0
        else 0.0
    )
    required_adr_remaining = calculate_required_adr_remaining(
        remaining_budget=remaining_budget,
        remaining_rooms_available=remaining_rooms_available,
        forecast_remaining_occ=forecast_remaining_occ,
    )

    variance_abs = month_end_forecast - monthly_budget
    variance_pct = (variance_abs / monthly_budget * 100.0) if monthly_budget > 0 else 0.0

    return {
        "actual_revenue_to_date": actual_to_date,
        "forecast_revenue_remaining": forecast_remaining_revenue,
        "month_end_forecast": month_end_forecast,
        "monthly_budget": monthly_budget,
        "variance_to_budget_abs": variance_abs,
        "variance_to_budget_pct": variance_pct,
        "remaining_budget": remaining_budget,
        "remaining_rooms_available": remaining_rooms_available,
        "forecast_remaining_occ": forecast_remaining_occ,
        "required_adr_remaining": required_adr_remaining,
    }


def current_month_context() -> Tuple[int, int, pd.Timestamp]:
    """Return (year, month, today) using system time."""
    now = datetime.now()
    today = pd.Timestamp(now.date())
    return int(now.year), int(now.month), today
