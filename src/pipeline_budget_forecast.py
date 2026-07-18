"""Month forecast and budget projection context for the RMS pipeline."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.budget import calculate_budget_progress, prepare_monthly_budget_targets


@dataclass(frozen=True)
class MonthForecastBudgetContext:
    budget_summary: dict
    target_year: int
    target_month: int
    anchor_date: pd.Timestamp
    month_future: pd.DataFrame
    forecast_rate_series: pd.Series
    actual_revenue_to_date: float
    forecast_revenue_remaining: float
    month_end_forecast: float
    monthly_room_capacity: float
    on_books_rooms_total: float
    remaining_budget_total: float
    budget_gap: float
    required_adr_remaining: float


def build_month_forecast_budget_context(
    *,
    future_context: pd.DataFrame,
    historical_df: pd.DataFrame,
    historical_metrics: pd.DataFrame,
    stly_df: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    default_current_rate,
    budget_path: str | None,
    config: dict,
) -> MonthForecastBudgetContext:
    """Build month-scoped on-books forecast and optional budget progress."""
    _future_dates = pd.to_datetime(future_context["stay_date"], errors="coerce").dropna() if len(future_context) else pd.Series(dtype="datetime64[ns]")
    _hist_dates = pd.to_datetime(historical_df["stay_date"], errors="coerce").dropna() if len(historical_df) else pd.Series(dtype="datetime64[ns]")

    if len(_future_dates):
        anchor_date = _future_dates.min()
    elif len(_hist_dates):
        anchor_date = _hist_dates.max()
    else:
        anchor_date = as_of_date

    target_year = int(anchor_date.year)
    target_month = int(anchor_date.month)

    stly_year: int | None = None
    if stly_df is not None and len(stly_df) > 0:
        _stly_dates = pd.to_datetime(stly_df["stay_date"], errors="coerce")
        _stly_years_for_month = (
            _stly_dates[_stly_dates.dt.month == target_month]
            .dt.year
            .dropna()
            .unique()
        )
        _prior_years = sorted([int(y) for y in _stly_years_for_month if int(y) < target_year], reverse=True)
        if _prior_years:
            stly_year = _prior_years[0]

    month_future = future_context[
        (pd.to_datetime(future_context["stay_date"], errors="coerce").dt.year == target_year)
        & (pd.to_datetime(future_context["stay_date"], errors="coerce").dt.month == target_month)
    ].copy() if len(future_context) else future_context.copy()

    fallback_future_rate = float(
        historical_metrics.get("adr", pd.Series(dtype=float)).dropna().median()
    ) if "adr" in historical_metrics.columns and historical_metrics["adr"].notna().any() else float(default_current_rate or 120.0)

    hist_adr = pd.to_numeric(historical_metrics.get("adr", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(hist_adr) >= 5:
        rate_floor_hist = float(hist_adr.quantile(0.10))
        rate_ceiling_hist = float(hist_adr.quantile(0.90))
        if rate_ceiling_hist < rate_floor_hist:
            rate_floor_hist, rate_ceiling_hist = rate_ceiling_hist, rate_floor_hist
    else:
        rate_floor_hist = max(0.0, float(fallback_future_rate) * 0.8)
        rate_ceiling_hist = float(fallback_future_rate) * 1.2

    forecast_rate_series = pd.to_numeric(month_future.get("current_rate", pd.Series(dtype=float)), errors="coerce")
    forecast_rate_series = forecast_rate_series.where(forecast_rate_series > 0).fillna(fallback_future_rate)
    forecast_rate_series = forecast_rate_series.clip(lower=rate_floor_hist, upper=rate_ceiling_hist)

    actual_revenue_to_date = 0.0
    if len(month_future):
        otb_revenue = pd.to_numeric(month_future.get("room_revenue", pd.Series(dtype=float)), errors="coerce")
        otb_rooms = pd.to_numeric(month_future.get("rooms_sold", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        otb_rate = pd.to_numeric(month_future.get("current_rate", pd.Series(dtype=float)), errors="coerce")
        otb_rate = otb_rate.where(otb_rate > 0).fillna(float(forecast_rate_series.median() if len(forecast_rate_series) else fallback_future_rate))
        derived_otb_revenue = otb_rooms * otb_rate
        otb_revenue = otb_revenue.where(otb_revenue > 0, derived_otb_revenue).fillna(0.0)
        actual_revenue_to_date = float(otb_revenue.sum())

    stly_has_data = False
    stly_month_revenue = 0.0

    if len(month_future):
        positive_inventory = pd.to_numeric(month_future.get("rooms_available", pd.Series(dtype=float)), errors="coerce")
        positive_inventory = positive_inventory[positive_inventory > 0]
        if len(positive_inventory) == 0:
            positive_inventory = pd.to_numeric(historical_df.get("rooms_available", pd.Series(dtype=float)), errors="coerce")
            positive_inventory = positive_inventory[positive_inventory > 0]
        derived_total_rooms = float(positive_inventory.median()) if len(positive_inventory) else 0.0
        days_in_month = monthrange(target_year, target_month)[1]
        monthly_room_capacity = max(0.0, derived_total_rooms * float(days_in_month))

        on_books_rooms_total = float(
            pd.to_numeric(month_future.get("rooms_sold", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        )

        if monthly_room_capacity <= 0 and on_books_rooms_total > 0:
            monthly_room_capacity = on_books_rooms_total

        stly_month_adr = 0.0
        stly_prior = pd.DataFrame()
        if stly_df is not None and len(stly_df) > 0:
            stly_dates = pd.to_datetime(stly_df["stay_date"], errors="coerce")
            if stly_year is not None:
                stly_prior = stly_df[
                    (stly_dates.dt.month == target_month)
                    & (stly_dates.dt.year == stly_year)
                ].copy()
            if len(stly_prior) == 0:
                stly_prior = stly_df[stly_dates.dt.month == target_month].copy()
            if len(stly_prior) > 0:
                stly_occ_col = pd.to_numeric(stly_prior.get("stly_occupancy", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                stly_adr_col = pd.to_numeric(stly_prior.get("stly_adr", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                stly_rev_col = pd.to_numeric(stly_prior.get("stly_revenue", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                stly_rooms_col = pd.to_numeric(stly_prior.get("stly_rooms_sold", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                stly_month_occ = float(stly_occ_col.mean())
                stly_month_adr = float(stly_adr_col[stly_adr_col > 0].mean()) if (stly_adr_col > 0).any() else fallback_future_rate
                stly_month_revenue = float(stly_rev_col.sum())
                if stly_month_revenue <= 0 and stly_month_adr > 0 and float(stly_rooms_col.sum()) > 0:
                    stly_month_revenue = float(stly_rooms_col.sum()) * stly_month_adr
                stly_has_data = stly_month_occ > 0

        stly_otb_rooms = 0.0
        if stly_has_data and "stly_rooms_sold" in stly_prior.columns:
            stly_otb_rooms = float(
                pd.to_numeric(stly_prior["stly_rooms_sold"], errors="coerce").fillna(0.0).sum()
            )
        if stly_otb_rooms == 0.0 and stly_has_data and stly_month_adr > 0:
            stly_otb_rooms = stly_month_revenue / stly_month_adr

        stly_close_factor = float(config.get("stly_close_factor", 1.15))
        if stly_has_data and stly_otb_rooms > 0:
            stly_final_rooms = stly_otb_rooms * stly_close_factor
            stly_remaining_pickup = max(0.0, stly_final_rooms - stly_otb_rooms)
        else:
            stly_remaining_pickup = 0.0

        config_pace_override = config.get("pickup_pace_factor", None)
        if config_pace_override is not None:
            pickup_pace_factor = float(config_pace_override)
        elif stly_otb_rooms > 0:
            raw_ratio = on_books_rooms_total / stly_otb_rooms
            pickup_pace_factor = float(np.clip(raw_ratio, 0.5, 2.0))
        else:
            pickup_pace_factor = 1.0

        forecast_rooms_total = on_books_rooms_total + stly_remaining_pickup * pickup_pace_factor

        if monthly_room_capacity > 0:
            forecast_rooms_total = min(forecast_rooms_total, monthly_room_capacity)

        forecast_rooms_total = max(forecast_rooms_total, on_books_rooms_total)

        forecast_occ_month = (
            forecast_rooms_total / monthly_room_capacity
            if monthly_room_capacity > 0
            else 0.0
        )

        pickup_rooms_total = max(0.0, forecast_rooms_total - on_books_rooms_total)

        on_books_adr = (
            actual_revenue_to_date / on_books_rooms_total
            if on_books_rooms_total > 0
            else fallback_future_rate
        )
        if stly_has_data and stly_month_adr > 0:
            forecast_adr = on_books_adr * 0.60 + stly_month_adr * 0.40
        else:
            avg_forecast_rate = float(forecast_rate_series.mean()) if len(forecast_rate_series) else fallback_future_rate
            forecast_adr = avg_forecast_rate

        forecast_adr = max(rate_floor_hist, min(forecast_adr, rate_ceiling_hist))
        forecast_revenue_remaining = pickup_rooms_total * forecast_adr
    else:
        monthly_room_capacity = 0.0
        on_books_rooms_total = 0.0
        forecast_occ_month = 0.0
        forecast_revenue_remaining = 0.0

    projected_from_otb = actual_revenue_to_date + max(0.0, forecast_revenue_remaining)
    month_end_forecast = projected_from_otb

    if stly_has_data and stly_month_revenue > 0:
        stly_anchor_weight = float(config.get("stly_revenue_anchor_weight", 0.35))
        stly_anchor_weight = float(np.clip(stly_anchor_weight, 0.0, 1.0))
        month_end_forecast = (
            projected_from_otb * (1.0 - stly_anchor_weight)
            + stly_month_revenue * stly_anchor_weight
        )

        stly_revenue_cap_ratio = float(config.get("stly_revenue_cap_ratio", 1.10))
        if stly_revenue_cap_ratio > 0:
            month_end_forecast = min(month_end_forecast, stly_month_revenue * stly_revenue_cap_ratio)

    month_end_forecast = max(month_end_forecast, actual_revenue_to_date)

    if not bool(config.get("allow_revenue_over_capacity", False)) and monthly_room_capacity > 0:
        revenue_capacity_ceiling = monthly_room_capacity * rate_ceiling_hist
        month_end_forecast = min(month_end_forecast, revenue_capacity_ceiling)

    forecast_revenue_remaining = max(0.0, month_end_forecast - actual_revenue_to_date)

    budget_summary = {
        "actual_revenue_to_date": actual_revenue_to_date,
        "forecast_revenue_remaining": forecast_revenue_remaining,
        "month_end_forecast": month_end_forecast,
        "monthly_budget": 0.0,
        "variance_to_budget_abs": 0.0,
        "variance_to_budget_pct": 0.0,
        "remaining_budget": 0.0,
        "remaining_rooms_available": max(0.0, monthly_room_capacity - on_books_rooms_total),
        "forecast_remaining_occ": forecast_occ_month,
        "month_room_capacity": monthly_room_capacity,
        "on_books_rooms": on_books_rooms_total,
        "required_adr_remaining": 0.0,
    }

    if budget_path:
        budget_warning = None
        try:
            daily_budget_df, _ = prepare_monthly_budget_targets(
                budget_path=budget_path,
                current_year=target_year,
                current_month=target_month,
                historical_df=historical_metrics,
                daily_distribution_method=str(config.get("budget_distribution", "dow_weighted")),
            )
        except ValueError as exc:
            message = str(exc)
            if "current month/year" not in message:
                raise
            daily_budget_df = pd.DataFrame()
            budget_warning = message

        if len(daily_budget_df):
            forecast_remaining_for_budget = month_future[["stay_date", "rooms_available", "forecast_rooms_sold", "rooms_sold"]].copy()
            budget_rate_series = pd.to_numeric(month_future.get("current_rate", pd.Series(dtype=float)), errors="coerce")
            budget_rate_series = budget_rate_series.where(budget_rate_series > 0).fillna(
                float(forecast_rate_series.median() if len(forecast_rate_series) else fallback_future_rate)
            )
            pickup_rooms_for_budget = (
                pd.to_numeric(forecast_remaining_for_budget["forecast_rooms_sold"], errors="coerce").fillna(0.0)
                - pd.to_numeric(forecast_remaining_for_budget["rooms_sold"], errors="coerce").fillna(0.0)
            ).clip(lower=0.0)
            forecast_remaining_for_budget["forecast_revenue"] = pickup_rooms_for_budget * budget_rate_series

            month_actual_for_budget = month_future[["stay_date", "room_revenue", "rooms_available", "rooms_sold"]].copy()
            month_actual_for_budget["room_revenue"] = pd.to_numeric(month_actual_for_budget["room_revenue"], errors="coerce")
            if month_actual_for_budget["room_revenue"].isna().any() or (month_actual_for_budget["room_revenue"] <= 0).all():
                fallback_rooms = pd.to_numeric(month_actual_for_budget["rooms_sold"], errors="coerce").fillna(0.0)
                fallback_rates = pd.to_numeric(month_future.get("current_rate", pd.Series(dtype=float)), errors="coerce")
                fallback_rates = fallback_rates.where(fallback_rates > 0).fillna(float(forecast_rate_series.median() if len(forecast_rate_series) else fallback_future_rate))
                month_actual_for_budget["room_revenue"] = month_actual_for_budget["room_revenue"].where(
                    month_actual_for_budget["room_revenue"] > 0,
                    fallback_rooms * fallback_rates,
                )

            budget_as_of_date = as_of_date
            if len(month_actual_for_budget):
                budget_as_of_date = pd.to_datetime(month_actual_for_budget["stay_date"], errors="coerce").max()
                if pd.isna(budget_as_of_date):
                    budget_as_of_date = as_of_date

            budget_summary = calculate_budget_progress(
                month_actual_df=month_actual_for_budget,
                forecast_remaining_df=forecast_remaining_for_budget,
                daily_budget_df=daily_budget_df,
                as_of_date=budget_as_of_date,
            )
        elif budget_warning:
            budget_summary["budget_warning"] = budget_warning

    remaining_budget_total = float(budget_summary.get("remaining_budget", 0.0))
    budget_gap = remaining_budget_total
    required_adr_remaining = float(budget_summary.get("required_adr_remaining", 0.0))

    return MonthForecastBudgetContext(
        budget_summary=budget_summary,
        target_year=target_year,
        target_month=target_month,
        anchor_date=anchor_date,
        month_future=month_future,
        forecast_rate_series=forecast_rate_series,
        actual_revenue_to_date=actual_revenue_to_date,
        forecast_revenue_remaining=forecast_revenue_remaining,
        month_end_forecast=month_end_forecast,
        monthly_room_capacity=monthly_room_capacity,
        on_books_rooms_total=on_books_rooms_total,
        remaining_budget_total=remaining_budget_total,
        budget_gap=budget_gap,
        required_adr_remaining=required_adr_remaining,
    )


def build_monthly_forecast_budget_summaries(
    *,
    future_context: pd.DataFrame,
    historical_df: pd.DataFrame,
    historical_metrics: pd.DataFrame,
    stly_df: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    default_current_rate,
    budget_path: str | None,
    config: dict,
) -> pd.DataFrame:
    """Build calendar-sorted forecast and budget summaries for every future month."""
    if future_context is None or len(future_context) == 0:
        return pd.DataFrame()

    dated = future_context.copy()
    dated["stay_date"] = pd.to_datetime(dated["stay_date"], errors="coerce")
    dated = dated.dropna(subset=["stay_date"])
    if len(dated) == 0:
        return pd.DataFrame()

    rows: list[dict] = []
    month_keys = (
        dated[["stay_date"]]
        .assign(year=lambda frame: frame["stay_date"].dt.year, month=lambda frame: frame["stay_date"].dt.month)
        [["year", "month"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
    )
    for month_key in month_keys.itertuples(index=False):
        month_future = dated[
            (dated["stay_date"].dt.year == int(month_key.year))
            & (dated["stay_date"].dt.month == int(month_key.month))
        ].copy()
        context = build_month_forecast_budget_context(
            future_context=month_future,
            historical_df=historical_df,
            historical_metrics=historical_metrics,
            stly_df=stly_df,
            as_of_date=as_of_date,
            default_current_rate=default_current_rate,
            budget_path=budget_path,
            config=config,
        )
        rows.append(
            {
                "year": int(context.target_year),
                "month": int(context.target_month),
                "month_name": context.anchor_date.strftime("%B"),
                **context.budget_summary,
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
