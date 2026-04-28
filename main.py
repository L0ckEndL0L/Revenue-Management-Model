"""
main.py
RMS entry point with forward-looking on-books pricing simulation.
"""

from __future__ import annotations

import argparse
import sys
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.budget import calculate_budget_progress, current_month_context, prepare_monthly_budget_targets
from src.elasticity import expected_rooms_sold
from src.evaluation import (
    build_policy_evaluation_metrics,
    calculate_forecast_metrics,
    plot_current_vs_recommended_rate,
    plot_expected_revenue_uplift,
    plot_forecast_vs_actual,
    plot_priority_score_by_date,
)
from src.events import apply_event_impacts, load_events
from src.forecast import (
    baseline_forecast,
    build_future_forecast,
    enhanced_forecast,
    evaluate_backtest,
    prepare_forecast_frame,
)
from src.ingest import process_file
from src.metrics import calculate_daily_metrics, export_metrics
from src.pace import calculate_pace_analysis, load_historical_data
from src.pricing import PricingConfig, build_priority_lists, generate_rate_recommendations, simulate_elasticity_pricing
from src.utils import ensure_directory_exists, get_timestamp, print_error
from src.validate import check_data_quality, save_validation_report, validate_data


def _prepare_future_dataset(
    future_path: str | None,
    input_df: pd.DataFrame,
    interactive: bool,
    mapping: dict | None,
    as_of_date: pd.Timestamp,
    manual_rooms_available: int | None = None,
) -> pd.DataFrame:
    """Load explicit future report, or derive future rows from input when absent."""
    required_future_cols = ["stay_date", "rooms_sold"]
    if manual_rooms_available is None:
        required_future_cols.insert(1, "rooms_available")

    if future_path:
        future_df = process_file(
            future_path,
            interactive=interactive,
            user_mapping=mapping,
            required_columns=required_future_cols,
        )
    else:
        future_df = input_df[input_df["stay_date"] > as_of_date].copy()

    if len(future_df) == 0:
        return future_df

    if "room_revenue" not in future_df.columns:
        future_df["room_revenue"] = pd.NA

    if "current_rate" not in future_df.columns:
        future_df["current_rate"] = pd.NA

    # If current_rate is missing, prefer ADR from report before using global fallback.
    if "adr" in future_df.columns:
        derived_from_adr = pd.to_numeric(future_df["adr"], errors="coerce")
        current_rate = pd.to_numeric(future_df["current_rate"], errors="coerce")
        use_adr_mask = current_rate.isna() | (current_rate <= 0)
        future_df.loc[use_adr_mask, "current_rate"] = derived_from_adr.loc[use_adr_mask]

    # Secondary fallback: derive from row-level revenue / sold where available.
    if "room_revenue" in future_df.columns:
        current_rate = pd.to_numeric(future_df["current_rate"], errors="coerce")
        sold = pd.to_numeric(future_df["rooms_sold"], errors="coerce").replace(0, pd.NA)
        derived_rate = pd.to_numeric(future_df["room_revenue"], errors="coerce") / sold
        use_derived_mask = current_rate.isna() | (current_rate <= 0)
        future_df.loc[use_derived_mask, "current_rate"] = derived_rate.loc[use_derived_mask]

    # In future feeds, rooms_sold is treated as rooms on books.
    future_df["rooms_sold"] = pd.to_numeric(future_df["rooms_sold"], errors="coerce").fillna(0.0)

    # Optional manual override for total inventory.
    if manual_rooms_available is not None:
        future_df["rooms_available"] = int(manual_rooms_available)

    return future_df.sort_values("stay_date").reset_index(drop=True)


def _build_baseline_vs_new_policy(
    historical_metrics: pd.DataFrame,
    baseline_rules_df: pd.DataFrame,
    elasticity: float,
) -> pd.DataFrame:
    """Per-date backtest style comparison of baseline policy vs new simulation policy."""
    if len(historical_metrics) == 0:
        return pd.DataFrame(
            columns=[
                "stay_date",
                "current_rate",
                "baseline_rate",
                "new_policy_rate",
                "forecast_rooms_sold",
                "baseline_expected_revenue",
                "new_policy_expected_revenue",
                "expected_revenue_uplift",
            ]
        )

    eval_df = historical_metrics.copy().sort_values("stay_date")
    eval_df["dow"] = eval_df["stay_date"].dt.dayofweek
    dow_avg = eval_df.groupby("dow")["rooms_sold"].transform("mean")
    eval_df["forecast_rooms_sold"] = dow_avg.fillna(eval_df["rooms_sold"].mean())
    eval_df["current_rate"] = np.where(eval_df["rooms_sold"] > 0, eval_df["room_revenue"] / eval_df["rooms_sold"], eval_df.get("adr", 0.0))
    eval_df["on_books_eval"] = np.minimum(eval_df["rooms_sold"] * 0.70, eval_df["forecast_rooms_sold"])

    merged = eval_df.merge(
        baseline_rules_df[["stay_date", "recommended_adr"]].rename(columns={"recommended_adr": "baseline_rate"}),
        on="stay_date",
        how="left",
    )

    merged["baseline_rate"] = merged["baseline_rate"].fillna(merged["current_rate"])

    # Build simulation input using historical rows as a what-if test bed.
    sim_input = merged[
        ["stay_date", "rooms_available", "on_books_eval", "current_rate", "forecast_rooms_sold", "occupancy"]
    ].copy()
    sim_input = sim_input.rename(columns={"on_books_eval": "rooms_sold", "occupancy": "forecast_occ"})
    sim_input["pace_variance"] = np.nan
    sim_input["impact_level"] = pd.NA
    sim_input["event_pct"] = 0.0

    sim_config = PricingConfig()
    new_policy_df, _ = simulate_elasticity_pricing(
        sim_input,
        config=sim_config,
        elasticity=elasticity,
        budget_gap=0.0,
        required_adr_remaining=0.0,
    )

    merged = merged.merge(
        new_policy_df[["stay_date", "recommended_rate"]].rename(columns={"recommended_rate": "new_policy_rate"}),
        on="stay_date",
        how="left",
    )
    merged["new_policy_rate"] = merged["new_policy_rate"].fillna(merged["current_rate"])

    merged["baseline_expected_revenue"] = merged.apply(
        lambda row: row["baseline_rate"]
        * expected_rooms_sold(
            base_demand=row["forecast_rooms_sold"],
            candidate_rate=row["baseline_rate"],
            current_rate=row["current_rate"],
            elasticity=elasticity,
            on_books=row["on_books_eval"],
            rooms_available=row["rooms_available"],
        ),
        axis=1,
    )
    merged["new_policy_expected_revenue"] = merged.apply(
        lambda row: row["new_policy_rate"]
        * expected_rooms_sold(
            base_demand=row["forecast_rooms_sold"],
            candidate_rate=row["new_policy_rate"],
            current_rate=row["current_rate"],
            elasticity=elasticity,
            on_books=row["on_books_eval"],
            rooms_available=row["rooms_available"],
        ),
        axis=1,
    )
    merged["expected_revenue_uplift"] = merged["new_policy_expected_revenue"] - merged["baseline_expected_revenue"]

    return merged[
        [
            "stay_date",
            "current_rate",
            "baseline_rate",
            "new_policy_rate",
            "forecast_rooms_sold",
            "baseline_expected_revenue",
            "new_policy_expected_revenue",
            "expected_revenue_uplift",
        ]
    ].sort_values("stay_date").reset_index(drop=True)


def run_pipeline(
    input_path: str,
    future_path: str | None = None,
    budget_path: str | None = None,
    events_path: str | None = None,
    config: dict | None = None,
):
    """Run end-to-end RMS pricing workflow and return output paths + summary payload."""
    config = config or {}

    output_root = ensure_directory_exists(config.get("output_dir", "outputs"))
    output_dir = ensure_directory_exists(str(output_root / f"run_{get_timestamp()}"))

    allow_overbooking = bool(config.get("allow_overbooking", False))
    interactive = bool(config.get("interactive", False))
    elasticity = float(config.get("elasticity", 1.2))
    default_current_rate = config.get("default_current_rate")
    manual_rooms_available = config.get("manual_rooms_available")
    if manual_rooms_available is not None:
        manual_rooms_available = int(manual_rooms_available)

    pricing_config = PricingConfig(
        high_threshold=float(config.get("high_threshold", 0.85)),
        low_threshold=float(config.get("low_threshold", 0.50)),
        floor_rate=float(config.get("rate_floor", 99.0)),
        ceiling_rate=float(config.get("rate_ceiling", 399.0)),
        max_daily_change_pct=float(config.get("max_change_pct", 0.10)),
        weekend_premium_min=float(config.get("weekend_premium_min", 1.0)),
        weekend_premium_max=float(config.get("weekend_premium_max", 1.15)),
    )

    # Historical base dataset.
    historical_required_cols = ["stay_date", "rooms_sold", "room_revenue"]
    if manual_rooms_available is None:
        historical_required_cols.insert(1, "rooms_available")

    historical_df = process_file(
        input_path,
        interactive=interactive,
        user_mapping=config.get("column_mapping"),
        required_columns=historical_required_cols,
    )

    # Optional manual override for total inventory.
    if manual_rooms_available is not None:
        historical_df["rooms_available"] = int(manual_rooms_available)

    _, _, as_of_date = current_month_context()

    historical_df, historical_validation = validate_data(
        historical_df,
        allow_overbooking=allow_overbooking,
        as_of_date=as_of_date,
        default_current_rate=default_current_rate,
    )
    if len(historical_df) == 0:
        issue_counts: dict[str, int] = {}
        for issue in historical_validation.issues:
            issue_type = str(issue.get("issue_type", "UNKNOWN"))
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        top_issues = ", ".join(
            f"{k}={v}" for k, v in sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        )
        if not top_issues:
            top_issues = "no issue details captured"
        raise ValueError(
            "No valid historical rows after validation. "
            f"Top issue types: {top_issues}. "
            "Check historical column mapping (stay_date, rooms_sold, room_revenue, rooms_available/occupancy_percent)."
        )

    check_data_quality(historical_df)

    # Future on-books dataset.
    future_df = _prepare_future_dataset(
        future_path=future_path,
        input_df=historical_df,
        interactive=interactive,
        mapping=config.get("future_column_mapping"),
        as_of_date=as_of_date,
        manual_rooms_available=manual_rooms_available,
    )
    if len(future_df) > 0:
        future_df, future_validation = validate_data(
            future_df,
            allow_overbooking=allow_overbooking,
            as_of_date=as_of_date,
            default_current_rate=default_current_rate,
        )
    else:
        future_validation = historical_validation

    # Save validation report with both blocks merged.
    historical_validation.issues.extend(future_validation.issues)
    historical_validation.total_rows = int(historical_validation.total_rows + future_validation.total_rows)
    historical_validation.valid_rows = int(historical_validation.valid_rows + future_validation.valid_rows)
    historical_validation.invalid_rows = int(historical_validation.invalid_rows + future_validation.invalid_rows)

    validation_path = output_dir / "validation_report.txt"
    save_validation_report(historical_validation, str(validation_path))

    historical_metrics = calculate_daily_metrics(historical_df)
    daily_metrics_path = output_dir / "daily_metrics.csv"
    export_metrics(historical_metrics, str(daily_metrics_path))

    cleaned_path = output_dir / "cleaned_data.csv"
    historical_df.to_csv(cleaned_path, index=False)

    events_df = load_events(events_path)

    # Forecast future demand.
    forecast_df = build_future_forecast(
        historical_df=historical_metrics,
        future_df=future_df,
        events_df=events_df,
        blend_on_books_floor=True,
    )
    forecast_path = output_dir / "forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    # Pace/event context.
    historical_dir = Path(__file__).parent / "data" / "historical"
    stly_df = load_historical_data(str(historical_dir))
    combined_for_pace = pd.concat([historical_df, future_df], ignore_index=True) if len(future_df) else historical_df.copy()
    pace_df = calculate_pace_analysis(combined_for_pace, stly_df)
    pace_df = apply_event_impacts(pace_df, events_df)

    future_context = future_df.merge(
        forecast_df[["stay_date", "forecast_rooms_sold", "forecast_occ", "base_demand", "on_books"]],
        on="stay_date",
        how="left",
    )
    future_context = future_context.merge(
        pace_df[["stay_date", "stly_occupancy", "pace_variance", "impact_level", "event_pct"]],
        on="stay_date",
        how="left",
    )
    future_context["forecast_occ"] = future_context["forecast_occ"].fillna(0.0)

    # Build a month-scoped on-books + pickup projection for dashboard metrics.
    # Derive the forecast month/year from the uploaded data itself, not from system date
    # or hardcoded offsets.  Priority: future report > historical report > system date.
    _future_dates = pd.to_datetime(future_context["stay_date"], errors="coerce").dropna() if len(future_context) else pd.Series(dtype="datetime64[ns]")
    _hist_dates = pd.to_datetime(historical_df["stay_date"], errors="coerce").dropna() if len(historical_df) else pd.Series(dtype="datetime64[ns]")

    if len(_future_dates):
        anchor_date = _future_dates.min()
    elif len(_hist_dates):
        anchor_date = _hist_dates.max()  # most recent historical date as proxy
    else:
        anchor_date = as_of_date

    target_year = int(anchor_date.year)
    target_month = int(anchor_date.month)

    # Derive the STLY year from the actual years present in the historical data files
    # for the same calendar month — no arithmetic year offset.
    stly_year: int | None = None
    if stly_df is not None and len(stly_df) > 0:
        _stly_dates = pd.to_datetime(stly_df["stay_date"], errors="coerce")
        _stly_years_for_month = (
            _stly_dates[_stly_dates.dt.month == target_month]
            .dt.year
            .dropna()
            .unique()
        )
        # Pick the most recent year in the historical files that is strictly before
        # target_year (i.e. the true prior year from the uploaded data).
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

    # Clip projected rates to a realistic band learned from historical ADR to avoid inflated outliers.
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

    # Revenue to date comes exclusively from the future on-books report.
    actual_revenue_to_date = float(
        pd.to_numeric(month_future.get("room_revenue", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    ) if len(month_future) else 0.0

    # Initialize STLY anchors for downstream forecast safeguards.
    stly_has_data = False
    stly_month_revenue = 0.0

    if len(month_future):
        positive_inventory = pd.to_numeric(month_future.get("rooms_available", pd.Series(dtype=float)), errors="coerce")
        positive_inventory = positive_inventory[positive_inventory > 0]
        # Fall back to historical report if future feed lacks inventory
        if len(positive_inventory) == 0:
            positive_inventory = pd.to_numeric(historical_df.get("rooms_available", pd.Series(dtype=float)), errors="coerce")
            positive_inventory = positive_inventory[positive_inventory > 0]
        derived_total_rooms = float(positive_inventory.median()) if len(positive_inventory) else 0.0
        days_in_month = monthrange(target_year, target_month)[1]
        monthly_room_capacity = max(0.0, derived_total_rooms * float(days_in_month))

        # Current on-books room nights come from the future on-books report for the
        # target month only.  The main --input file covers a different period and must
        # not inflate this figure.
        on_books_rooms_total = float(
            pd.to_numeric(month_future.get("rooms_sold", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        )

        if monthly_room_capacity <= 0 and on_books_rooms_total > 0:
            monthly_room_capacity = on_books_rooms_total

        on_books_occ_month = (
            on_books_rooms_total / monthly_room_capacity
            if monthly_room_capacity > 0
            else 0.0
        )

        # --- STLY (same-month prior year) anchor ---
        # Pull actuals for the matching calendar month using the year derived from the
        # historical data files — no hardcoded year offset.
        stly_month_occ = 0.0
        stly_month_adr = 0.0
        stly_month_revenue = 0.0
        stly_has_data = False
        if stly_df is not None and len(stly_df) > 0:
            stly_dates = pd.to_datetime(stly_df["stay_date"], errors="coerce")
            if stly_year is not None:
                stly_prior = stly_df[
                    (stly_dates.dt.month == target_month)
                    & (stly_dates.dt.year == stly_year)
                ].copy()
            else:
                stly_prior = pd.DataFrame()
            # If historical dir has no exact prior-year rows for this month, fall back
            # to any year present in the files for the same calendar month.
            if len(stly_prior) == 0:
                stly_prior = stly_df[stly_dates.dt.month == target_month].copy()
            if len(stly_prior) > 0:
                stly_occ_col = pd.to_numeric(stly_prior.get("stly_occupancy", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                stly_adr_col = pd.to_numeric(stly_prior.get("stly_adr", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                stly_rev_col = pd.to_numeric(stly_prior.get("stly_revenue", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                stly_rooms_col = pd.to_numeric(stly_prior.get("stly_rooms_sold", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                # Weighted average occupancy across the month
                stly_month_occ = float(stly_occ_col.mean())
                stly_month_adr = float(stly_adr_col[stly_adr_col > 0].mean()) if (stly_adr_col > 0).any() else fallback_future_rate
                stly_month_revenue = float(stly_rev_col.sum())
                if stly_month_revenue <= 0 and stly_month_adr > 0 and float(stly_rooms_col.sum()) > 0:
                    # Some PMS extracts carry revenue in non-standard columns; when
                    # STLY revenue is missing/zero, derive a usable month total from
                    # STLY room nights and ADR for anchoring.
                    stly_month_revenue = float(stly_rooms_col.sum()) * stly_month_adr
                stly_has_data = stly_month_occ > 0

        # --- STLY-anchored room-night forecast ---
        #
        # Formula:
        #   Forecast Rooms = Current OTB Rooms
        #                  + (STLY Remaining Pickup) * Pickup Pace Factor
        #
        # STLY Remaining Pickup = STLY Final Rooms - STLY OTB Rooms at same cutoff
        #
        # Pickup Pace Factor:
        #   = current_otb_rooms / stly_otb_rooms  (auto-derived from data)
        #   = 1.0 if pacing identically to last year
        #   > 1.0 if currently ahead of STLY pace  → expect more remaining pickup
        #   < 1.0 if currently behind STLY pace    → expect less remaining pickup
        #   Override via config key "pickup_pace_factor" (float).
        #
        # STLY Final Rooms = STLY OTB at cutoff * stly_close_factor
        #   stly_close_factor represents the ratio of final-close to OTB-snapshot
        #   for the same calendar month last year.
        #   Configure via "stly_close_factor" (default 1.15, meaning 15% pickup
        #   materialised after the OTB snapshot was taken).
        #
        # Worked example (stly_close_factor = 1.20):
        #   Current OTB Rooms       = 1,864
        #   STLY OTB at cutoff      = 1,864
        #   STLY Final Rooms        = 1,864 * 1.20 = 2,237
        #   STLY Remaining Pickup   = 2,237 − 1,864 = 373
        #   Pickup Pace Factor      = 1,864 / 1,864 = 1.00
        #   Forecast Rooms          = 1,864 + 373 * 1.00 = 2,237
        #   Forecast Revenue ≈ 2,237 * blended_ADR

        # Step 1 — STLY OTB room nights at same booking-window cutoff.
        # stly_rooms_sold is the per-day rooms-on-books from the historical file,
        # which IS an OTB snapshot captured at the same relative point last year.
        stly_otb_rooms = 0.0
        if stly_has_data and "stly_rooms_sold" in stly_prior.columns:
            stly_otb_rooms = float(
                pd.to_numeric(stly_prior["stly_rooms_sold"], errors="coerce").fillna(0.0).sum()
            )
        # Fallback: derive STLY OTB rooms from revenue ÷ ADR when rooms_sold unavailable.
        if stly_otb_rooms == 0.0 and stly_has_data and stly_month_adr > 0:
            stly_otb_rooms = stly_month_revenue / stly_month_adr

        # Step 2 — STLY Final Rooms (what the month eventually closed at).
        # We multiply STLY OTB by stly_close_factor because the historical files are
        # OTB snapshots, not final-close reports.  Set "stly_close_factor" in config
        # to match your property's observed close-rate for this calendar month.
        stly_close_factor = float(config.get("stly_close_factor", 1.15))
        if stly_has_data and stly_otb_rooms > 0:
            stly_final_rooms = stly_otb_rooms * stly_close_factor
            stly_remaining_pickup = max(0.0, stly_final_rooms - stly_otb_rooms)
        else:
            stly_final_rooms = 0.0
            stly_remaining_pickup = 0.0

        # Step 3 — Pickup Pace Factor.
        # Reflects how current OTB compares to where STLY was at the same point.
        # Auto-derived from the data; can be overridden via config.
        config_pace_override = config.get("pickup_pace_factor", None)
        if config_pace_override is not None:
            pickup_pace_factor = float(config_pace_override)
        elif stly_otb_rooms > 0:
            raw_ratio = on_books_rooms_total / stly_otb_rooms
            # Clip to a sane range to prevent extreme swings from outlier data.
            pickup_pace_factor = float(np.clip(raw_ratio, 0.5, 2.0))
        else:
            pickup_pace_factor = 1.0

        # Step 4 — Forecast room nights (room-night first, derive occupancy from that).
        forecast_rooms_total = on_books_rooms_total + stly_remaining_pickup * pickup_pace_factor

        # Guardrail: cannot physically exceed total room-night capacity for the month.
        if monthly_room_capacity > 0:
            forecast_rooms_total = min(forecast_rooms_total, monthly_room_capacity)

        # Never forecast below what is already on the books (no wash model).
        forecast_rooms_total = max(forecast_rooms_total, on_books_rooms_total)

        # Step 5 — Derive occupancy from forecast room nights.
        forecast_occ_month = (
            forecast_rooms_total / monthly_room_capacity
            if monthly_room_capacity > 0
            else 0.0
        )

        # Pickup room nights = incremental rooms above current OTB.
        pickup_rooms_total = max(0.0, forecast_rooms_total - on_books_rooms_total)

        # --- Forecast ADR for pickup rooms ---
        # Blend current on-books ADR (60 %) with STLY ADR (40 %) when available.
        # This reflects realistic year-over-year rate growth without overstating.
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
        on_books_occ_month = 0.0
        forecast_occ_month = 0.0
        forecast_revenue_remaining = 0.0

    # Keep OTB unchanged and add projected pickup on top.
    month_end_forecast = actual_revenue_to_date + max(0.0, forecast_revenue_remaining)

    # Revenue anchor: keep forecast close to prior-year OTB month-end revenue when
    # STLY data is available. This prevents trivial uplift over OTB in low-pace cases.
    # Default ratio 1.00 means "at least match STLY OTB month-end revenue".
    if stly_has_data and stly_month_revenue > 0:
        stly_revenue_floor_ratio = float(config.get("stly_revenue_floor_ratio", 1.00))
        stly_revenue_floor_ratio = max(0.0, stly_revenue_floor_ratio)
        stly_revenue_floor = stly_month_revenue * stly_revenue_floor_ratio
        month_end_forecast = max(month_end_forecast, stly_revenue_floor)

    # Reasonableness guardrail: do not exceed a high-side revenue capacity envelope
    # unless explicitly overridden.
    if not bool(config.get("allow_revenue_over_capacity", False)) and monthly_room_capacity > 0:
        revenue_capacity_ceiling = monthly_room_capacity * rate_ceiling_hist
        month_end_forecast = min(month_end_forecast, revenue_capacity_ceiling)

    # Keep component metrics internally consistent after any anchoring/guardrails.
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
        current_year, current_month, _ = current_month_context()
        daily_budget_df, _ = prepare_monthly_budget_targets(
            budget_path=budget_path,
            current_year=current_year,
            current_month=current_month,
            historical_df=historical_metrics,
            daily_distribution_method=str(config.get("budget_distribution", "dow_weighted")),
        )
        forecast_remaining_for_budget = future_context[["stay_date", "rooms_available", "forecast_rooms_sold"]].copy()
        forecast_remaining_for_budget["forecast_revenue"] = (
            forecast_remaining_for_budget["forecast_rooms_sold"] * future_context["current_rate"].fillna(future_context["current_rate"].median())
        )
        month_actual_for_budget = historical_metrics[["stay_date", "room_revenue", "rooms_available", "rooms_sold"]].copy()
        budget_summary = calculate_budget_progress(
            month_actual_df=month_actual_for_budget,
            forecast_remaining_df=forecast_remaining_for_budget,
            daily_budget_df=daily_budget_df,
            as_of_date=as_of_date,
        )

    # Baseline policy output for comparison.
    baseline_rules_df = generate_rate_recommendations(
        pace_df.rename(columns={"current_adr": "current_adr"}),
        pricing_config,
    ) if len(pace_df) else pd.DataFrame(columns=["stay_date", "recommended_adr"])

    # New policy (elasticity simulation).
    recommendations_df, simulation_df = simulate_elasticity_pricing(
        future_context,
        config=pricing_config,
        elasticity=elasticity,
        budget_gap=float(budget_summary.get("remaining_budget", 0.0)),
        required_adr_remaining=float(budget_summary.get("required_adr_remaining", 0.0)),
    )

    recommendations_path = output_dir / "rate_recommendations.csv"
    recommendations_df.to_csv(recommendations_path, index=False)

    top_raise_df, top_rescue_df, top_monitor_df, priority_full_df = build_priority_lists(
        recommendations_df,
        budget_gap=float(budget_summary.get("remaining_budget", 0.0)),
        target_occ=float(config.get("target_occ", 0.80)),
    )

    top_raise_path = output_dir / "top_raise_opportunities.csv"
    top_rescue_path = output_dir / "top_rescue_dates.csv"
    top_monitor_path = output_dir / "top_monitor_dates.csv"
    top_raise_df.to_csv(top_raise_path, index=False)
    top_rescue_df.to_csv(top_rescue_path, index=False)
    top_monitor_df.to_csv(top_monitor_path, index=False)

    # Forecast quality metrics from holdout backtest.
    model_df = prepare_forecast_frame(daily_df=historical_metrics, events_df=events_df, stly_df=stly_df)
    backtest_df = evaluate_backtest(model_df=model_df, as_of_date=as_of_date)
    forecast_metrics = calculate_forecast_metrics(
        actual=backtest_df.get("actual_rooms_sold", pd.Series(dtype=float)),
        predicted=backtest_df.get("baseline_rooms_sold", pd.Series(dtype=float)),
    )

    baseline_vs_new_df = _build_baseline_vs_new_policy(historical_metrics, baseline_rules_df, elasticity=elasticity)
    baseline_vs_new_path = output_dir / "baseline_vs_new_policy.csv"
    baseline_vs_new_df.to_csv(baseline_vs_new_path, index=False)

    projected_uplift = float(baseline_vs_new_df.get("expected_revenue_uplift", pd.Series(dtype=float)).sum())
    evaluation_metrics_df = build_policy_evaluation_metrics(forecast_metrics, projected_uplift_vs_baseline=projected_uplift)
    evaluation_metrics_path = output_dir / "evaluation_metrics.csv"
    evaluation_metrics_df.to_csv(evaluation_metrics_path, index=False)

    # Build full-range forecast vs actual chart data (entire historical timeframe, not only holdout slice).
    full_baseline_pred = baseline_forecast(train_df=model_df, target_df=model_df)
    full_enhanced_pred = enhanced_forecast(train_df=model_df, target_df=model_df)
    full_forecast_vs_actual_df = pd.DataFrame(
        {
            "stay_date": model_df["stay_date"],
            "actual_rooms_sold": model_df["rooms_sold"],
            "baseline_rooms_sold": full_baseline_pred["forecast_rooms_sold"],
            "enhanced_rooms_sold": full_enhanced_pred["forecast_rooms_sold"],
            "actual_revenue": model_df["room_revenue"],
            "baseline_revenue": full_baseline_pred["forecast_revenue"],
            "enhanced_revenue": full_enhanced_pred["forecast_revenue"],
        }
    ).sort_values("stay_date").reset_index(drop=True)

    forecast_vs_actual_csv_path = output_dir / "forecast_vs_actual.csv"
    full_forecast_vs_actual_df.to_csv(forecast_vs_actual_csv_path, index=False)

    # Charts required by spec.
    plot_current_vs_recommended_rate(recommendations_df, str(output_dir / "current_vs_recommended_rate.png"))
    plot_expected_revenue_uplift(recommendations_df, str(output_dir / "expected_revenue_uplift.png"))
    plot_priority_score_by_date(priority_full_df, str(output_dir / "priority_score_by_date.png"))
    plot_forecast_vs_actual(full_forecast_vs_actual_df, str(output_dir / "forecast_vs_actual.png"))

    output_paths = {
        "output_dir": str(output_dir),
        "cleaned_data": str(cleaned_path),
        "daily_metrics": str(daily_metrics_path),
        "validation_report": str(validation_path),
        "forecast": str(forecast_path),
        "rate_recommendations": str(recommendations_path),
        "top_raise_opportunities": str(top_raise_path),
        "top_rescue_dates": str(top_rescue_path),
        "top_monitor_dates": str(top_monitor_path),
        "evaluation_metrics": str(evaluation_metrics_path),
        "baseline_vs_new_policy": str(baseline_vs_new_path),
        "forecast_vs_actual": str(forecast_vs_actual_csv_path),
    }

    summary = {
        "budget_summary": budget_summary,
        "forecast_metrics": forecast_metrics,
        "projected_uplift_vs_baseline": projected_uplift,
        "heavy_need_days": int(len(top_raise_df)),
    }

    return output_paths, summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hotel RMS - Forward-looking pricing simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Historical PMS report (CSV/XLSX)")
    parser.add_argument("--future", default=None, help="Future on-the-books report (CSV/XLSX)")
    parser.add_argument("--events", default=None, help="Optional events CSV")
    parser.add_argument("--budget", default=None, help="Optional budget file")
    parser.add_argument("--output", "-o", default="outputs", help="Output base directory")

    parser.add_argument("--rate_floor", type=float, default=99.0)
    parser.add_argument("--rate_ceiling", type=float, default=399.0)
    parser.add_argument("--max_change_pct", type=float, default=0.10)
    parser.add_argument("--elasticity", type=float, default=1.2)
    parser.add_argument("--manual_rooms_available", type=int, default=None, help="Optional manual override for rooms_available across all rows")

    parser.add_argument("--allow-overbooking", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")

    return parser.parse_args()


def _print_summary(summary: dict) -> None:
    budget = summary.get("budget_summary", {})
    forecast = summary.get("forecast_metrics", {})
    print("\n" + "=" * 60)
    print("RMS SUMMARY")
    print("=" * 60)
    print(f"Month-end forecast: ${budget.get('month_end_forecast', 0.0):,.2f}")
    print(f"Budget variance: ${budget.get('variance_to_budget_abs', 0.0):,.2f} ({budget.get('variance_to_budget_pct', 0.0):.2f}%)")
    print(f"Required ADR remaining: ${budget.get('required_adr_remaining', 0.0):,.2f}")
    print(f"Forecast MAE: {forecast.get('mae', np.nan):.3f}")
    print(f"Forecast MAPE: {forecast.get('mape', np.nan):.2f}%")
    print(f"Projected uplift vs baseline: ${summary.get('projected_uplift_vs_baseline', 0.0):,.2f}")
    print("=" * 60)


def main() -> int:
    args = parse_arguments()
    try:
        output_paths, summary = run_pipeline(
            input_path=args.input,
            future_path=args.future,
            events_path=args.events,
            budget_path=args.budget,
            config={
                "output_dir": args.output,
                "rate_floor": args.rate_floor,
                "rate_ceiling": args.rate_ceiling,
                "max_change_pct": args.max_change_pct,
                "elasticity": args.elasticity,
                "manual_rooms_available": args.manual_rooms_available,
                "allow_overbooking": args.allow_overbooking,
                "interactive": not args.no_interactive,
            },
        )
        _print_summary(summary)
        print(f"All files saved to: {Path(output_paths['output_dir']).absolute()}")
        return 0
    except FileNotFoundError as exc:
        print_error(f"File not found: {exc}")
        return 1
    except ValueError as exc:
        print_error(f"Validation error: {exc}")
        return 1
    except Exception as exc:
        print_error(f"Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
