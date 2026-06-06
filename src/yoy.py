"""Year-over-year comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


YOY_REQUIRED_FIELDS = ["stay_date", "rooms_available", "rooms_sold", "room_revenue"]


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _coerce_occupancy_decimal(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    if out.notna().any() and float(out.max()) > 1.5:
        out = out / 100.0
    return out


def _safe_pct_variance(current: pd.Series, prior: pd.Series) -> pd.Series:
    current_num = pd.to_numeric(current, errors="coerce")
    prior_num = pd.to_numeric(prior, errors="coerce")
    valid_mask = current_num.notna() & prior_num.notna() & (prior_num != 0)
    out = pd.Series(np.nan, index=current.index, dtype=float)
    out.loc[valid_mask] = ((current_num.loc[valid_mask] - prior_num.loc[valid_mask]) / prior_num.loc[valid_mask]) * 100.0
    return out


def _aggregate_current_daily(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    current["stay_date"] = pd.to_datetime(current.get("stay_date"), errors="coerce")
    current = current.dropna(subset=["stay_date"]).copy()

    if len(current) == 0:
        return pd.DataFrame(columns=["stay_date", "calendar_date"])

    current["rooms_available"] = _safe_numeric(current, "rooms_available")
    current["rooms_sold"] = _safe_numeric(current, "rooms_sold")
    current["room_revenue"] = _safe_numeric(current, "room_revenue")

    grouped = (
        current.groupby("stay_date", as_index=False)
        .agg(
            rooms_available=("rooms_available", "sum"),
            rooms_sold=("rooms_sold", "sum"),
            room_revenue=("room_revenue", "sum"),
        )
        .sort_values("stay_date")
        .reset_index(drop=True)
    )

    grouped["current_occupancy"] = np.where(
        grouped["rooms_available"] > 0,
        grouped["rooms_sold"] / grouped["rooms_available"],
        np.nan,
    )
    grouped["current_occupancy_pct"] = grouped["current_occupancy"] * 100.0
    grouped["current_adr"] = np.where(
        grouped["rooms_sold"] > 0,
        grouped["room_revenue"] / grouped["rooms_sold"],
        np.nan,
    )
    grouped["current_revpar"] = np.where(
        grouped["rooms_available"] > 0,
        grouped["room_revenue"] / grouped["rooms_available"],
        np.nan,
    )

    grouped = grouped.rename(
        columns={
            "rooms_available": "current_rooms_available",
            "rooms_sold": "current_rooms_sold",
            "room_revenue": "current_room_revenue",
        }
    )
    grouped["calendar_date"] = grouped["stay_date"].dt.strftime("%m-%d")
    return grouped


def _normalize_prior_columns(prior_df: pd.DataFrame) -> pd.DataFrame:
    prior = prior_df.copy()
    prior["stay_date"] = pd.to_datetime(prior.get("stay_date"), errors="coerce")
    prior = prior.dropna(subset=["stay_date"]).copy()

    if len(prior) == 0:
        return pd.DataFrame(columns=["stay_date", "calendar_date"])

    rooms_available = _safe_numeric(prior, "rooms_available")
    rooms_sold = _safe_numeric(prior, "rooms_sold")
    room_revenue = _safe_numeric(prior, "room_revenue")

    if "stly_rooms_sold" in prior.columns:
        stly_rooms = _safe_numeric(prior, "stly_rooms_sold")
        rooms_sold = rooms_sold.combine_first(stly_rooms)

    if "stly_revenue" in prior.columns:
        stly_revenue = _safe_numeric(prior, "stly_revenue")
        room_revenue = room_revenue.combine_first(stly_revenue)

    occupancy_decimal = pd.Series(np.nan, index=prior.index, dtype=float)
    if "occupancy" in prior.columns:
        occupancy_decimal = _coerce_occupancy_decimal(prior["occupancy"])
    if "occupancy_percent" in prior.columns:
        occupancy_decimal = occupancy_decimal.combine_first(_coerce_occupancy_decimal(prior["occupancy_percent"]))
    if "stly_occupancy" in prior.columns:
        occupancy_decimal = occupancy_decimal.combine_first(_coerce_occupancy_decimal(prior["stly_occupancy"]))

    adr = _safe_numeric(prior, "adr")
    if "stly_adr" in prior.columns:
        adr = adr.combine_first(_safe_numeric(prior, "stly_adr"))

    if rooms_available.isna().all() and occupancy_decimal.notna().any() and rooms_sold.notna().any():
        rooms_available = np.where(occupancy_decimal > 0, rooms_sold / occupancy_decimal, np.nan)
        rooms_available = pd.Series(rooms_available, index=prior.index, dtype=float)

    prior_norm = pd.DataFrame(
        {
            "stay_date": prior["stay_date"],
            "prior_year_rooms_available": rooms_available,
            "prior_year_rooms_sold": rooms_sold,
            "prior_year_room_revenue": room_revenue,
            "prior_year_occupancy": occupancy_decimal,
            "prior_year_adr": adr,
        }
    )

    grouped = (
        prior_norm.groupby("stay_date", as_index=False)
        .agg(
            prior_year_rooms_available=("prior_year_rooms_available", lambda s: s.sum(min_count=1)),
            prior_year_rooms_sold=("prior_year_rooms_sold", lambda s: s.sum(min_count=1)),
            prior_year_room_revenue=("prior_year_room_revenue", lambda s: s.sum(min_count=1)),
            prior_year_occupancy=("prior_year_occupancy", "mean"),
            prior_year_adr=("prior_year_adr", "mean"),
        )
        .sort_values("stay_date")
        .reset_index(drop=True)
    )

    derived_occ = np.where(
        grouped["prior_year_rooms_available"] > 0,
        grouped["prior_year_rooms_sold"] / grouped["prior_year_rooms_available"],
        np.nan,
    )
    grouped["prior_year_occupancy"] = pd.Series(derived_occ, index=grouped.index).combine_first(grouped["prior_year_occupancy"])
    grouped["prior_year_occupancy_pct"] = grouped["prior_year_occupancy"] * 100.0

    derived_adr = np.where(
        grouped["prior_year_rooms_sold"] > 0,
        grouped["prior_year_room_revenue"] / grouped["prior_year_rooms_sold"],
        np.nan,
    )
    grouped["prior_year_adr"] = pd.Series(derived_adr, index=grouped.index).combine_first(grouped["prior_year_adr"])

    grouped["prior_year_revpar"] = np.where(
        grouped["prior_year_rooms_available"] > 0,
        grouped["prior_year_room_revenue"] / grouped["prior_year_rooms_available"],
        np.nan,
    )
    grouped["calendar_date"] = grouped["stay_date"].dt.strftime("%m-%d")
    return grouped


def build_yoy_comparison(current_df: pd.DataFrame, historical_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build a row-level YoY comparison table.

    Aligns current rows to prior year using exact prior-year date first, then falls
    back to calendar-date (MM-DD) matching. Computes absolute and percentage
    variances for occupancy, ADR, RevPAR, rooms sold, and revenue.
    """
    if current_df is None or len(current_df) == 0:
        return pd.DataFrame()

    current_daily = _aggregate_current_daily(current_df)
    if len(current_daily) == 0:
        return pd.DataFrame()

    if historical_df is None:
        historical_df = pd.DataFrame()
    prior_daily = _normalize_prior_columns(historical_df)

    result = current_daily.copy()
    result["prior_year_target_date"] = result["stay_date"] - pd.DateOffset(years=1)

    if len(prior_daily) > 0:
        exact_prior = prior_daily.copy()
        exact_prior["prior_year_target_date"] = exact_prior["stay_date"]
        exact_prior = exact_prior.rename(
            columns={
                "stay_date": "prior_year_stay_date_exact",
                "prior_year_rooms_available": "prior_year_rooms_available_exact",
                "prior_year_rooms_sold": "prior_year_rooms_sold_exact",
                "prior_year_room_revenue": "prior_year_room_revenue_exact",
                "prior_year_occupancy": "prior_year_occupancy_exact",
                "prior_year_occupancy_pct": "prior_year_occupancy_pct_exact",
                "prior_year_adr": "prior_year_adr_exact",
                "prior_year_revpar": "prior_year_revpar_exact",
            }
        )
        exact_cols = [
            "prior_year_target_date",
            "prior_year_stay_date_exact",
            "prior_year_rooms_available_exact",
            "prior_year_rooms_sold_exact",
            "prior_year_room_revenue_exact",
            "prior_year_occupancy_exact",
            "prior_year_occupancy_pct_exact",
            "prior_year_adr_exact",
            "prior_year_revpar_exact",
        ]
        exact_prior = exact_prior[exact_cols]
        result = result.merge(exact_prior, on="prior_year_target_date", how="left")

        calendar_candidates = result[["stay_date", "calendar_date"]].merge(
            prior_daily,
            on="calendar_date",
            how="left",
            suffixes=("", "_candidate"),
        )
        calendar_candidates = calendar_candidates[
            calendar_candidates["stay_date_candidate"].notna()
            & (calendar_candidates["stay_date_candidate"] < calendar_candidates["stay_date"])
        ].copy()

        if len(calendar_candidates) > 0:
            calendar_candidates = calendar_candidates.sort_values(["stay_date", "stay_date_candidate"])
            calendar_candidates = calendar_candidates.drop_duplicates(subset=["stay_date"], keep="last")
            calendar_candidates = calendar_candidates.rename(
                columns={
                    "stay_date_candidate": "prior_year_stay_date_calendar",
                    "prior_year_rooms_available": "prior_year_rooms_available_calendar",
                    "prior_year_rooms_sold": "prior_year_rooms_sold_calendar",
                    "prior_year_room_revenue": "prior_year_room_revenue_calendar",
                    "prior_year_occupancy": "prior_year_occupancy_calendar",
                    "prior_year_occupancy_pct": "prior_year_occupancy_pct_calendar",
                    "prior_year_adr": "prior_year_adr_calendar",
                    "prior_year_revpar": "prior_year_revpar_calendar",
                }
            )
            calendar_cols = [
                "stay_date",
                "prior_year_stay_date_calendar",
                "prior_year_rooms_available_calendar",
                "prior_year_rooms_sold_calendar",
                "prior_year_room_revenue_calendar",
                "prior_year_occupancy_calendar",
                "prior_year_occupancy_pct_calendar",
                "prior_year_adr_calendar",
                "prior_year_revpar_calendar",
            ]
            result = result.merge(calendar_candidates[calendar_cols], on="stay_date", how="left")
        else:
            result["prior_year_stay_date_calendar"] = pd.NaT
            result["prior_year_rooms_available_calendar"] = np.nan
            result["prior_year_rooms_sold_calendar"] = np.nan
            result["prior_year_room_revenue_calendar"] = np.nan
            result["prior_year_occupancy_calendar"] = np.nan
            result["prior_year_occupancy_pct_calendar"] = np.nan
            result["prior_year_adr_calendar"] = np.nan
            result["prior_year_revpar_calendar"] = np.nan
    else:
        result["prior_year_stay_date_exact"] = pd.NaT
        result["prior_year_rooms_available_exact"] = np.nan
        result["prior_year_rooms_sold_exact"] = np.nan
        result["prior_year_room_revenue_exact"] = np.nan
        result["prior_year_occupancy_exact"] = np.nan
        result["prior_year_occupancy_pct_exact"] = np.nan
        result["prior_year_adr_exact"] = np.nan
        result["prior_year_revpar_exact"] = np.nan
        result["prior_year_stay_date_calendar"] = pd.NaT
        result["prior_year_rooms_available_calendar"] = np.nan
        result["prior_year_rooms_sold_calendar"] = np.nan
        result["prior_year_room_revenue_calendar"] = np.nan
        result["prior_year_occupancy_calendar"] = np.nan
        result["prior_year_occupancy_pct_calendar"] = np.nan
        result["prior_year_adr_calendar"] = np.nan
        result["prior_year_revpar_calendar"] = np.nan

    exact_match = result["prior_year_stay_date_exact"].notna()
    calendar_match = result["prior_year_stay_date_calendar"].notna()

    result["prior_year_stay_date"] = result["prior_year_stay_date_exact"].combine_first(result["prior_year_stay_date_calendar"])
    result["prior_year_rooms_available"] = result["prior_year_rooms_available_exact"].combine_first(result["prior_year_rooms_available_calendar"])
    result["prior_year_rooms_sold"] = result["prior_year_rooms_sold_exact"].combine_first(result["prior_year_rooms_sold_calendar"])
    result["prior_year_room_revenue"] = result["prior_year_room_revenue_exact"].combine_first(result["prior_year_room_revenue_calendar"])
    result["prior_year_occupancy"] = result["prior_year_occupancy_exact"].combine_first(result["prior_year_occupancy_calendar"])
    result["prior_year_occupancy_pct"] = result["prior_year_occupancy_pct_exact"].combine_first(result["prior_year_occupancy_pct_calendar"])
    result["prior_year_adr"] = result["prior_year_adr_exact"].combine_first(result["prior_year_adr_calendar"])
    result["prior_year_revpar"] = result["prior_year_revpar_exact"].combine_first(result["prior_year_revpar_calendar"])

    result["yoy_alignment_method"] = np.where(
        exact_match,
        "exact_prior_date",
        np.where(calendar_match, "calendar_date_fallback", "no_prior_match"),
    )

    result["current_required_fields_present"] = (
        result[["stay_date", "current_rooms_available", "current_rooms_sold", "current_room_revenue"]].notna().all(axis=1)
    )
    result["prior_year_required_fields_present"] = (
        result[["prior_year_stay_date", "prior_year_rooms_available", "prior_year_rooms_sold", "prior_year_room_revenue"]].notna().all(axis=1)
    )

    result["prior_year_data_available"] = result[
        [
            "prior_year_occupancy_pct",
            "prior_year_adr",
            "prior_year_revpar",
            "prior_year_rooms_sold",
            "prior_year_room_revenue",
        ]
    ].notna().any(axis=1)

    result["yoy_status"] = np.where(
        ~result["prior_year_data_available"],
        "PRIOR_YEAR_UNAVAILABLE",
        np.where(result["prior_year_required_fields_present"], "OK", "PRIOR_YEAR_INCOMPLETE"),
    )

    result["occupancy_variance"] = result["current_occupancy_pct"] - result["prior_year_occupancy_pct"]
    result["occupancy_variance_pct"] = _safe_pct_variance(result["current_occupancy_pct"], result["prior_year_occupancy_pct"])

    result["adr_variance"] = result["current_adr"] - result["prior_year_adr"]
    result["adr_variance_pct"] = _safe_pct_variance(result["current_adr"], result["prior_year_adr"])

    result["revpar_variance"] = result["current_revpar"] - result["prior_year_revpar"]
    result["revpar_variance_pct"] = _safe_pct_variance(result["current_revpar"], result["prior_year_revpar"])

    result["rooms_sold_variance"] = result["current_rooms_sold"] - result["prior_year_rooms_sold"]
    result["rooms_sold_variance_pct"] = _safe_pct_variance(result["current_rooms_sold"], result["prior_year_rooms_sold"])

    result["revenue_variance"] = result["current_room_revenue"] - result["prior_year_room_revenue"]
    result["revenue_variance_pct"] = _safe_pct_variance(result["current_room_revenue"], result["prior_year_room_revenue"])

    # Legacy aliases used by the current UI panel and prior reports.
    result["Current OCC %"] = result["current_occupancy_pct"]
    result["STLY OCC %"] = result["prior_year_occupancy_pct"]
    result["OCC Var %"] = result["occupancy_variance"]
    result["Current ADR"] = result["current_adr"]
    result["STLY ADR"] = result["prior_year_adr"]
    result["ADR Var %"] = result["adr_variance_pct"]
    result["Current Revenue"] = result["current_room_revenue"]
    result["STLY Revenue"] = result["prior_year_room_revenue"]
    result["Revenue Var %"] = result["revenue_variance_pct"]

    ordered_cols = [
        "stay_date",
        "prior_year_target_date",
        "prior_year_stay_date",
        "calendar_date",
        "yoy_alignment_method",
        "yoy_status",
        "current_required_fields_present",
        "prior_year_required_fields_present",
        "prior_year_data_available",
        "current_rooms_available",
        "current_rooms_sold",
        "current_room_revenue",
        "current_occupancy",
        "current_occupancy_pct",
        "current_adr",
        "current_revpar",
        "prior_year_rooms_available",
        "prior_year_rooms_sold",
        "prior_year_room_revenue",
        "prior_year_occupancy",
        "prior_year_occupancy_pct",
        "prior_year_adr",
        "prior_year_revpar",
        "occupancy_variance",
        "occupancy_variance_pct",
        "adr_variance",
        "adr_variance_pct",
        "revpar_variance",
        "revpar_variance_pct",
        "rooms_sold_variance",
        "rooms_sold_variance_pct",
        "revenue_variance",
        "revenue_variance_pct",
        "Current OCC %",
        "STLY OCC %",
        "OCC Var %",
        "Current ADR",
        "STLY ADR",
        "ADR Var %",
        "Current Revenue",
        "STLY Revenue",
        "Revenue Var %",
    ]
    ordered_cols = [c for c in ordered_cols if c in result.columns]
    return result[ordered_cols].sort_values("stay_date").reset_index(drop=True)


def summarize_yoy(yoy_df: pd.DataFrame) -> dict:
    """Calculate summary statistics for YoY comparison output."""
    if yoy_df is None or len(yoy_df) == 0:
        return {}

    current_occ_col = "current_occupancy_pct" if "current_occupancy_pct" in yoy_df.columns else "Current OCC %"
    prior_occ_col = "prior_year_occupancy_pct" if "prior_year_occupancy_pct" in yoy_df.columns else "STLY OCC %"
    current_adr_col = "current_adr" if "current_adr" in yoy_df.columns else "Current ADR"
    prior_adr_col = "prior_year_adr" if "prior_year_adr" in yoy_df.columns else "STLY ADR"
    current_rev_col = "current_room_revenue" if "current_room_revenue" in yoy_df.columns else "Current Revenue"
    prior_rev_col = "prior_year_room_revenue" if "prior_year_room_revenue" in yoy_df.columns else "STLY Revenue"

    current_occ = pd.to_numeric(yoy_df.get(current_occ_col, pd.Series(dtype=float)), errors="coerce")
    prior_occ = pd.to_numeric(yoy_df.get(prior_occ_col, pd.Series(dtype=float)), errors="coerce")
    current_adr = pd.to_numeric(yoy_df.get(current_adr_col, pd.Series(dtype=float)), errors="coerce")
    prior_adr = pd.to_numeric(yoy_df.get(prior_adr_col, pd.Series(dtype=float)), errors="coerce")
    current_rev = pd.to_numeric(yoy_df.get(current_rev_col, pd.Series(dtype=float)), errors="coerce")
    prior_rev = pd.to_numeric(yoy_df.get(prior_rev_col, pd.Series(dtype=float)), errors="coerce")

    matched_rows = int((yoy_df.get("yoy_status", pd.Series(dtype=str)) == "OK").sum())
    missing_rows = int((yoy_df.get("yoy_status", pd.Series(dtype=str)) == "PRIOR_YEAR_UNAVAILABLE").sum())
    incomplete_rows = int((yoy_df.get("yoy_status", pd.Series(dtype=str)) == "PRIOR_YEAR_INCOMPLETE").sum())
    has_comparable_data = matched_rows > 0

    summary = {
        "avg_current_occupancy_pct": float(current_occ.dropna().mean()) if current_occ.notna().any() else 0.0,
        "avg_stly_occupancy_pct": float(prior_occ.dropna().mean()) if has_comparable_data and prior_occ.notna().any() else np.nan,
        "avg_current_adr": float(current_adr.dropna().mean()) if current_adr.notna().any() else 0.0,
        "avg_stly_adr": float(prior_adr.dropna().mean()) if has_comparable_data and prior_adr.notna().any() else np.nan,
        "total_current_revenue": float(current_rev.fillna(0.0).sum()),
        "total_stly_revenue": float(prior_rev.fillna(0.0).sum()) if has_comparable_data and prior_rev.notna().any() else np.nan,
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "incomplete_rows": incomplete_rows,
        "has_comparable_data": has_comparable_data,
    }

    summary["occupancy_change_pct"] = (
        float(summary["avg_current_occupancy_pct"] - summary["avg_stly_occupancy_pct"])
        if pd.notna(summary["avg_stly_occupancy_pct"])
        else np.nan
    )
    summary["adr_change_pct"] = (
        float(((summary["avg_current_adr"] - summary["avg_stly_adr"]) / summary["avg_stly_adr"]) * 100.0)
        if pd.notna(summary["avg_stly_adr"]) and summary["avg_stly_adr"] != 0.0
        else np.nan
    )
    summary["revenue_change_pct"] = (
        float(((summary["total_current_revenue"] - summary["total_stly_revenue"]) / summary["total_stly_revenue"]) * 100.0)
        if pd.notna(summary["total_stly_revenue"]) and summary["total_stly_revenue"] != 0.0
        else np.nan
    )
    return summary
