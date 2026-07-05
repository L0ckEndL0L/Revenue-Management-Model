"""Intraday comp-set update replay for tailored rate recommendations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.tailored import build_tailored_recommendations, update_daily_median_rates, validate_tailored_settings


INTRADAY_UPDATE_COLUMNS = [
    "stay_date",
    "update_timestamp",
    "manual_daily_median_rate",
    "reason_summary",
]

INTRADAY_RESULT_COLUMNS = [
    "update_sequence",
    "stay_date",
    "update_timestamp",
    "old_rate",
    "new_rate",
    "absolute_change",
    "percent_change",
    "updated_median_rate",
    "median_rate_source",
    "model_status",
    "reason_summary",
]


def _normalize_update_rows(updates_df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Clean intraday update rows and return user-facing validation messages."""
    if updates_df is None or len(updates_df) == 0:
        return pd.DataFrame(columns=INTRADAY_UPDATE_COLUMNS), []

    df = updates_df.copy()
    errors: list[str] = []
    missing = [column for column in ["stay_date", "update_timestamp"] if column not in df.columns]
    if missing:
        return pd.DataFrame(columns=INTRADAY_UPDATE_COLUMNS), [f"intraday updates missing required columns: {', '.join(missing)}"]

    if "manual_daily_median_rate" not in df.columns:
        if "median_rate" in df.columns:
            df["manual_daily_median_rate"] = df["median_rate"]
        elif "rate" in df.columns:
            df["manual_daily_median_rate"] = df["rate"]
        else:
            return pd.DataFrame(columns=INTRADAY_UPDATE_COLUMNS), ["intraday updates need manual_daily_median_rate, median_rate, or rate"]

    if "reason_summary" not in df.columns:
        df["reason_summary"] = ""

    df["stay_date"] = pd.to_datetime(df["stay_date"], errors="coerce")
    df["update_timestamp"] = pd.to_datetime(df["update_timestamp"], errors="coerce")
    df["manual_daily_median_rate"] = pd.to_numeric(df["manual_daily_median_rate"], errors="coerce")
    df["reason_summary"] = df["reason_summary"].fillna("").astype(str)

    invalid_dates = int(df["stay_date"].isna().sum())
    invalid_timestamps = int(df["update_timestamp"].isna().sum())
    invalid_rates = int((df["manual_daily_median_rate"].isna() | (df["manual_daily_median_rate"] <= 0)).sum())
    if invalid_dates:
        errors.append(f"{invalid_dates} intraday update row(s) have invalid stay_date")
    if invalid_timestamps:
        errors.append(f"{invalid_timestamps} intraday update row(s) have invalid update_timestamp")
    if invalid_rates:
        errors.append(f"{invalid_rates} intraday update row(s) have missing or non-positive rates")

    df = df.dropna(subset=["stay_date", "update_timestamp", "manual_daily_median_rate"])
    df = df[df["manual_daily_median_rate"] > 0].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=INTRADAY_UPDATE_COLUMNS), errors

    df["stay_date"] = df["stay_date"].dt.normalize()
    df = df.sort_values(["stay_date", "update_timestamp"]).reset_index(drop=True)
    return df[INTRADAY_UPDATE_COLUMNS], errors


def validate_intraday_updates(updates_df: pd.DataFrame | None) -> dict[str, Any]:
    """Validate whether an intraday update table can drive recommendation replay."""
    cleaned, errors = _normalize_update_rows(updates_df)
    duplicate_slots = 0
    if len(cleaned) > 0:
        duplicate_slots = int(cleaned.duplicated(subset=["stay_date", "update_timestamp"]).sum())
        if duplicate_slots:
            errors.append(f"{duplicate_slots} intraday update row(s) share the same stay_date and timestamp")

    return {
        "available": updates_df is not None and len(updates_df) > 0,
        "valid_rows": int(len(cleaned)),
        "errors": errors,
        "is_valid": len(errors) == 0 and len(cleaned) > 0,
    }


def _rate_for_date(results_df: pd.DataFrame, stay_date: pd.Timestamp) -> pd.Series | None:
    if len(results_df) == 0 or "stay_date" not in results_df.columns:
        return None
    dates = pd.to_datetime(results_df["stay_date"], errors="coerce").dt.normalize()
    match = results_df.loc[dates == stay_date.normalize()]
    if len(match) == 0:
        return None
    return match.iloc[0]


def process_intraday_updates(
    future_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    settings: dict[str, Any] | None,
    updates_df: pd.DataFrame | None,
    comp_set_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Replay same-day comp-set updates in timestamp order and track recommendation changes."""
    updates, validation_errors = _normalize_update_rows(updates_df)
    if len(updates) == 0:
        return pd.DataFrame(columns=INTRADAY_RESULT_COLUMNS), validation_errors

    active_settings, settings_errors = validate_tailored_settings(settings)
    if settings_errors:
        return pd.DataFrame(columns=INTRADAY_RESULT_COLUMNS), validation_errors + settings_errors

    current_results = build_tailored_recommendations(
        future_df,
        baseline_df,
        active_settings,
        comp_set_df=comp_set_df,
    )

    rows: list[dict[str, Any]] = []
    daily_rows = list(active_settings.get("daily_median_rates", []))
    for sequence, update in enumerate(updates.itertuples(index=False), start=1):
        stay_date = pd.Timestamp(update.stay_date).normalize()
        previous = _rate_for_date(current_results, stay_date)
        old_rate = float(previous["tailored_recommendation"]) if previous is not None and pd.notna(previous.get("tailored_recommendation")) else np.nan

        replacement_by_date = {row.get("stay_date"): dict(row) for row in daily_rows if isinstance(row, dict)}
        replacement_by_date[stay_date.date().isoformat()] = {
            "stay_date": stay_date.date().isoformat(),
            "manual_daily_median_rate": float(update.manual_daily_median_rate),
            "last_median_update_timestamp": pd.Timestamp(update.update_timestamp).isoformat(),
        }
        daily_rows = list(replacement_by_date.values())
        active_settings = update_daily_median_rates(
            active_settings,
            daily_rows,
            updated_at=pd.Timestamp(update.update_timestamp).to_pydatetime(),
        )

        current_results = build_tailored_recommendations(
            future_df,
            baseline_df,
            active_settings,
            comp_set_df=comp_set_df,
            reference_time=pd.Timestamp(update.update_timestamp).to_pydatetime(),
        )
        refreshed = _rate_for_date(current_results, stay_date)
        new_rate = float(refreshed["tailored_recommendation"]) if refreshed is not None and pd.notna(refreshed.get("tailored_recommendation")) else np.nan
        absolute_change = new_rate - old_rate if pd.notna(new_rate) and pd.notna(old_rate) else np.nan
        percent_change = (absolute_change / old_rate) * 100.0 if pd.notna(absolute_change) and old_rate not in (0, np.nan) else np.nan

        summary = str(update.reason_summary).strip()
        if not summary:
            direction = "increased" if pd.notna(absolute_change) and absolute_change > 0 else "decreased" if pd.notna(absolute_change) and absolute_change < 0 else "held"
            summary = f"Comp-set median updated to ${float(update.manual_daily_median_rate):,.2f}; tailored rate {direction} after replay."

        rows.append(
            {
                "update_sequence": sequence,
                "stay_date": stay_date.date().isoformat(),
                "update_timestamp": pd.Timestamp(update.update_timestamp).isoformat(),
                "old_rate": old_rate,
                "new_rate": new_rate,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "updated_median_rate": float(update.manual_daily_median_rate),
                "median_rate_source": refreshed.get("median_rate_source") if refreshed is not None else "",
                "model_status": refreshed.get("model_status") if refreshed is not None else "UNAVAILABLE",
                "reason_summary": summary,
            }
        )

    return pd.DataFrame(rows, columns=INTRADAY_RESULT_COLUMNS), validation_errors
