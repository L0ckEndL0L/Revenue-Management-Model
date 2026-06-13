"""Input preparation helpers for the RMS pipeline."""

from __future__ import annotations

import pandas as pd

from src.ingest import process_file
from src.metrics import calculate_daily_metrics


def prepare_future_dataset(
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

    if "stay_date" in future_df.columns:
        future_df["stay_date"] = pd.to_datetime(future_df["stay_date"], errors="coerce")
        future_df = future_df.dropna(subset=["stay_date"]).reset_index(drop=True)

    if "room_revenue" not in future_df.columns:
        future_df["room_revenue"] = pd.NA

    if "current_rate" not in future_df.columns:
        future_df["current_rate"] = pd.NA

    if "adr" in future_df.columns:
        derived_from_adr = pd.to_numeric(future_df["adr"], errors="coerce")
        current_rate = pd.to_numeric(future_df["current_rate"], errors="coerce")
        use_adr_mask = current_rate.isna() | (current_rate <= 0)
        future_df.loc[use_adr_mask, "current_rate"] = derived_from_adr.loc[use_adr_mask]

    if "room_revenue" in future_df.columns:
        current_rate = pd.to_numeric(future_df["current_rate"], errors="coerce")
        sold = pd.to_numeric(future_df["rooms_sold"], errors="coerce").replace(0, pd.NA)
        derived_rate = pd.to_numeric(future_df["room_revenue"], errors="coerce") / sold
        use_derived_mask = current_rate.isna() | (current_rate <= 0)
        future_df.loc[use_derived_mask, "current_rate"] = derived_rate.loc[use_derived_mask]

    future_df["rooms_sold"] = pd.to_numeric(future_df["rooms_sold"], errors="coerce").fillna(0.0)

    if manual_rooms_available is not None:
        future_df["rooms_available"] = int(manual_rooms_available)

    return future_df.sort_values("stay_date").reset_index(drop=True)


def build_uploaded_stly_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded comparison dataset into the STLY shape used downstream."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["stay_date", "stly_occupancy", "stly_rooms_sold", "stly_adr", "stly_revenue"])

    ref = df.copy()
    ref["stay_date"] = pd.to_datetime(ref["stay_date"], errors="coerce")
    ref = ref.dropna(subset=["stay_date"]).copy()
    if len(ref) == 0:
        return pd.DataFrame(columns=["stay_date", "stly_occupancy", "stly_rooms_sold", "stly_adr", "stly_revenue"])

    rename_map = {}
    if "occupancy" in ref.columns:
        rename_map["occupancy"] = "stly_occupancy"
    if "rooms_sold" in ref.columns:
        rename_map["rooms_sold"] = "stly_rooms_sold"
    if "adr" in ref.columns:
        rename_map["adr"] = "stly_adr"
    if "room_revenue" in ref.columns:
        rename_map["room_revenue"] = "stly_revenue"

    ref = ref.rename(columns=rename_map)
    cols = ["stay_date", "stly_occupancy", "stly_rooms_sold", "stly_adr", "stly_revenue"]
    for col in cols:
        if col not in ref.columns:
            ref[col] = pd.NA
    return ref[cols].copy()


def select_user_comparison_frames(
    historical_metrics: pd.DataFrame,
    future_df: pd.DataFrame,
    repo_stly_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, bool]:
    """Choose current/prior comparison frames, preferring uploaded adjacent-year files."""
    fallback_current = historical_metrics
    fallback_prior = repo_stly_df

    if future_df is None or len(future_df) == 0:
        return fallback_current, fallback_prior, fallback_prior, False

    future_candidate = future_df.copy()
    future_candidate["stay_date"] = pd.to_datetime(future_candidate.get("stay_date"), errors="coerce")
    future_candidate = future_candidate.dropna(subset=["stay_date"]).copy()
    if len(future_candidate) == 0:
        return fallback_current, fallback_prior, fallback_prior, False

    required_metric_cols = {"rooms_available", "rooms_sold", "room_revenue"}
    if not required_metric_cols.issubset(future_candidate.columns):
        return fallback_current, fallback_prior, fallback_prior, False

    hist_years = set(pd.to_datetime(historical_metrics["stay_date"], errors="coerce").dropna().dt.year.unique().tolist())
    future_years = set(future_candidate["stay_date"].dt.year.unique().tolist())
    if not hist_years or not future_years:
        return fallback_current, fallback_prior, fallback_prior, False

    if min(future_years) <= max(hist_years):
        return fallback_current, fallback_prior, fallback_prior, False

    future_metrics = calculate_daily_metrics(future_candidate)
    uploaded_prior = build_uploaded_stly_reference(historical_metrics)
    return future_metrics, historical_metrics, uploaded_prior, True
