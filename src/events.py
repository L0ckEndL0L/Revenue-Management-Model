"""
events.py
Optional events loader and event impact enrichment for Week 3 pricing logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


DEFAULT_EVENT_IMPACTS: Dict[str, float] = {
    "low": 0.02,
    "medium": 0.04,
    "high": 0.07,
}


def load_events(events_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load events CSV if provided. Required columns: date, event_name, impact_level."""
    if not events_path:
        return None

    path = Path(events_path)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    events = pd.read_csv(path)
    required = {"date", "event_name", "impact_level"}
    missing = required - set(events.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Events file missing required columns: {missing_cols}")

    events = events.copy()
    events["stay_date"] = pd.to_datetime(events["date"], errors="coerce")
    if events["stay_date"].isna().any():
        raise ValueError("Events file contains invalid dates in 'date' column")

    events["impact_level"] = events["impact_level"].astype(str).str.lower().str.strip()
    invalid = ~events["impact_level"].isin(DEFAULT_EVENT_IMPACTS.keys())
    if invalid.any():
        bad_levels = sorted(events.loc[invalid, "impact_level"].unique().tolist())
        raise ValueError(
            "Invalid impact_level values in events file: " + ", ".join(bad_levels)
        )

    events = events[["stay_date", "event_name", "impact_level"]].copy()

    # If multiple events exist on the same date, keep highest impact event.
    impact_rank = {"low": 1, "medium": 2, "high": 3}
    events["_rank"] = events["impact_level"].map(impact_rank)
    events = (
        events.sort_values(["stay_date", "_rank"], ascending=[True, False])
        .drop_duplicates(subset=["stay_date"], keep="first")
        .drop(columns=["_rank"])
    )

    return events


def apply_event_impacts(
    df: pd.DataFrame,
    events_df: Optional[pd.DataFrame],
    impact_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Merge events into recommendation frame and add event_pct adjustment."""
    result = df.copy()
    if impact_map is None:
        impact_map = DEFAULT_EVENT_IMPACTS

    if events_df is None or len(events_df) == 0:
        result["event_name"] = pd.NA
        result["impact_level"] = pd.NA
        result["event_pct"] = 0.0
        return result

    result = result.merge(events_df, on="stay_date", how="left")
    result["event_pct"] = result["impact_level"].map(impact_map).fillna(0.0)
    return result
