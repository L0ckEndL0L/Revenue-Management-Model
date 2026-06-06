"""Simple once-daily baseline pricing model for capstone comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

BASELINE_REQUIRED_INPUT_FIELDS = ["stay_date"]
BASELINE_MODEL_TYPE = "simple_once_daily_rule_based_v1"


@dataclass(frozen=True)
class BaselinePricingConfig:
    """Configurable thresholds for a simple occupancy-based baseline model."""

    high_occupancy_threshold: float = 0.85
    low_occupancy_threshold: float = 0.55
    high_occupancy_increase_pct: float = 0.05
    low_occupancy_decrease_pct: float = -0.05
    moderate_occupancy_change_pct: float = 0.00
    rate_floor: float = 40.0
    rate_ceiling: float = 600.0


def validate_baseline_inputs(df: pd.DataFrame | None) -> Dict[str, object]:
    """Validate minimum required fields for running the baseline model."""
    if df is None:
        return {
            "available": False,
            "missing_fields": list(BASELINE_REQUIRED_INPUT_FIELDS),
            "can_calculate_occupancy": False,
            "can_calculate_adr": False,
            "is_valid": False,
        }

    columns = set(df.columns)
    missing_fields = [field for field in BASELINE_REQUIRED_INPUT_FIELDS if field not in columns]

    has_occupancy_input = "occupancy" in columns
    can_calculate_occupancy = has_occupancy_input or {"rooms_available", "rooms_sold"}.issubset(columns)

    has_adr_input = "adr" in columns or "current_rate" in columns
    can_calculate_adr = has_adr_input or {"room_revenue", "rooms_sold"}.issubset(columns)

    return {
        "available": True,
        "missing_fields": missing_fields,
        "can_calculate_occupancy": can_calculate_occupancy,
        "can_calculate_adr": can_calculate_adr,
        "is_valid": len(missing_fields) == 0 and can_calculate_occupancy and can_calculate_adr,
    }


def _coerce_occupancy_decimal(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    if out.notna().any() and float(out.max()) > 1.5:
        out = out / 100.0
    return out


def _derive_current_occupancy(df: pd.DataFrame) -> pd.Series:
    if "occupancy" in df.columns:
        occupancy = _coerce_occupancy_decimal(df["occupancy"])
    else:
        occupancy = pd.Series(np.nan, index=df.index, dtype=float)

    if {"rooms_available", "rooms_sold"}.issubset(df.columns):
        rooms_available = pd.to_numeric(df["rooms_available"], errors="coerce")
        rooms_sold = pd.to_numeric(df["rooms_sold"], errors="coerce")
        derived = np.where(rooms_available > 0, rooms_sold / rooms_available, np.nan)
        occupancy = occupancy.combine_first(pd.Series(derived, index=df.index, dtype=float))

    return occupancy


def _derive_current_adr(df: pd.DataFrame) -> pd.Series:
    adr = pd.Series(np.nan, index=df.index, dtype=float)

    if "adr" in df.columns:
        adr = pd.to_numeric(df["adr"], errors="coerce")

    if "current_rate" in df.columns:
        adr = adr.combine_first(pd.to_numeric(df["current_rate"], errors="coerce"))

    if {"room_revenue", "rooms_sold"}.issubset(df.columns):
        room_revenue = pd.to_numeric(df["room_revenue"], errors="coerce")
        rooms_sold = pd.to_numeric(df["rooms_sold"], errors="coerce")
        derived = np.where(rooms_sold > 0, room_revenue / rooms_sold, np.nan)
        adr = adr.combine_first(pd.Series(derived, index=df.index, dtype=float))

    return adr


def _derive_historical_adr_by_dow(historical_df: pd.DataFrame | None) -> pd.Series:
    if historical_df is None or len(historical_df) == 0:
        return pd.Series(dtype=float)

    hist = historical_df.copy()
    hist["stay_date"] = pd.to_datetime(hist.get("stay_date"), errors="coerce")
    hist = hist.dropna(subset=["stay_date"]).copy()
    if len(hist) == 0:
        return pd.Series(dtype=float)

    hist_adr = pd.Series(np.nan, index=hist.index, dtype=float)
    if "adr" in hist.columns:
        hist_adr = pd.to_numeric(hist["adr"], errors="coerce")
    if "stly_adr" in hist.columns:
        hist_adr = hist_adr.combine_first(pd.to_numeric(hist["stly_adr"], errors="coerce"))

    if {"room_revenue", "rooms_sold"}.issubset(hist.columns):
        revenue = pd.to_numeric(hist["room_revenue"], errors="coerce")
        sold = pd.to_numeric(hist["rooms_sold"], errors="coerce")
        derived = np.where(sold > 0, revenue / sold, np.nan)
        hist_adr = hist_adr.combine_first(pd.Series(derived, index=hist.index, dtype=float))

    if {"stly_revenue", "stly_rooms_sold"}.issubset(hist.columns):
        stly_revenue = pd.to_numeric(hist["stly_revenue"], errors="coerce")
        stly_sold = pd.to_numeric(hist["stly_rooms_sold"], errors="coerce")
        derived_stly = np.where(stly_sold > 0, stly_revenue / stly_sold, np.nan)
        hist_adr = hist_adr.combine_first(pd.Series(derived_stly, index=hist.index, dtype=float))

    hist = hist.assign(day_of_week=hist["stay_date"].dt.day_name(), hist_adr=hist_adr)
    hist = hist.dropna(subset=["hist_adr"])
    if len(hist) == 0:
        return pd.Series(dtype=float)

    return hist.groupby("day_of_week")["hist_adr"].median()


def generate_baseline_pricing_recommendations(
    input_df: pd.DataFrame,
    historical_df: pd.DataFrame | None = None,
    config: BaselinePricingConfig | None = None,
) -> pd.DataFrame:
    """Generate simple once-daily baseline pricing recommendations."""
    config = config or BaselinePricingConfig()

    validation = validate_baseline_inputs(input_df)
    if not validation.get("is_valid", False):
        raise ValueError(
            "Baseline model missing required inputs. "
            f"missing_fields={validation.get('missing_fields', [])}, "
            f"can_calculate_occupancy={validation.get('can_calculate_occupancy', False)}, "
            f"can_calculate_adr={validation.get('can_calculate_adr', False)}"
        )

    df = input_df.copy()
    df["stay_date"] = pd.to_datetime(df.get("stay_date"), errors="coerce")
    df = df.dropna(subset=["stay_date"]).sort_values("stay_date").reset_index(drop=True)

    current_occupancy = _derive_current_occupancy(df)
    current_adr = _derive_current_adr(df)

    historical_adr_by_dow = _derive_historical_adr_by_dow(historical_df)
    day_of_week = df["stay_date"].dt.day_name()
    historical_adr = day_of_week.map(historical_adr_by_dow).astype(float)

    rate_anchor = current_adr.combine_first(historical_adr)

    adjustments = []
    reasons = []
    statuses = []

    for idx in range(len(df)):
        occ = current_occupancy.iloc[idx]
        anchor = rate_anchor.iloc[idx]

        if pd.isna(occ) or pd.isna(anchor) or anchor <= 0:
            adjustments.append(np.nan)
            reasons.append("Insufficient occupancy or ADR inputs; baseline recommendation unavailable")
            statuses.append("UNAVAILABLE")
            continue

        if occ >= config.high_occupancy_threshold:
            adjustments.append(config.high_occupancy_increase_pct)
            reasons.append("High occupancy detected; baseline model applies a standard rate increase")
        elif occ <= config.low_occupancy_threshold:
            adjustments.append(config.low_occupancy_decrease_pct)
            reasons.append("Low occupancy detected; baseline model applies a standard rate decrease")
        else:
            adjustments.append(config.moderate_occupancy_change_pct)
            reasons.append("Moderate occupancy detected; baseline model holds near current or historical ADR")
        statuses.append("OK")

    adjustment_pct = pd.Series(adjustments, dtype=float)
    recommended = rate_anchor * (1.0 + adjustment_pct)
    recommended = recommended.clip(lower=config.rate_floor, upper=config.rate_ceiling)

    adjustment_amount = recommended - current_adr
    adjustment_amount = adjustment_amount.where(current_adr.notna(), np.nan)
    adjustment_percent_display = adjustment_pct * 100.0

    out = pd.DataFrame(
        {
            "stay_date": df["stay_date"],
            "day_of_week": day_of_week,
            "current_occupancy": current_occupancy,
            "current_ADR": current_adr,
            "baseline_recommended_rate": recommended,
            "baseline_adjustment_amount": adjustment_amount,
            "baseline_adjustment_percent": adjustment_percent_display,
            "baseline_reason": reasons,
            "baseline_model_type": BASELINE_MODEL_TYPE,
            "baseline_status": statuses,
        }
    )

    return out
