"""Tailored pricing model built on top of baseline recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_UPDATE_FREQUENCIES = {
    "Every 30 minutes": timedelta(minutes=30),
    "Every hour": timedelta(hours=1),
    "Every 2 hours": timedelta(hours=2),
    "Daily": timedelta(days=1),
    "Manual only": None,
}
MONTHLY_COMP_RATE_MODE = "Monthly comp rate"
DAILY_COMP_RATE_MODE = "Daily comp rates"
ALLOWED_COMP_RATE_INPUT_MODES = {MONTHLY_COMP_RATE_MODE, DAILY_COMP_RATE_MODE}

MANUAL_DAILY_SOURCE = "Manual daily input"
DATASET_DERIVED_SOURCE = "Dataset-derived daily median"
GLOBAL_FALLBACK_SOURCE = "Global manual median fallback"
MISSING_MEDIAN_SOURCE = "Missing median"

DAILY_MEDIAN_COLUMNS = [
    "stay_date",
    "suggested_dataset_median_rate",
    "manual_daily_median_rate",
    "global_median_fallback",
    "final_median_rate_used",
    "median_rate_source",
    "last_median_update_timestamp",
]


def _finite_float_or_default(value: Any, default: float = 0.0) -> float:
    """Return a finite float, replacing missing or invalid model inputs."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric_value if np.isfinite(numeric_value) else float(default)

SEGMENT_FOCUS_PRESETS = {
    "balanced": {
        "demand_bias": 0.00,
        "strategy_bias": 0.00,
        "anchor_bias": 0.00,
        "rate_bias": 0.00,
        "movement_multiplier": 1.00,
        "description": "balanced rate and occupancy posture",
    },
    "revenue": {
        "demand_bias": 0.02,
        "strategy_bias": 0.14,
        "anchor_bias": 0.08,
        "rate_bias": 0.012,
        "movement_multiplier": 1.10,
        "description": "revenue-focused posture with stronger rate confidence",
    },
    "occupancy": {
        "demand_bias": -0.03,
        "strategy_bias": -0.20,
        "anchor_bias": -0.06,
        "rate_bias": -0.014,
        "movement_multiplier": 0.90,
        "description": "occupancy-focused posture that protects volume",
    },
    "corporate": {
        "demand_bias": 0.01,
        "strategy_bias": 0.04,
        "anchor_bias": -0.08,
        "rate_bias": -0.004,
        "movement_multiplier": 0.70,
        "description": "corporate posture with steadier, less volatile rate movement",
    },
    "group": {
        "demand_bias": -0.04,
        "strategy_bias": -0.24,
        "anchor_bias": -0.12,
        "rate_bias": -0.020,
        "movement_multiplier": 0.80,
        "description": "group-focused posture that favors volume and conversion",
    },
    "premium": {
        "demand_bias": 0.05,
        "strategy_bias": 0.22,
        "anchor_bias": 0.14,
        "rate_bias": 0.020,
        "movement_multiplier": 1.15,
        "description": "premium posture with stronger comp-rate anchoring and rate lift",
    },
    "leisure": {
        "demand_bias": 0.015,
        "strategy_bias": 0.05,
        "anchor_bias": 0.04,
        "rate_bias": 0.006,
        "movement_multiplier": 1.05,
        "description": "leisure posture that leans into seasonal and event demand",
    },
}

PROPERTY_TYPE_PRESETS = {
    "full service": {
        "demand_bias": 0.01,
        "strategy_bias": 0.04,
        "anchor_bias": 0.02,
        "rate_bias": 0.004,
        "seasonality_multiplier": 1.00,
        "event_multiplier": 1.00,
        "movement_multiplier": 1.00,
        "description": "full-service posture with mild revenue confidence",
    },
    "limited service": {
        "demand_bias": -0.015,
        "strategy_bias": -0.08,
        "anchor_bias": -0.04,
        "rate_bias": -0.006,
        "seasonality_multiplier": 0.90,
        "event_multiplier": 0.85,
        "movement_multiplier": 0.90,
        "description": "limited-service posture that keeps rates practical and occupancy-aware",
    },
    "select service": {
        "demand_bias": -0.01,
        "strategy_bias": -0.04,
        "anchor_bias": -0.02,
        "rate_bias": -0.004,
        "seasonality_multiplier": 0.95,
        "event_multiplier": 0.90,
        "movement_multiplier": 0.95,
        "description": "select-service posture with restrained rate movement",
    },
    "luxury": {
        "demand_bias": 0.05,
        "strategy_bias": 0.18,
        "anchor_bias": 0.16,
        "rate_bias": 0.018,
        "seasonality_multiplier": 1.05,
        "event_multiplier": 1.10,
        "movement_multiplier": 1.10,
        "description": "luxury posture with stronger comp-rate anchoring and rate confidence",
    },
    "resort": {
        "demand_bias": 0.035,
        "strategy_bias": 0.10,
        "anchor_bias": 0.08,
        "rate_bias": 0.012,
        "seasonality_multiplier": 1.25,
        "event_multiplier": 1.20,
        "movement_multiplier": 1.08,
        "description": "resort posture that leans into seasonality and event compression",
    },
    "boutique": {
        "demand_bias": 0.025,
        "strategy_bias": 0.10,
        "anchor_bias": 0.10,
        "rate_bias": 0.010,
        "seasonality_multiplier": 1.05,
        "event_multiplier": 1.05,
        "movement_multiplier": 1.05,
        "description": "boutique posture with premium positioning and comp-set influence",
    },
    "extended stay": {
        "demand_bias": -0.015,
        "strategy_bias": -0.10,
        "anchor_bias": -0.10,
        "rate_bias": -0.006,
        "seasonality_multiplier": 0.75,
        "event_multiplier": 0.70,
        "movement_multiplier": 0.65,
        "description": "extended-stay posture with slower, steadier rate movement",
    },
    "economy": {
        "demand_bias": -0.035,
        "strategy_bias": -0.18,
        "anchor_bias": -0.08,
        "rate_bias": -0.014,
        "seasonality_multiplier": 0.85,
        "event_multiplier": 0.80,
        "movement_multiplier": 0.85,
        "description": "economy posture that protects conversion and price sensitivity",
    },
}


@dataclass
class TailoredModelSettings:
    """User-configurable settings for property-specific pricing adjustments."""

    property_type: str = "Full Service"
    segment_focus: str = "Balanced"
    baseline_occupancy_sensitivity: float = 1.0
    adr_sensitivity: float = 1.0
    revpar_priority: float = 1.0
    rooms_sold_priority: float = 1.0
    revenue_priority: float = 1.0
    demand_adjustment_factor: float = 1.0
    seasonality_adjustment_factor: float = 1.0
    event_impact_factor: float = 1.0
    minimum_acceptable_rate: float = 80.0
    maximum_recommended_rate: float = 450.0
    comp_rate_input_mode: str = DAILY_COMP_RATE_MODE
    global_median_rate_fallback: float | None = None
    median_rate: float | None = None
    median_rate_update_frequency: str = "Manual only"
    median_rate_last_updated: str | None = None
    daily_median_rates: list[dict[str, Any]] = field(default_factory=list)


def default_tailored_settings() -> dict[str, Any]:
    """Return safe default settings for the tailored model."""
    return asdict(TailoredModelSettings())


def normalize_tailored_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user-provided settings onto defaults without validating ranges."""
    normalized = default_tailored_settings()
    if settings:
        for key, value in settings.items():
            if key in normalized:
                normalized[key] = value

    if normalized.get("global_median_rate_fallback") in (None, "") and normalized.get("median_rate") not in (None, ""):
        normalized["global_median_rate_fallback"] = normalized.get("median_rate")

    if normalized.get("daily_median_rates") is None:
        normalized["daily_median_rates"] = []
    if normalized.get("comp_rate_input_mode") not in ALLOWED_COMP_RATE_INPUT_MODES:
        normalized["comp_rate_input_mode"] = DAILY_COMP_RATE_MODE

    return normalized


def _coerce_float(value: Any, field_name: str, errors: list[str], allow_none: bool = False) -> float | None:
    if value is None or value == "":
        if allow_none:
            return None
        errors.append(f"{field_name} is required")
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{field_name} must be numeric")
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _normalize_stay_date(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize().date().isoformat()


def _positive_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(df[column], errors="coerce")
    values = values.where(values > 0)
    return values.dropna()


def _series_or_default(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _derive_current_adr(df: pd.DataFrame) -> pd.Series:
    current_rate = _series_or_default(df, "current_rate")
    adr = _series_or_default(df, "adr")
    recommended_rate = _series_or_default(df, "recommended_rate")
    room_revenue = _series_or_default(df, "room_revenue")
    rooms_sold = _series_or_default(df, "rooms_sold").replace(0, np.nan)
    derived = room_revenue / rooms_sold
    return current_rate.combine_first(adr).combine_first(recommended_rate).combine_first(derived)


def _derive_occupancy(df: pd.DataFrame) -> pd.Series:
    occupancy = _series_or_default(df, "occupancy")
    if occupancy.notna().any() and float(occupancy.max()) > 1.5:
        occupancy = occupancy / 100.0
    forecast_occ = _series_or_default(df, "forecast_occ")
    rooms_available = _series_or_default(df, "rooms_available").replace(0, np.nan)
    rooms_sold = _series_or_default(df, "rooms_sold")
    derived = rooms_sold / rooms_available
    return occupancy.combine_first(forecast_occ).combine_first(derived)


def validate_tailored_settings(settings: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Validate tailored settings and return a sanitized copy plus user-facing errors."""
    sanitized = normalize_tailored_settings(settings)
    errors: list[str] = []

    range_fields = [
        "baseline_occupancy_sensitivity",
        "adr_sensitivity",
        "revpar_priority",
        "rooms_sold_priority",
        "revenue_priority",
        "demand_adjustment_factor",
        "seasonality_adjustment_factor",
        "event_impact_factor",
    ]

    for field_name in range_fields:
        value = _coerce_float(sanitized.get(field_name), field_name.replace("_", " "), errors)
        if value is None:
            continue
        if value < 0.0 or value > 2.0:
            errors.append(f"{field_name.replace('_', ' ')} must be between 0.0 and 2.0")
            continue
        sanitized[field_name] = float(value)

    minimum_rate = _coerce_float(sanitized.get("minimum_acceptable_rate"), "minimum acceptable rate", errors)
    if minimum_rate is not None:
        if minimum_rate <= 0:
            errors.append("minimum acceptable rate must be greater than 0")
        else:
            sanitized["minimum_acceptable_rate"] = float(minimum_rate)

    maximum_rate = _coerce_float(sanitized.get("maximum_recommended_rate"), "maximum recommended rate", errors)
    if maximum_rate is not None:
        if maximum_rate <= 0:
            errors.append("maximum recommended rate must be greater than 0")
        else:
            sanitized["maximum_recommended_rate"] = float(maximum_rate)

    if minimum_rate is not None and maximum_rate is not None and maximum_rate <= minimum_rate:
        errors.append("maximum recommended rate must be greater than minimum acceptable rate")

    comp_rate_input_mode = str(sanitized.get("comp_rate_input_mode", DAILY_COMP_RATE_MODE))
    if comp_rate_input_mode not in ALLOWED_COMP_RATE_INPUT_MODES:
        errors.append("comp rate input mode must be one of: Monthly comp rate, Daily comp rates")
        comp_rate_input_mode = DAILY_COMP_RATE_MODE
    sanitized["comp_rate_input_mode"] = comp_rate_input_mode

    global_median = _coerce_float(sanitized.get("global_median_rate_fallback"), "global median fallback", errors, allow_none=True)
    if global_median is not None:
        if global_median <= 0:
            errors.append("global median fallback must be greater than 0")
            sanitized["global_median_rate_fallback"] = None
        else:
            sanitized["global_median_rate_fallback"] = float(global_median)
    else:
        sanitized["global_median_rate_fallback"] = None

    sanitized["median_rate"] = sanitized.get("global_median_rate_fallback")

    update_frequency = str(sanitized.get("median_rate_update_frequency", "Manual only"))
    if update_frequency not in ALLOWED_UPDATE_FREQUENCIES:
        errors.append("update frequency must be one of: Every 30 minutes, Every hour, Every 2 hours, Daily, Manual only")
    else:
        sanitized["median_rate_update_frequency"] = update_frequency

    if not str(sanitized.get("property_type", "")).strip():
        sanitized["property_type"] = TailoredModelSettings.property_type
    if not str(sanitized.get("segment_focus", "")).strip():
        sanitized["segment_focus"] = TailoredModelSettings.segment_focus

    timestamp = _parse_timestamp(sanitized.get("median_rate_last_updated"))
    sanitized["median_rate_last_updated"] = timestamp.isoformat() if timestamp else None

    raw_daily_rows = sanitized.get("daily_median_rates", [])
    if isinstance(raw_daily_rows, pd.DataFrame):
        raw_daily_rows = raw_daily_rows.to_dict("records")
    if not isinstance(raw_daily_rows, list):
        errors.append("daily median rates must be a list of rows")
        raw_daily_rows = []

    daily_rows: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for raw_row in raw_daily_rows:
        if not isinstance(raw_row, dict):
            errors.append("daily median rows must be objects")
            continue

        stay_date = _normalize_stay_date(raw_row.get("stay_date"))
        if stay_date is None:
            errors.append("daily median rows must include a valid stay date")
            continue
        if stay_date in seen_dates:
            errors.append(f"duplicate daily median row for {stay_date}")
            continue
        seen_dates.add(stay_date)

        manual_daily = _coerce_float(raw_row.get("manual_daily_median_rate"), f"manual daily median rate for {stay_date}", errors, allow_none=True)
        if manual_daily is not None and manual_daily <= 0:
            errors.append(f"manual daily median rate for {stay_date} must be greater than 0")
            manual_daily = None

        suggested_daily = _coerce_float(raw_row.get("suggested_dataset_median_rate"), f"suggested dataset median rate for {stay_date}", errors, allow_none=True)
        if suggested_daily is not None and suggested_daily <= 0:
            suggested_daily = None

        row_timestamp = _parse_timestamp(raw_row.get("last_median_update_timestamp") or raw_row.get("median_rate_last_updated"))

        daily_rows.append(
            {
                "stay_date": stay_date,
                "suggested_dataset_median_rate": float(suggested_daily) if suggested_daily is not None else None,
                "manual_daily_median_rate": float(manual_daily) if manual_daily is not None else None,
                "last_median_update_timestamp": row_timestamp.isoformat() if row_timestamp else None,
            }
        )

    sanitized["daily_median_rates"] = sorted(daily_rows, key=lambda row: row["stay_date"])
    return sanitized, errors


def is_median_rate_stale(settings: dict[str, Any], reference_time: datetime | None = None) -> bool:
    """Return True when the global median fallback is older than its configured review cadence."""
    reference_time = reference_time or datetime.now()
    validated, _ = validate_tailored_settings(settings)
    cadence = ALLOWED_UPDATE_FREQUENCIES.get(validated["median_rate_update_frequency"])
    if cadence is None:
        return False

    timestamp = _parse_timestamp(validated.get("median_rate_last_updated"))
    if timestamp is None:
        return bool(validated.get("global_median_rate_fallback"))

    return reference_time - timestamp > cadence


def update_median_rate(settings: dict[str, Any] | None, median_rate: float, updated_at: datetime | None = None) -> dict[str, Any]:
    """Update the global median fallback and timestamp in one step."""
    updated_at = updated_at or datetime.now()
    merged = normalize_tailored_settings(settings)
    merged["global_median_rate_fallback"] = float(median_rate)
    merged["median_rate"] = float(median_rate)
    merged["median_rate_last_updated"] = updated_at.isoformat()
    return merged


def update_daily_median_rates(
    settings: dict[str, Any] | None,
    daily_rows: pd.DataFrame | list[dict[str, Any]],
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist manual day-by-day median edits while preserving timestamps for unchanged rows."""
    updated_at = updated_at or datetime.now()
    merged = normalize_tailored_settings(settings)
    existing_settings, _ = validate_tailored_settings(merged)
    existing_rows = {row["stay_date"]: row for row in existing_settings.get("daily_median_rates", [])}

    if isinstance(daily_rows, pd.DataFrame):
        row_dicts = daily_rows.to_dict("records")
    else:
        row_dicts = list(daily_rows)

    normalized_rows: list[dict[str, Any]] = []
    for row in row_dicts:
        stay_date = _normalize_stay_date(row.get("stay_date"))
        if stay_date is None:
            continue

        manual_raw = row.get("manual_daily_median_rate")
        manual_value = None if manual_raw in (None, "") or pd.isna(manual_raw) else float(manual_raw)
        if manual_value is not None and manual_value <= 0:
            manual_value = None

        previous = existing_rows.get(stay_date, {})
        previous_manual = previous.get("manual_daily_median_rate")
        if manual_value is not None and manual_value != previous_manual:
            row_timestamp = updated_at.isoformat()
        elif manual_value is not None:
            row_timestamp = previous.get("last_median_update_timestamp")
        else:
            row_timestamp = None

        suggested_raw = row.get("suggested_dataset_median_rate")
        suggested_value = None if suggested_raw in (None, "") or pd.isna(suggested_raw) else float(suggested_raw)

        normalized_rows.append(
            {
                "stay_date": stay_date,
                "suggested_dataset_median_rate": suggested_value,
                "manual_daily_median_rate": manual_value,
                "last_median_update_timestamp": row_timestamp,
            }
        )

    merged["daily_median_rates"] = normalized_rows
    merged["median_rate"] = merged.get("global_median_rate_fallback")
    return merged


def infer_median_rate_from_dataset(df: pd.DataFrame, baseline_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Infer practical median rates per stay date from available dataset columns."""
    baseline_df = baseline_df if baseline_df is not None else pd.DataFrame()
    if df is None or len(df) == 0 or "stay_date" not in df.columns:
        return pd.DataFrame(columns=["stay_date", "suggested_dataset_median_rate"])

    base = df.copy()
    base["stay_date"] = pd.to_datetime(base.get("stay_date"), errors="coerce")
    base = base.dropna(subset=["stay_date"]).copy()
    if len(base) == 0:
        return pd.DataFrame(columns=["stay_date", "suggested_dataset_median_rate"])

    candidate_frames: list[pd.DataFrame] = []
    rate_columns = ["current_rate", "adr", "recommended_rate"]
    for column in rate_columns:
        if column in base.columns:
            values = pd.to_numeric(base[column], errors="coerce")
            values = values.where(values > 0)
            candidate_frames.append(pd.DataFrame({"stay_date": base["stay_date"], "candidate_rate": values}))

    derived_adr = _derive_current_adr(base)
    candidate_frames.append(pd.DataFrame({"stay_date": base["stay_date"], "candidate_rate": derived_adr.where(derived_adr > 0)}))

    candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(columns=["stay_date", "candidate_rate"])
    candidate_df = candidate_df.dropna(subset=["stay_date", "candidate_rate"])
    candidate_df = candidate_df[candidate_df["candidate_rate"] > 0]
    if len(candidate_df) == 0:
        return pd.DataFrame(columns=["stay_date", "suggested_dataset_median_rate"])

    result = (
        candidate_df.groupby(candidate_df["stay_date"].dt.normalize())["candidate_rate"]
        .median()
        .reset_index()
        .rename(columns={"candidate_rate": "suggested_dataset_median_rate"})
        .sort_values("stay_date")
        .reset_index(drop=True)
    )
    return result


def infer_median_rate_from_comp_set(comp_set_df: pd.DataFrame | None) -> pd.DataFrame:
    """Infer date-level median comp rates from normalized rate-shop rows."""
    if comp_set_df is None or len(comp_set_df) == 0:
        return pd.DataFrame(columns=["stay_date", "suggested_dataset_median_rate"])

    required = {"stay_date", "rate"}
    if not required.issubset(set(comp_set_df.columns)):
        return pd.DataFrame(columns=["stay_date", "suggested_dataset_median_rate"])

    comp = comp_set_df.copy()
    comp["stay_date"] = pd.to_datetime(comp["stay_date"], errors="coerce")
    comp["rate"] = pd.to_numeric(comp["rate"], errors="coerce")
    comp = comp.dropna(subset=["stay_date", "rate"])
    comp = comp[comp["rate"] > 0]
    if len(comp) == 0:
        return pd.DataFrame(columns=["stay_date", "suggested_dataset_median_rate"])

    return (
        comp.groupby(comp["stay_date"].dt.normalize())["rate"]
        .median()
        .reset_index()
        .rename(columns={"rate": "suggested_dataset_median_rate"})
        .sort_values("stay_date")
        .reset_index(drop=True)
    )


def build_daily_median_rate_table(
    future_df: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    baseline_df: pd.DataFrame | None = None,
    comp_set_df: pd.DataFrame | None = None,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """Create a date-level median table for the UI and recommendation logic."""
    reference_time = reference_time or datetime.now()
    validated_settings, errors = validate_tailored_settings(settings)
    if errors:
        raise ValueError("Tailored model settings are invalid: " + "; ".join(errors))

    if future_df is None or len(future_df) == 0 or "stay_date" not in future_df.columns:
        return pd.DataFrame(columns=DAILY_MEDIAN_COLUMNS)

    forecast_dates = pd.DataFrame({"stay_date": pd.to_datetime(future_df["stay_date"], errors="coerce")})
    forecast_dates = forecast_dates.dropna(subset=["stay_date"]).drop_duplicates().sort_values("stay_date").reset_index(drop=True)
    suggested = infer_median_rate_from_comp_set(comp_set_df)
    if len(suggested) == 0:
        suggested = infer_median_rate_from_dataset(future_df, baseline_df=baseline_df)
    use_daily_comp_rates = validated_settings.get("comp_rate_input_mode") == DAILY_COMP_RATE_MODE
    manual_lookup = pd.DataFrame(validated_settings.get("daily_median_rates", []) if use_daily_comp_rates else [])
    if len(manual_lookup) > 0:
        manual_lookup["stay_date"] = pd.to_datetime(manual_lookup["stay_date"], errors="coerce")
    else:
        manual_lookup = pd.DataFrame(columns=["stay_date", "manual_daily_median_rate", "last_median_update_timestamp"])

    table = forecast_dates.merge(suggested, on="stay_date", how="left")
    table = table.merge(
        manual_lookup[[column for column in ["stay_date", "manual_daily_median_rate", "last_median_update_timestamp"] if column in manual_lookup.columns]],
        on="stay_date",
        how="left",
    )
    table["global_median_fallback"] = validated_settings.get("global_median_rate_fallback")

    source_values = []
    final_values = []
    timestamp_values = []
    global_stale = is_median_rate_stale(validated_settings, reference_time=reference_time)
    for row in table.itertuples(index=False):
        manual_daily = getattr(row, "manual_daily_median_rate", np.nan)
        suggested_daily = getattr(row, "suggested_dataset_median_rate", np.nan)
        global_fallback = getattr(row, "global_median_fallback", np.nan)
        row_timestamp = getattr(row, "last_median_update_timestamp", None)

        if pd.notna(manual_daily):
            source_values.append(MANUAL_DAILY_SOURCE)
            final_values.append(float(manual_daily))
            timestamp_values.append(row_timestamp)
        elif use_daily_comp_rates and pd.notna(suggested_daily):
            source_values.append(DATASET_DERIVED_SOURCE)
            final_values.append(float(suggested_daily))
            timestamp_values.append(None)
        elif pd.notna(global_fallback):
            source_values.append(GLOBAL_FALLBACK_SOURCE)
            final_values.append(float(global_fallback))
            timestamp_values.append(validated_settings.get("median_rate_last_updated"))
        else:
            source_values.append(MISSING_MEDIAN_SOURCE)
            final_values.append(np.nan)
            timestamp_values.append(None)

    table["final_median_rate_used"] = final_values
    table["median_rate_source"] = source_values
    table["last_median_update_timestamp"] = timestamp_values
    if global_stale:
        table.loc[table["median_rate_source"] == GLOBAL_FALLBACK_SOURCE, "last_median_update_timestamp"] = validated_settings.get("median_rate_last_updated")

    return table[DAILY_MEDIAN_COLUMNS].sort_values("stay_date").reset_index(drop=True)


def _property_type_preset(property_type: str) -> dict[str, float | str]:
    property_key = str(property_type).strip().lower()
    return PROPERTY_TYPE_PRESETS.get(property_key, PROPERTY_TYPE_PRESETS["full service"])


def _segment_focus_preset(segment_focus: str) -> dict[str, float | str]:
    segment = str(segment_focus).strip().lower()
    return SEGMENT_FOCUS_PRESETS.get(segment, SEGMENT_FOCUS_PRESETS["balanced"])


def _seasonality_index(stay_date: pd.Timestamp) -> float:
    month = int(stay_date.month)
    dow = int(stay_date.dayofweek)

    if month in {6, 7, 8, 12}:
        score = 0.25
    elif month in {1, 2}:
        score = -0.15
    else:
        score = 0.05

    if dow in {4, 5}:
        score += 0.08
    return score


def _event_index(event_pct: float, impact_level: Any) -> float:
    impact_map = {"low": 0.05, "medium": 0.10, "high": 0.18}
    impact_signal = impact_map.get(str(impact_level).strip().lower(), 0.0)
    return float(event_pct) + impact_signal


def _confidence_label(warnings: list[str], inputs_complete: bool) -> str:
    if not inputs_complete:
        return "low"
    if len(warnings) >= 2:
        return "medium"
    return "high"


def build_tailored_recommendations(
    future_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    comp_set_df: pd.DataFrame | None = None,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """Adjust baseline recommendations using property-specific tailored settings."""
    reference_time = reference_time or datetime.now()
    validated_settings, errors = validate_tailored_settings(settings)
    if errors:
        raise ValueError("Tailored model settings are invalid: " + "; ".join(errors))

    output_columns = [
        "stay_date",
        "occupancy",
        "adr",
        "revpar",
        "rooms_sold",
        "room_revenue",
        "baseline_recommendation",
        "suggested_dataset_median_rate",
        "manual_daily_median_rate",
        "global_median_fallback",
        "median_rate_used",
        "median_rate_source",
        "difference_from_median_rate",
        "tailored_recommendation",
        "recommended_rate_adjustment",
        "model_status",
        "confidence",
        "reasoning_notes",
        "warnings",
        "last_median_update_timestamp",
        "median_rate_last_updated",
        "median_rate_update_frequency",
        "property_type",
        "segment_focus",
    ]
    if len(future_df) == 0:
        return pd.DataFrame(columns=output_columns)

    df = future_df.copy()
    df["stay_date"] = pd.to_datetime(df.get("stay_date"), errors="coerce")

    baseline_merge_cols = ["stay_date", "baseline_recommended_rate", "baseline_status", "baseline_reason"]
    baseline = baseline_df.copy()
    baseline["stay_date"] = pd.to_datetime(baseline.get("stay_date"), errors="coerce")
    baseline = baseline[[col for col in baseline_merge_cols if col in baseline.columns]]
    merged = df.merge(baseline, on="stay_date", how="left")

    merged["occupancy"] = _derive_occupancy(merged)
    merged["adr"] = _derive_current_adr(merged)
    merged["rooms_sold"] = _series_or_default(merged, "rooms_sold", default=0.0).fillna(0.0)
    merged["room_revenue"] = _series_or_default(merged, "room_revenue")
    merged["room_revenue"] = merged["room_revenue"].combine_first(merged["adr"] * merged["rooms_sold"])
    merged["revpar"] = merged["adr"] * merged["occupancy"]

    daily_median_table = build_daily_median_rate_table(
        merged,
        validated_settings,
        baseline_df=baseline_df,
        comp_set_df=comp_set_df,
        reference_time=reference_time,
    )
    merged = merged.merge(daily_median_table, on="stay_date", how="left")

    property_preset = _property_type_preset(str(validated_settings["property_type"]))
    segment_preset = _segment_focus_preset(str(validated_settings["segment_focus"]))
    global_stale = is_median_rate_stale(validated_settings, reference_time=reference_time)

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        baseline_rate = float(getattr(row, "baseline_recommended_rate", np.nan))
        adr = float(getattr(row, "adr", np.nan)) if pd.notna(getattr(row, "adr", np.nan)) else np.nan
        occupancy = float(getattr(row, "occupancy", np.nan)) if pd.notna(getattr(row, "occupancy", np.nan)) else np.nan
        stay_date = pd.Timestamp(getattr(row, "stay_date"))
        pace_variance = _finite_float_or_default(getattr(row, "pace_variance", 0.0))
        event_pct = _finite_float_or_default(getattr(row, "event_pct", 0.0))
        impact_level = getattr(row, "impact_level", None)
        median_rate_used = getattr(row, "final_median_rate_used", np.nan)
        median_rate_source = getattr(row, "median_rate_source", MISSING_MEDIAN_SOURCE)
        manual_daily_median_rate = getattr(row, "manual_daily_median_rate", np.nan)
        suggested_dataset_median_rate = getattr(row, "suggested_dataset_median_rate", np.nan)
        global_median_fallback = getattr(row, "global_median_fallback", np.nan)
        last_median_update_timestamp = getattr(row, "last_median_update_timestamp", None)
        warnings: list[str] = []

        if pd.isna(baseline_rate) or baseline_rate <= 0:
            baseline_rate = adr if pd.notna(adr) and adr > 0 else np.nan
            warnings.append("Baseline recommendation unavailable; using current ADR where possible")

        if median_rate_source == MISSING_MEDIAN_SOURCE or pd.isna(median_rate_used):
            warnings.append("No daily or global median rate was available for this forecast date")
        elif median_rate_source == DATASET_DERIVED_SOURCE:
            warnings.append("Manual daily median was blank; using dataset-derived daily median")
        elif median_rate_source == GLOBAL_FALLBACK_SOURCE:
            warnings.append("Using the global median fallback because no date-level median was available")

        if global_stale and median_rate_source == GLOBAL_FALLBACK_SOURCE:
            warnings.append("Global median fallback review cadence is overdue")

        if pd.notna(occupancy) and (occupancy < 0 or occupancy > 1.2):
            warnings.append("Occupancy is outside expected range")

        if pd.notna(median_rate_used):
            if median_rate_used < validated_settings["minimum_acceptable_rate"]:
                warnings.append("Median rate is below the minimum acceptable rate")
            if median_rate_used > validated_settings["maximum_recommended_rate"]:
                warnings.append("Median rate is above the maximum recommended rate")

        if pd.isna(baseline_rate):
            tailored_rate = np.nan
            model_status = "REVIEW_REQUIRED"
            reasoning = "Tailored recommendation unavailable because neither baseline rate nor ADR could be derived."
        else:
            occupancy_value = occupancy if pd.notna(occupancy) else 0.0
            demand_index = (
                (occupancy_value - 0.72) * 0.80 * validated_settings["baseline_occupancy_sensitivity"]
                + pace_variance * 0.60 * validated_settings["demand_adjustment_factor"]
                + _seasonality_index(stay_date) * 0.40 * validated_settings["seasonality_adjustment_factor"] * float(property_preset["seasonality_multiplier"])
                + _event_index(event_pct, impact_level) * 0.45 * validated_settings["event_impact_factor"] * float(property_preset["event_multiplier"])
                + float(property_preset["demand_bias"])
                + float(segment_preset["demand_bias"])
            )
            demand_index = float(np.clip(demand_index, -0.50, 0.60))

            strategy_index = (
                (validated_settings["revenue_priority"] - 1.0) * 0.55
                + (validated_settings["revpar_priority"] - 1.0) * 0.35
                - (validated_settings["rooms_sold_priority"] - 1.0) * 0.45
                + float(property_preset["strategy_bias"])
                + float(segment_preset["strategy_bias"])
            )
            strategy_index = float(np.clip(strategy_index, -0.35, 0.35))

            if pd.notna(median_rate_used) and median_rate_used > 0:
                anchor_weight = float(np.clip(
                    0.45
                    + strategy_index * 0.25
                    + demand_index * 0.30
                    + float(property_preset["anchor_bias"])
                    + float(segment_preset["anchor_bias"]),
                    0.15,
                    0.95,
                ))
                anchored_rate = baseline_rate + (median_rate_used - baseline_rate) * anchor_weight
                if abs(baseline_rate - median_rate_used) / median_rate_used > 0.20:
                    warnings.append("Baseline recommendation is materially different from the selected median rate")
            else:
                anchored_rate = baseline_rate

            adjustment_pct = (
                demand_index * 0.12
                + strategy_index * 0.05
                + float(property_preset["rate_bias"])
                + float(segment_preset["rate_bias"])
            ) * float(property_preset["movement_multiplier"]) * float(segment_preset["movement_multiplier"])
            tailored_rate = anchored_rate * (1.0 + adjustment_pct)

            if pd.notna(occupancy) and occupancy >= 0.90 and pd.notna(median_rate_used):
                tailored_rate = max(tailored_rate, median_rate_used * 1.02)
            elif pd.notna(occupancy) and occupancy <= 0.55 and pd.notna(median_rate_used):
                tailored_rate = min(tailored_rate, median_rate_used * 0.99)

            minimum_rate = float(validated_settings["minimum_acceptable_rate"])
            maximum_rate = float(validated_settings["maximum_recommended_rate"])
            unclipped_rate = tailored_rate
            tailored_rate = float(np.clip(tailored_rate, minimum_rate, maximum_rate))
            if tailored_rate != unclipped_rate:
                warnings.append("Tailored recommendation was capped by property rate limits")

            if median_rate_source == MISSING_MEDIAN_SOURCE:
                model_status = "WARNING_MISSING_MEDIAN"
            elif global_stale and median_rate_source == GLOBAL_FALLBACK_SOURCE:
                model_status = "WARNING_STALE_MEDIAN"
            else:
                model_status = "OK" if not warnings else "WARNING"

            reasoning_parts = [
                f"Baseline rate started at ${baseline_rate:,.2f}.",
                f"Demand index {demand_index:+.2f} reflects occupancy, pace, seasonality, and event context.",
                f"Strategy index {strategy_index:+.2f} reflects RevPAR, rooms sold, and revenue priorities.",
                f"Property type applied a {property_preset['description']}.",
                f"Segment focus applied a {segment_preset['description']}.",
                f"Median source for this date: {median_rate_source.lower()}.",
            ]
            if pd.notna(median_rate_used):
                reasoning_parts.append(f"The tailored model used ${median_rate_used:,.2f} as the date-level median anchor.")
            else:
                reasoning_parts.append("No usable date-level or global median rate was available, so the tailored model stayed closer to baseline.")
            reasoning = " ".join(reasoning_parts)

        inputs_complete = pd.notna(baseline_rate) and (pd.notna(median_rate_used) or pd.notna(adr))
        confidence = _confidence_label(warnings, inputs_complete=bool(inputs_complete))
        if model_status == "OK" and confidence == "low":
            model_status = "WARNING"

        rows.append(
            {
                "stay_date": stay_date,
                "occupancy": occupancy,
                "adr": adr,
                "revpar": float(getattr(row, "revpar", np.nan)) if pd.notna(getattr(row, "revpar", np.nan)) else np.nan,
                "rooms_sold": float(getattr(row, "rooms_sold", 0.0) or 0.0),
                "room_revenue": float(getattr(row, "room_revenue", np.nan)) if pd.notna(getattr(row, "room_revenue", np.nan)) else np.nan,
                "baseline_recommendation": baseline_rate,
                "suggested_dataset_median_rate": suggested_dataset_median_rate,
                "manual_daily_median_rate": manual_daily_median_rate,
                "global_median_fallback": global_median_fallback,
                "median_rate_used": median_rate_used,
                "median_rate_source": median_rate_source,
                "difference_from_median_rate": (tailored_rate - median_rate_used) if pd.notna(median_rate_used) and pd.notna(tailored_rate) else np.nan,
                "tailored_recommendation": tailored_rate,
                "recommended_rate_adjustment": (tailored_rate - baseline_rate) if pd.notna(tailored_rate) and pd.notna(baseline_rate) else np.nan,
                "model_status": model_status,
                "confidence": confidence,
                "reasoning_notes": reasoning,
                "warnings": " | ".join(warnings),
                "last_median_update_timestamp": last_median_update_timestamp,
                "median_rate_last_updated": last_median_update_timestamp,
                "median_rate_update_frequency": validated_settings["median_rate_update_frequency"],
                "property_type": validated_settings["property_type"],
                "segment_focus": validated_settings["segment_focus"],
            }
        )

    return pd.DataFrame(rows, columns=output_columns).sort_values("stay_date").reset_index(drop=True)


def build_tailored_summary(results_df: pd.DataFrame, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Create a one-row summary export for tailored model results."""
    validated_settings, errors = validate_tailored_settings(settings)
    if errors:
        raise ValueError("Tailored model settings are invalid: " + "; ".join(errors))

    warning_count = int((results_df.get("warnings", pd.Series(dtype=str)).fillna("") != "").sum())
    source_series = results_df.get("median_rate_source", pd.Series(dtype=str)).fillna("")
    summary = {
        "property_type": validated_settings["property_type"],
        "segment_focus": validated_settings["segment_focus"],
        "global_median_rate_fallback": validated_settings.get("global_median_rate_fallback"),
        "comp_rate_input_mode": validated_settings["comp_rate_input_mode"],
        "median_rate": validated_settings.get("global_median_rate_fallback"),
        "median_rate_last_updated": validated_settings.get("median_rate_last_updated"),
        "median_rate_update_frequency": validated_settings["median_rate_update_frequency"],
        "rows": int(len(results_df)),
        "warning_rows": warning_count,
        "manual_daily_median_dates": int((source_series == MANUAL_DAILY_SOURCE).sum()),
        "dataset_derived_daily_median_dates": int((source_series == DATASET_DERIVED_SOURCE).sum()),
        "global_fallback_median_dates": int((source_series == GLOBAL_FALLBACK_SOURCE).sum()),
        "missing_median_dates": int((source_series == MISSING_MEDIAN_SOURCE).sum()),
        "avg_baseline_recommendation": float(pd.to_numeric(results_df.get("baseline_recommendation", pd.Series(dtype=float)), errors="coerce").mean()) if len(results_df) else np.nan,
        "avg_final_median_rate_used": float(pd.to_numeric(results_df.get("median_rate_used", pd.Series(dtype=float)), errors="coerce").mean()) if len(results_df) else np.nan,
        "avg_tailored_recommendation": float(pd.to_numeric(results_df.get("tailored_recommendation", pd.Series(dtype=float)), errors="coerce").mean()) if len(results_df) else np.nan,
        "avg_adjustment_amount": float(pd.to_numeric(results_df.get("recommended_rate_adjustment", pd.Series(dtype=float)), errors="coerce").mean()) if len(results_df) else np.nan,
        "avg_difference_from_median": float(pd.to_numeric(results_df.get("difference_from_median_rate", pd.Series(dtype=float)), errors="coerce").mean()) if len(results_df) else np.nan,
    }
    return pd.DataFrame([summary])
