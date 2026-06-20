from __future__ import annotations

from datetime import datetime
from typing import Dict

import pandas as pd
import streamlit as st

from src.ingest import read_table_source
from src.tailored import (
    ALLOWED_UPDATE_FREQUENCIES,
    DAILY_COMP_RATE_MODE,
    MONTHLY_COMP_RATE_MODE,
    build_daily_median_rate_table,
    default_tailored_settings,
    update_daily_median_rates,
    update_median_rate,
    validate_tailored_settings,
)


TAILORED_PROPERTY_TYPES = [
    "Full Service",
    "Limited Service",
    "Select Service",
    "Luxury",
    "Resort",
    "Boutique",
    "Extended Stay",
    "Economy",
]
TAILORED_SEGMENT_OPTIONS = [
    "Balanced",
    "Revenue",
    "Occupancy",
    "Corporate",
    "Group",
    "Premium",
    "Leisure",
]


def tailored_state_key(name: str) -> str:
    return f"tailored_{name}"


def tailored_pending_state_key() -> str:
    return "_tailored_pending_session_update"


def initialize_tailored_session(settings: Dict | None = None) -> None:
    defaults = default_tailored_settings()
    merged = {**defaults, **(settings or {})}
    for key, value in merged.items():
        state_key = tailored_state_key(key)
        normalized_value = "" if key in {"median_rate", "global_median_rate_fallback"} and value in (None, "") else value
        if settings is not None or state_key not in st.session_state:
            st.session_state[state_key] = normalized_value


def current_tailored_settings() -> Dict[str, object]:
    defaults = default_tailored_settings()
    current: Dict[str, object] = {}
    for key, default_value in defaults.items():
        current[key] = st.session_state.get(tailored_state_key(key), default_value)
    return current


def format_optional_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def format_optional_timestamp(value: str | None) -> str:
    if not value:
        return "Not updated yet"
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def queue_tailored_session_update(settings: Dict[str, object]) -> None:
    st.session_state[tailored_pending_state_key()] = dict(settings)


def apply_pending_tailored_session_update() -> None:
    pending = st.session_state.pop(tailored_pending_state_key(), None)
    if pending:
        initialize_tailored_session(pending)


def prepare_tailored_future_preview(raw_df: pd.DataFrame | None, mapping: Dict[str, str] | None) -> pd.DataFrame:
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame(columns=["stay_date"])

    mapping = mapping or {}
    candidate_columns = [
        "stay_date",
        "rooms_available",
        "rooms_sold",
        "room_revenue",
        "current_rate",
        "adr",
        "recommended_rate",
        "occupancy",
        "forecast_occ",
        "pace_variance",
        "event_pct",
        "impact_level",
    ]

    rename_map: Dict[str, str] = {}
    for canonical_name in candidate_columns:
        mapped_name = mapping.get(canonical_name)
        if mapped_name in raw_df.columns:
            rename_map[mapped_name] = canonical_name
        elif canonical_name in raw_df.columns:
            rename_map[canonical_name] = canonical_name

    preview = raw_df.rename(columns=rename_map).copy()
    available_columns = [column for column in candidate_columns if column in preview.columns]
    preview = preview[available_columns].copy() if available_columns else pd.DataFrame(columns=["stay_date"])
    if "stay_date" in preview.columns:
        preview["stay_date"] = pd.to_datetime(preview["stay_date"], errors="coerce")
        preview = preview.dropna(subset=["stay_date"]).sort_values("stay_date").reset_index(drop=True)
    return preview


def render_tailored_sidebar() -> None:
    st.header("Tailored Model")
    current_property_type = str(st.session_state.get(tailored_state_key("property_type"), TAILORED_PROPERTY_TYPES[0]))
    property_options = TAILORED_PROPERTY_TYPES if current_property_type in TAILORED_PROPERTY_TYPES else [current_property_type, *TAILORED_PROPERTY_TYPES]
    st.selectbox(
        "Property type",
        options=property_options,
        index=property_options.index(current_property_type),
        key=tailored_state_key("property_type"),
    )
    st.caption(
        "Property type changes the operating posture: luxury and boutique lean premium, resort responds more to seasonality/events, "
        "extended stay dampens volatility, and economy/limited service protect conversion."
    )

    current_segment_focus = str(st.session_state.get(tailored_state_key("segment_focus"), TAILORED_SEGMENT_OPTIONS[0]))
    segment_options = TAILORED_SEGMENT_OPTIONS if current_segment_focus in TAILORED_SEGMENT_OPTIONS else [current_segment_focus, *TAILORED_SEGMENT_OPTIONS]
    st.selectbox(
        "Segment focus",
        options=segment_options,
        index=segment_options.index(current_segment_focus),
        key=tailored_state_key("segment_focus"),
    )
    st.caption(
        "Segment focus now changes the tailored model posture: revenue and premium lean more rate-confident, "
        "occupancy and group protect volume, corporate dampens volatility, and leisure leans into seasonal/event demand."
    )

    tc1, tc2 = st.columns(2)
    with tc1:
        st.slider("Baseline occupancy sensitivity", min_value=0.0, max_value=2.0, step=0.1, key=tailored_state_key("baseline_occupancy_sensitivity"))
        st.slider("ADR sensitivity", min_value=0.0, max_value=2.0, step=0.1, key=tailored_state_key("adr_sensitivity"))
        st.slider("RevPAR priority", min_value=0.0, max_value=2.0, step=0.1, key=tailored_state_key("revpar_priority"))
        st.slider("Rooms sold priority", min_value=0.0, max_value=2.0, step=0.1, key=tailored_state_key("rooms_sold_priority"))
        st.slider("Revenue priority", min_value=0.0, max_value=2.0, step=0.1, key=tailored_state_key("revenue_priority"))
    with tc2:
        st.slider("Demand adjustment factor", min_value=0.0, max_value=2.0, step=0.1, key=tailored_state_key("demand_adjustment_factor"))
        st.slider("Seasonality adjustment factor", min_value=0.0, max_value=2.0, step=0.1, key=tailored_state_key("seasonality_adjustment_factor"))
        st.slider("Event or compression-night impact", min_value=0.0, max_value=2.0, step=0.1, key=tailored_state_key("event_impact_factor"))
        st.number_input("Minimum acceptable rate", min_value=0.0, step=1.0, key=tailored_state_key("minimum_acceptable_rate"))
        st.number_input("Maximum recommended rate", min_value=0.0, step=1.0, key=tailored_state_key("maximum_recommended_rate"))


def render_comp_rate_controls(tailored_future_preview: pd.DataFrame) -> None:
    st.subheader("Comp Rate Input")
    st.caption("Choose one monthly comp-set median rate or manage a separate comp rate by forecast date.")

    current_mode = str(st.session_state.get(tailored_state_key("comp_rate_input_mode"), DAILY_COMP_RATE_MODE))
    mode_options = [MONTHLY_COMP_RATE_MODE, DAILY_COMP_RATE_MODE]
    if current_mode not in mode_options:
        current_mode = DAILY_COMP_RATE_MODE

    st.radio(
        "Comp rate entry mode",
        options=mode_options,
        index=mode_options.index(current_mode),
        horizontal=True,
        key=tailored_state_key("comp_rate_input_mode"),
    )

    comp_set_file = st.file_uploader(
        "Optional comp-set rate shop (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        key="comp_set_upload_file",
    )
    if comp_set_file is not None:
        try:
            st.session_state.comp_set_df = read_table_source(comp_set_file, filename=comp_set_file.name)
            st.success(f"Loaded comp-set rows: {len(st.session_state.comp_set_df)}")
        except Exception as exc:
            st.error(f"Failed to load comp-set file: {exc}")

    comp_set_df = st.session_state.get("comp_set_df")
    if comp_set_df is not None and len(comp_set_df) > 0:
        with st.expander("Active comp-set preview", expanded=False):
            st.dataframe(comp_set_df.head(40), use_container_width=True)
        if {"stay_date", "rate"}.issubset(set(comp_set_df.columns)):
            st.caption("Comp-set medians from this table will be used as the suggested daily comp rates.")
        else:
            st.warning("Comp-set data needs stay_date and rate columns to drive suggested daily comp rates.")

    st.text_input(
        "Monthly comp-set median rate",
        key=tailored_state_key("global_median_rate_fallback"),
        placeholder="Used for the whole month or as the daily fallback",
    )
    st.caption("In monthly mode this value anchors every forecast date. In daily mode it is used only when a date-level comp rate is blank.")

    median_frequency_options = list(ALLOWED_UPDATE_FREQUENCIES.keys())
    current_frequency = str(st.session_state.get(tailored_state_key("median_rate_update_frequency"), "Manual only"))
    st.selectbox(
        "Median rate update frequency",
        options=median_frequency_options,
        index=median_frequency_options.index(current_frequency) if current_frequency in median_frequency_options else median_frequency_options.index("Manual only"),
        key=tailored_state_key("median_rate_update_frequency"),
    )
    st.caption(
        "Last comp rate update: "
        + format_optional_timestamp(st.session_state.get(tailored_state_key("median_rate_last_updated")))
    )

    median_col1, median_col2 = st.columns(2)
    with median_col1:
        if st.button("Clear Monthly Comp Rate", key="tailored_clear_global_fallback", use_container_width=True):
            refreshed = dict(current_tailored_settings())
            refreshed["global_median_rate_fallback"] = None
            refreshed["median_rate"] = None
            refreshed["median_rate_last_updated"] = None
            queue_tailored_session_update(refreshed)
            st.rerun()
    with median_col2:
        if st.button("Update Comp Rate Timestamp", key="tailored_update_median_timestamp", use_container_width=True):
            current_settings = current_tailored_settings()
            median_text = str(current_settings.get("global_median_rate_fallback", "")).strip()
            if not median_text:
                st.error("Enter a monthly comp-set median rate before updating its timestamp.")
            else:
                validated_settings, tailored_errors = validate_tailored_settings(current_settings)
                if tailored_errors:
                    st.error("Tailored Model settings: " + "; ".join(tailored_errors))
                else:
                    refreshed = update_median_rate(validated_settings, float(validated_settings["global_median_rate_fallback"]))
                    queue_tailored_session_update(refreshed)
                    st.rerun()

    if st.session_state.get(tailored_state_key("comp_rate_input_mode"), DAILY_COMP_RATE_MODE) == DAILY_COMP_RATE_MODE:
        render_daily_median_editor(tailored_future_preview, comp_set_df=comp_set_df)
    else:
        st.info("Daily comp-rate editing is off. The monthly comp-set median rate will be used for each forecast date.")


def render_daily_median_editor(tailored_future_preview: pd.DataFrame, comp_set_df: pd.DataFrame | None = None) -> None:
    if len(tailored_future_preview) == 0:
        st.info("Load demo data or upload a future on-books report to edit daily comp rates.")
        return

    st.subheader("Daily Comp Rates")
    st.caption("Edit comp-set median rates by forecast date. Blank manual values fall back to the dataset-derived daily rate or monthly comp-set median.")

    daily_median_df = build_daily_median_rate_table(
        tailored_future_preview,
        current_tailored_settings(),
        comp_set_df=comp_set_df,
    )
    daily_median_editor_df = daily_median_df.copy()
    if "stay_date" in daily_median_editor_df.columns:
        daily_median_editor_df["stay_date"] = pd.to_datetime(daily_median_editor_df["stay_date"], errors="coerce")

    edited_daily_median_df = st.data_editor(
        daily_median_editor_df,
        use_container_width=True,
        hide_index=True,
        key="tailored_daily_median_editor",
        disabled=[
            "stay_date",
            "suggested_dataset_median_rate",
            "global_median_fallback",
            "final_median_rate_used",
            "median_rate_source",
            "last_median_update_timestamp",
        ],
        column_config={
            "stay_date": st.column_config.DateColumn("Forecast date", format="YYYY-MM-DD"),
            "suggested_dataset_median_rate": st.column_config.NumberColumn("Suggested dataset comp rate", format="$%.2f"),
            "manual_daily_median_rate": st.column_config.NumberColumn("Manual daily comp rate", min_value=0.01, step=1.0, format="$%.2f"),
            "global_median_fallback": st.column_config.NumberColumn("Monthly comp-set median", format="$%.2f"),
            "final_median_rate_used": st.column_config.NumberColumn("Final median rate used", format="$%.2f"),
            "median_rate_source": "Comp-rate source",
            "last_median_update_timestamp": "Last updated timestamp",
        },
    )

    updated_tailored_settings = update_daily_median_rates(
        current_tailored_settings(),
        edited_daily_median_df,
    )
    st.session_state[tailored_state_key("daily_median_rates")] = updated_tailored_settings.get("daily_median_rates", [])
