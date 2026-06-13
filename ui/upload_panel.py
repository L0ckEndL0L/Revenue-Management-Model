from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from src.ingest import read_table_source
from src.schema import auto_map_columns
from ui.tailored_panel import initialize_tailored_session


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    return read_table_source(uploaded_file, filename=uploaded_file.name)


def mapping_ui(raw_df: pd.DataFrame, required: list[str], key_prefix: str) -> Dict[str, str]:
    mapping = auto_map_columns(raw_df)
    missing = [c for c in required if c not in mapping]

    if missing:
        st.warning(f"Missing required fields for {key_prefix}: {', '.join(missing)}")
        for field in missing:
            mapping[field] = st.selectbox(
                f"Map {key_prefix} field '{field}'",
                options=list(raw_df.columns),
                key=f"{key_prefix}_{field}",
            )

    return mapping


def merge_saved_mapping_with_auto(raw_df: pd.DataFrame, saved_mapping: Dict[str, str] | None) -> Dict[str, str]:
    merged = dict(saved_mapping or {})
    auto_mapping = auto_map_columns(raw_df)
    for key, value in auto_mapping.items():
        merged.setdefault(key, value)
    return merged


def render_upload_panel(use_manual_rooms_available: bool) -> dict:
    st.subheader("Uploads")

    dataset_loaded_mode = (
        bool(st.session_state.get("load_dataset_success", False))
        and bool(st.session_state.get("loaded_dataset_name"))
        and "historical_df" in st.session_state
        and "future_df" in st.session_state
    )

    historical_file = None
    future_file = None
    events_file = None
    hist_preview = None
    fut_preview = None
    historical_mapping: Dict[str, str] = {}
    future_mapping: Dict[str, str] = {}

    if dataset_loaded_mode:
        st.info(f"Using loaded dataset: **{st.session_state.get('loaded_dataset_name', 'Loaded Dataset')}**")
        hist_preview = st.session_state.historical_df
        fut_preview = st.session_state.future_df

        historical_mapping = merge_saved_mapping_with_auto(
            hist_preview,
            st.session_state.get("historical_mapping", {}),
        )
        future_mapping = merge_saved_mapping_with_auto(
            fut_preview,
            st.session_state.get("future_mapping", {}),
        )

        st.session_state.historical_mapping = historical_mapping
        st.session_state.future_mapping = future_mapping

        if st.button("Clear Loaded Dataset", key="clear_dataset_btn"):
            for key in [
                "historical_df",
                "future_df",
                "events_df",
                "budget_df",
                "loaded_dataset_name",
                "historical_mapping",
                "future_mapping",
                "load_dataset_success",
                "use_manual_rooms_available",
                "manual_rooms_available",
                "tailored_daily_median_editor",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            for key in list(st.session_state.keys()):
                if key.startswith("tailored_"):
                    del st.session_state[key]
            initialize_tailored_session()
            st.rerun()
    else:
        historical_file = st.file_uploader("Historical PMS report (CSV/XLSX)", type=["csv", "xlsx", "xls"])
        future_file = st.file_uploader("Future on-books report (CSV/XLSX)", type=["csv", "xlsx", "xls"])
        events_file = st.file_uploader("Optional events.csv", type=["csv"])

        if historical_file is not None:
            hist_preview = read_uploaded_table(historical_file)
            st.session_state.historical_df = hist_preview
            st.session_state.load_dataset_success = False
            st.session_state.pop("loaded_dataset_name", None)
            st.caption(f"Historical columns: {', '.join([str(c) for c in hist_preview.columns])}")

        if future_file is not None:
            fut_preview = read_uploaded_table(future_file)
            st.session_state.future_df = fut_preview
            st.session_state.load_dataset_success = False
            st.session_state.pop("loaded_dataset_name", None)
            st.caption(f"Future columns: {', '.join([str(c) for c in fut_preview.columns])}")

        if events_file is not None:
            events_preview = read_uploaded_table(events_file)
            st.session_state.events_df = events_preview

    if hist_preview is not None:
        historical_required = ["stay_date", "rooms_sold", "room_revenue"]
        if not use_manual_rooms_available:
            historical_required.insert(1, "rooms_available")
        historical_mapping = mapping_ui(hist_preview, historical_required, "historical")
        with st.expander("Historical preview", expanded=False):
            st.dataframe(hist_preview.head(40), use_container_width=True)

        with st.expander("Optional historical pricing mappings", expanded=False):
            optional_choices = ["(auto)"] + list(hist_preview.columns)
            hist_adr_choice = st.selectbox(
                "Map historical ADR (optional if room revenue is already mapped)",
                options=optional_choices,
                key="hist_opt_adr",
            )
            if hist_adr_choice != "(auto)":
                historical_mapping["adr"] = hist_adr_choice
        st.session_state.historical_mapping = historical_mapping

    if fut_preview is not None:
        st.caption(f"Future columns: {', '.join([str(c) for c in fut_preview.columns])}")
        future_required = ["stay_date", "rooms_sold"]
        if not use_manual_rooms_available:
            future_required.insert(1, "rooms_available")
        future_mapping = mapping_ui(fut_preview, future_required, "future")
        with st.expander("Future preview", expanded=False):
            st.dataframe(fut_preview.head(40), use_container_width=True)

        with st.expander("Optional future pricing mappings", expanded=False):
            optional_choices = ["(auto)"] + list(fut_preview.columns)
            fut_revenue_choice = st.selectbox(
                "Map future room revenue / total revenue (recommended for YoY, ADR, and baseline comparison)",
                options=optional_choices,
                key="fut_opt_room_revenue",
            )
            fut_current_rate_choice = st.selectbox(
                "Map future current_rate / sell rate (optional for pricing engine)",
                options=optional_choices,
                key="fut_opt_current_rate",
            )
            fut_adr_choice = st.selectbox(
                "Map future ADR (optional if revenue is already mapped)",
                options=optional_choices,
                key="fut_opt_adr",
            )
            if fut_revenue_choice != "(auto)":
                future_mapping["room_revenue"] = fut_revenue_choice
            if fut_current_rate_choice != "(auto)":
                future_mapping["current_rate"] = fut_current_rate_choice
            if fut_adr_choice != "(auto)":
                future_mapping["adr"] = fut_adr_choice
        st.session_state.future_mapping = future_mapping

    return {
        "historical_file": historical_file,
        "future_file": future_file,
        "events_file": events_file,
        "hist_preview": hist_preview,
        "fut_preview": fut_preview,
        "historical_mapping": historical_mapping,
        "future_mapping": future_mapping,
    }
