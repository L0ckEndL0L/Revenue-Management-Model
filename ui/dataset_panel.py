from __future__ import annotations

import streamlit as st

from src.dataset_manager import delete_dataset, get_dataset_info, list_datasets, load_dataset, save_dataset
from src.tailored import validate_tailored_settings
from ui.tailored_panel import current_tailored_settings, initialize_tailored_session


def render_dataset_panel() -> None:
    st.header("Datasets")

    dataset_tab1, dataset_tab2, dataset_tab3 = st.tabs(["Load", "Save", "Manage"])

    with dataset_tab1:
        st.subheader("Load Dataset")
        saved_datasets = list_datasets()
        if saved_datasets:
            selected_dataset = st.selectbox(
                "Select saved dataset",
                options=saved_datasets,
                key="load_dataset_select",
            )

            if st.button("Load Dataset", key="load_dataset_btn"):
                hist_df, fut_df, events_df, budget_df, hist_map, fut_map, use_manual_rooms, manual_rooms, tailored_settings = load_dataset(selected_dataset)
                if hist_df is not None and fut_df is not None:
                    st.session_state.loaded_dataset_name = selected_dataset
                    st.session_state.historical_df = hist_df
                    st.session_state.future_df = fut_df
                    st.session_state.events_df = events_df
                    st.session_state.budget_df = budget_df
                    st.session_state.historical_mapping = hist_map
                    st.session_state.future_mapping = fut_map
                    st.session_state.use_manual_rooms_available = use_manual_rooms
                    st.session_state.manual_rooms_available = manual_rooms
                    initialize_tailored_session(tailored_settings)
                    st.session_state.load_dataset_success = True
                else:
                    st.error("Failed to load dataset")

            if "load_dataset_success" in st.session_state and st.session_state.load_dataset_success:
                dataset_info = get_dataset_info(selected_dataset)
                if dataset_info:
                    st.success(f"Loaded: {selected_dataset}")
                    st.caption(f"Created: {dataset_info.get('created_at', 'N/A')[:10]}")
                    st.caption(f"Rows: {dataset_info.get('rows_historical', 0)} hist / {dataset_info.get('rows_future', 0)} future")
        else:
            st.info("No saved datasets yet. Save one using the 'Save' tab.")

    with dataset_tab2:
        st.subheader("Save Current Dataset")
        dataset_name = st.text_input(
            "Dataset name",
            key="save_dataset_name",
            placeholder="e.g., Q2_2025_April",
        )

        if st.button("Save Dataset", key="save_dataset_btn"):
            if "historical_df" in st.session_state and "future_df" in st.session_state:
                tailored_settings, tailored_errors = validate_tailored_settings(current_tailored_settings())
                if tailored_errors:
                    st.error("Tailored Model settings: " + "; ".join(tailored_errors))
                else:
                    success = save_dataset(
                        name=dataset_name,
                        historical_df=st.session_state.historical_df,
                        future_df=st.session_state.future_df,
                        events_df=st.session_state.get("events_df"),
                        budget_df=st.session_state.get("budget_df"),
                        historical_mapping=st.session_state.get("historical_mapping"),
                        future_mapping=st.session_state.get("future_mapping"),
                        use_manual_rooms_available=st.session_state.get("use_manual_rooms_available", False),
                        manual_rooms_available=st.session_state.get("manual_rooms_available"),
                        tailored_settings=tailored_settings,
                    )
                    if success:
                        st.success(f"Saved dataset: {dataset_name}")
                    else:
                        st.error("Failed to save dataset")
            else:
                st.warning("Load or upload data first before saving")

    with dataset_tab3:
        st.subheader("Manage Datasets")
        saved_datasets = list_datasets()
        if saved_datasets:
            dataset_to_delete = st.selectbox(
                "Select dataset to delete",
                options=saved_datasets,
                key="delete_dataset_select",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Delete", key="delete_dataset_btn", use_container_width=True):
                    if delete_dataset(dataset_to_delete):
                        st.success(f"Deleted: {dataset_to_delete}")
                        st.rerun()
                    else:
                        st.error("Failed to delete dataset")

            with col2:
                if st.button("Info", key="info_dataset_btn", use_container_width=True):
                    info = get_dataset_info(dataset_to_delete)
                    if info:
                        st.json(info)

            st.divider()
            st.subheader("All Datasets")
            for ds_name in saved_datasets:
                info = get_dataset_info(ds_name)
                with st.expander(ds_name):
                    st.caption(f"Created: {info.get('created_at', 'N/A')[:10]}")
                    st.caption(f"Updated: {info.get('updated_at', 'N/A')[:10]}")
                    st.caption(f"Rows: {info.get('rows_historical', 0)} historical / {info.get('rows_future', 0)} future")
                    if info.get("has_events"):
                        st.caption("Has events data")
                    if info.get("has_budget"):
                        st.caption("Has budget data")
                    if info.get("has_tailored_settings"):
                        st.caption("Has tailored model settings")
        else:
            st.info("No datasets to manage yet")
