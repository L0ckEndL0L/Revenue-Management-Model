from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dataset_manager import list_budget_profiles, load_budget_profile, save_budget_profile
from ui.upload_panel import read_uploaded_table


def coerce_budget_dataframe(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None

    budget_df = df.copy()
    budget_df.columns = [str(c).strip().lower() for c in budget_df.columns]

    if {"stay_date", "budget_revenue"}.issubset(set(budget_df.columns)):
        budget_df = budget_df[["stay_date", "budget_revenue"]].copy()
        budget_df["stay_date"] = pd.to_datetime(budget_df["stay_date"], errors="coerce")
        budget_df["budget_revenue"] = pd.to_numeric(budget_df["budget_revenue"], errors="coerce")
        budget_df = budget_df.dropna(subset=["stay_date", "budget_revenue"]).copy()
        budget_df["stay_date"] = budget_df["stay_date"].dt.strftime("%Y-%m-%d")
        return budget_df.sort_values("stay_date").reset_index(drop=True)

    if {"year", "month", "budget_revenue"}.issubset(set(budget_df.columns)):
        budget_df = budget_df[["year", "month", "budget_revenue"]].copy()
        budget_df["year"] = pd.to_numeric(budget_df["year"], errors="coerce")
        budget_df["month"] = pd.to_numeric(budget_df["month"], errors="coerce")
        budget_df["budget_revenue"] = pd.to_numeric(budget_df["budget_revenue"], errors="coerce")
        budget_df = budget_df.dropna(subset=["year", "month", "budget_revenue"]).copy()
        budget_df["year"] = budget_df["year"].astype(int)
        budget_df["month"] = budget_df["month"].astype(int)
        budget_df = budget_df[(budget_df["month"] >= 1) & (budget_df["month"] <= 12)].copy()
        return budget_df.sort_values(["year", "month"]).reset_index(drop=True)

    return budget_df


def build_daily_budget_template(start_date: pd.Timestamp, end_date: pd.Timestamp, default_amount: float) -> pd.DataFrame:
    if end_date < start_date:
        end_date = start_date
    stay_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return pd.DataFrame(
        {
            "stay_date": stay_dates,
            "budget_revenue": [float(default_amount)] * len(stay_dates),
        }
    )


def render_budget_panel() -> tuple[pd.DataFrame | None, str]:
    st.subheader("Budget Input")
    budget_input_mode = st.radio(
        "Budget entry mode",
        options=["Upload spreadsheet", "Manual entry", "No budget"],
        horizontal=True,
        key="budget_input_mode",
    )

    if budget_input_mode == "Upload spreadsheet":
        budget_file = st.file_uploader("Budget file (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="budget_upload_file")
        if budget_file is not None:
            budget_preview = coerce_budget_dataframe(read_uploaded_table(budget_file))
            if budget_preview is None or len(budget_preview) == 0:
                st.error("Uploaded budget file is empty after cleaning.")
            else:
                st.session_state.budget_df = budget_preview
                st.caption(f"Loaded budget rows: {len(budget_preview)}")
        elif "budget_df" in st.session_state and st.session_state.budget_df is not None:
            st.caption("Using currently active budget data. Upload a new budget file to replace it.")

    elif budget_input_mode == "Manual entry":
        manual_budget_mode = st.radio(
            "Manual budget format",
            options=["Monthly target", "Daily targets"],
            horizontal=True,
            key="manual_budget_mode",
        )

        if manual_budget_mode == "Monthly target":
            now = pd.Timestamp.now()
            mb_col1, mb_col2, mb_col3 = st.columns(3)
            with mb_col1:
                budget_year = int(st.number_input("Budget year", min_value=2000, max_value=2100, value=int(now.year), step=1))
            with mb_col2:
                budget_month = int(st.number_input("Budget month", min_value=1, max_value=12, value=int(now.month), step=1))
            with mb_col3:
                budget_revenue = float(st.number_input("Monthly budget revenue", min_value=0.0, value=0.0, step=100.0))

            manual_monthly_budget = pd.DataFrame(
                [{"year": budget_year, "month": budget_month, "budget_revenue": budget_revenue}]
            )
            st.session_state.budget_df = coerce_budget_dataframe(manual_monthly_budget)
            st.caption("Manual monthly budget is active for this run.")

        else:
            today = pd.Timestamp.now().normalize()
            month_start = today.replace(day=1)
            month_end = month_start + pd.offsets.MonthEnd(0)

            db_col1, db_col2, db_col3 = st.columns(3)
            with db_col1:
                daily_start = pd.Timestamp(st.date_input("Daily budget start", value=month_start.date(), key="budget_daily_start"))
            with db_col2:
                daily_end = pd.Timestamp(st.date_input("Daily budget end", value=month_end.date(), key="budget_daily_end"))
            with db_col3:
                default_daily_budget = float(
                    st.number_input("Default daily budget", min_value=0.0, value=0.0, step=50.0, key="budget_daily_default")
                )

            if (
                "manual_budget_daily_df" not in st.session_state
                or st.button("Reset Daily Budget Grid", key="reset_manual_budget_daily")
            ):
                st.session_state.manual_budget_daily_df = build_daily_budget_template(
                    start_date=daily_start,
                    end_date=daily_end,
                    default_amount=default_daily_budget,
                )

            edited_manual_daily_budget = st.data_editor(
                st.session_state.manual_budget_daily_df,
                use_container_width=True,
                hide_index=True,
                key="manual_daily_budget_editor",
                column_config={
                    "stay_date": st.column_config.DateColumn("Stay Date", format="YYYY-MM-DD"),
                    "budget_revenue": st.column_config.NumberColumn("Budget Revenue", min_value=0.0, step=10.0, format="$%.2f"),
                },
            )
            st.session_state.manual_budget_daily_df = edited_manual_daily_budget
            st.session_state.budget_df = coerce_budget_dataframe(edited_manual_daily_budget)
            st.caption("Manual daily budget is active for this run.")

    else:
        if "budget_df" in st.session_state:
            del st.session_state["budget_df"]
        st.caption("Budget is disabled for this run.")

    active_budget_df = coerce_budget_dataframe(st.session_state.get("budget_df"))
    if active_budget_df is not None and len(active_budget_df) > 0:
        st.session_state.budget_df = active_budget_df
        with st.expander("Active budget preview", expanded=False):
            st.dataframe(active_budget_df.head(40), use_container_width=True)
            st.caption(f"Columns: {', '.join([str(c) for c in active_budget_df.columns])}")

    bp_col1, bp_col2 = st.columns(2)
    with bp_col1:
        st.markdown("**Load Saved Budget**")
        saved_budget_profiles = list_budget_profiles()
        if saved_budget_profiles:
            selected_budget_profile = st.selectbox(
                "Hotel budget profile",
                options=saved_budget_profiles,
                key="load_budget_profile_select",
            )
            if st.button("Load Budget Profile", key="load_budget_profile_btn", use_container_width=True):
                loaded_budget_profile = load_budget_profile(selected_budget_profile)
                normalized_budget_profile = coerce_budget_dataframe(loaded_budget_profile)
                if normalized_budget_profile is None or len(normalized_budget_profile) == 0:
                    st.error("Selected budget profile is empty or invalid.")
                else:
                    st.session_state.budget_df = normalized_budget_profile
                    st.success(f"Loaded budget profile: {selected_budget_profile}")
                    st.rerun()
        else:
            st.caption("No saved budget profiles yet.")

    with bp_col2:
        st.markdown("**Save Active Budget**")
        budget_profile_name = st.text_input(
            "Hotel / property name",
            key="save_budget_profile_name",
            placeholder="e.g., Downtown_Boston_Hotel",
        )
        if st.button("Save Budget Profile", key="save_budget_profile_btn", use_container_width=True):
            normalized_budget = coerce_budget_dataframe(st.session_state.get("budget_df"))
            if normalized_budget is None or len(normalized_budget) == 0:
                st.error("No active budget to save. Upload or enter a manual budget first.")
            elif not budget_profile_name.strip():
                st.error("Enter a hotel/property name before saving.")
            elif save_budget_profile(budget_profile_name, normalized_budget):
                st.success(f"Saved budget profile: {budget_profile_name}")
            else:
                st.error("Failed to save budget profile.")

    if st.button("Clear Active Budget", key="clear_active_budget_btn"):
        for key in ["budget_df", "manual_budget_daily_df"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    return active_budget_df, budget_input_mode
