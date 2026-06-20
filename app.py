"""
Streamlit UI for RateAnchor with forward-looking on-books pricing simulation.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

from main import run_pipeline
from src.tailored import validate_tailored_settings
from ui.budget_panel import coerce_budget_dataframe, render_budget_panel
from ui.dataset_panel import render_dataset_panel
from ui.results_panel import render_results
from ui.tailored_panel import (
    apply_pending_tailored_session_update,
    current_tailored_settings,
    initialize_tailored_session,
    prepare_tailored_future_preview,
    render_comp_rate_controls,
    render_tailored_sidebar,
)
from ui.upload_panel import render_upload_panel


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _is_running_in_streamlit():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(stcli.main())


def _save_uploaded_file(uploaded_file, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as out:
        out.write(uploaded_file.getbuffer())
    return destination


def _render_pricing_sidebar() -> dict:
    with st.expander("Advanced Pricing Settings", expanded=False):
        st.header("Pricing Controls")
        st.caption("Defaults are tuned for the demo. Adjust only when you want to show a sensitivity scenario.")
        use_manual_rooms_available = st.checkbox(
            "Manually set total rooms for all dates",
            value=st.session_state.get("use_manual_rooms_available", False),
        )
        manual_rooms_available = None
        if use_manual_rooms_available:
            manual_rooms_default = st.session_state.get("manual_rooms_available")
            if manual_rooms_default is None:
                manual_rooms_default = 100
            manual_rooms_available = int(
                st.number_input(
                    "Manual rooms available",
                    min_value=0,
                    value=int(manual_rooms_default),
                    step=1,
                )
            )

        st.session_state.use_manual_rooms_available = use_manual_rooms_available
        st.session_state.manual_rooms_available = manual_rooms_available

        return {
            "use_manual_rooms_available": use_manual_rooms_available,
            "manual_rooms_available": manual_rooms_available,
            "rate_floor": st.number_input("Rate floor", min_value=0.0, value=99.0, step=1.0),
            "rate_ceiling": st.number_input("Rate ceiling", min_value=0.0, value=399.0, step=1.0),
            "max_change_pct": st.slider("Max daily change %", min_value=0.0, max_value=50.0, value=10.0, step=0.5),
            "high_threshold": st.slider("High occupancy threshold %", min_value=50.0, max_value=100.0, value=85.0),
            "low_threshold": st.slider("Low occupancy threshold %", min_value=0.0, max_value=80.0, value=50.0),
            "elasticity": st.number_input("Elasticity", min_value=0.2, max_value=3.0, value=1.2, step=0.1),
            "default_current_rate": st.number_input("Default current rate fallback", min_value=0.0, value=120.0, step=1.0),
            "use_interactive_charts": st.checkbox("Use interactive charts", value=True),
            "output_base": st.text_input("Output folder", value="outputs"),
        }


def _run_dashboard() -> None:
    st.set_page_config(page_title="RateAnchor Pricing Simulation", layout="wide")

    apply_pending_tailored_session_update()
    initialize_tailored_session()

    st.title("RateAnchor")
    st.write(
        "A hotel revenue-management dashboard that cleans PMS-style data, forecasts demand, "
        "compares baseline and tailored pricing, and produces explainable rate recommendations."
    )
    st.markdown(
        """
        **Five-minute demo flow**
        1. Click **Load Demo Dataset** in the sidebar
        2. Review the loaded historical and future on-books previews
        3. Click **Run Pricing Simulation**
        4. Open the Forecast, Baseline, Tailored Model, Rate Recommendations, YoY, and Downloads sections
        """
    )

    with st.sidebar:
        render_dataset_panel()
        st.divider()
        pricing_config = _render_pricing_sidebar()

    setup_tab, comp_rate_tab = st.tabs(["Simulation Setup", "Comp Rate Controls"])
    with setup_tab:
        upload_state = render_upload_panel(pricing_config["use_manual_rooms_available"])
        with st.expander("Budget Settings", expanded=False):
            _, budget_input_mode = render_budget_panel()

    tailored_future_preview = prepare_tailored_future_preview(
        upload_state["fut_preview"],
        upload_state["future_mapping"] or st.session_state.get("future_mapping", {}),
    )
    with comp_rate_tab:
        render_tailored_sidebar()
        st.divider()
        render_comp_rate_controls(tailored_future_preview)

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        run_clicked = st.button("Run Pricing Simulation", type="primary", use_container_width=True)
    with action_col2:
        refresh_tailored_clicked = st.button("Refresh Tailored Recommendation", use_container_width=True)
    st.caption("Tailored refresh reuses the data already loaded in session, so you can update the median rate without re-importing files.")

    if not (run_clicked or refresh_tailored_clicked):
        return

    hist_preview = upload_state["hist_preview"]
    fut_preview = upload_state["fut_preview"]
    if hist_preview is None:
        st.error("Load the RateAnchor demo dataset or upload a historical PMS report first.")
        return
    if fut_preview is None:
        st.error("Load the RateAnchor demo dataset or upload a future on-books report first.")
        return

    try:
        tailored_settings, tailored_errors = validate_tailored_settings(current_tailored_settings())
        if tailored_errors:
            st.error("Tailored Model settings: " + "; ".join(tailored_errors))
            st.stop()

        workspace_root = Path(__file__).parent
        uploads_dir = workspace_root / "outputs" / "_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        historical_file = upload_state["historical_file"]
        if historical_file is not None:
            historical_path = _save_uploaded_file(historical_file, uploads_dir / f"{timestamp}_{historical_file.name}")
        else:
            historical_path = uploads_dir / f"{timestamp}_historical_loaded.csv"
            hist_preview.to_csv(historical_path, index=False)

        future_file = upload_state["future_file"]
        if future_file is not None:
            future_path = str(_save_uploaded_file(future_file, uploads_dir / f"{timestamp}_{future_file.name}"))
        else:
            future_path = str(uploads_dir / f"{timestamp}_future_loaded.csv")
            fut_preview.to_csv(future_path, index=False)

        events_path = None
        events_file = upload_state["events_file"]
        if events_file is not None:
            events_path = str(_save_uploaded_file(events_file, uploads_dir / f"{timestamp}_{events_file.name}"))
        elif "events_df" in st.session_state and st.session_state.events_df is not None:
            events_path = str(uploads_dir / f"{timestamp}_events_loaded.csv")
            st.session_state.events_df.to_csv(events_path, index=False)

        budget_path = None
        active_budget_for_run = coerce_budget_dataframe(st.session_state.get("budget_df"))
        if budget_input_mode != "No budget" and active_budget_for_run is not None and len(active_budget_for_run) > 0:
            budget_path = str(uploads_dir / f"{timestamp}_budget_active.csv")
            active_budget_for_run.to_csv(budget_path, index=False)

        progress = st.progress(0, text="Running pipeline")
        with st.status("Executing demand forecast and pricing simulation", expanded=True) as status:
            st.write("1/4 Ingest + validate data")
            progress.progress(20)
            st.write("2/4 Forecast future demand")
            progress.progress(45)
            st.write("3/4 Simulate candidate rates")
            progress.progress(70)
            st.write("4/4 Build priorities + evaluation")

            output_paths, summary = run_pipeline(
                input_path=str(historical_path),
                future_path=future_path,
                events_path=events_path,
                budget_path=budget_path,
                config={
                    "output_dir": pricing_config["output_base"],
                    "interactive": False,
                    "column_mapping": upload_state["historical_mapping"] or None,
                    "future_column_mapping": upload_state["future_mapping"] or None,
                    "rate_floor": float(pricing_config["rate_floor"]),
                    "rate_ceiling": float(pricing_config["rate_ceiling"]),
                    "max_change_pct": float(pricing_config["max_change_pct"] / 100.0),
                    "high_threshold": float(pricing_config["high_threshold"] / 100.0),
                    "low_threshold": float(pricing_config["low_threshold"] / 100.0),
                    "elasticity": float(pricing_config["elasticity"]),
                    "default_current_rate": float(pricing_config["default_current_rate"]),
                    "manual_rooms_available": pricing_config["manual_rooms_available"],
                    "tailored_settings": tailored_settings,
                },
            )
            status.update(label="Simulation complete", state="complete")

        progress.progress(100, text="Completed")
        st.success(f"Outputs created in: {output_paths['output_dir']}")
        render_results(
            output_paths=output_paths,
            summary=summary,
            use_interactive_charts=pricing_config["use_interactive_charts"],
            timestamp=timestamp,
        )
    except Exception as ex:
        st.error(f"Run failed: {ex}")


_run_dashboard()
