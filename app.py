"""
Streamlit UI for RMS with forward-looking on-books pricing simulation.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st
from pandas.errors import ParserError

from main import run_pipeline
from src.pace import load_historical_data
from src.schema import auto_map_columns
from src.yoy import build_yoy_comparison, summarize_yoy


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


st.set_page_config(page_title="Hotel RMS Pricing Simulation", layout="wide")


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        def _try_read_csv_preview() -> pd.DataFrame:
            uploaded_file.seek(0)
            preview_df = pd.read_csv(uploaded_file)

            # Handle PMS exports where metadata lines appear before the real header row.
            first_col = str(preview_df.columns[0]).strip().lower()
            if first_col in {"start date:", "end date:"}:
                uploaded_file.seek(0)
                raw = pd.read_csv(uploaded_file, header=None)
                header_row = None
                scan_rows = min(20, len(raw))
                for i in range(scan_rows):
                    row_values = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
                    if "date" in row_values and ("room revenue" in row_values or "occupancy %" in row_values):
                        header_row = i
                        break
                if header_row is not None:
                    uploaded_file.seek(0)
                    preview_df = pd.read_csv(uploaded_file, skiprows=header_row)

            return preview_df

        uploaded_file.seek(0)
        try:
            return _try_read_csv_preview()
        except (ParserError, UnicodeDecodeError):
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, sep=None, engine="python")
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, engine="python", sep=None, quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    if suffix in [".xlsx", ".xls"]:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format. Upload CSV or XLSX.")


def _save_uploaded_file(uploaded_file, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as out:
        out.write(uploaded_file.getbuffer())
    return destination


def _build_zip_bytes(output_dir: Path) -> bytes:
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(output_dir))
    memory_file.seek(0)
    return memory_file.read()


def _safe_read_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _build_yoy_outputs(output_paths: Dict[str, str], workspace_root: Path) -> tuple[pd.DataFrame, dict]:
    daily_metrics_df = _safe_read_csv(output_paths.get("daily_metrics", ""))
    if daily_metrics_df is None or len(daily_metrics_df) == 0:
        return pd.DataFrame(), {}

    historical_dir = workspace_root / "data" / "historical"
    historical_df = load_historical_data(historical_dir)
    if historical_df is None or len(historical_df) == 0:
        return pd.DataFrame(), {}

    yoy_df = build_yoy_comparison(daily_metrics_df, historical_df)
    if len(yoy_df) == 0:
        return yoy_df, {}

    yoy_summary = summarize_yoy(yoy_df)
    yoy_path = Path(output_paths["output_dir"]) / "yoy_comparison.csv"
    yoy_df.to_csv(yoy_path, index=False)
    output_paths["yoy_comparison"] = str(yoy_path)

    return yoy_df, yoy_summary


def _mapping_ui(raw_df: pd.DataFrame, required: list[str], key_prefix: str) -> Dict[str, str]:
    mapping = auto_map_columns(raw_df)
    missing = [c for c in required if c not in mapping]
    selected: Dict[str, str] = {}

    if missing:
        st.warning(f"Missing required fields for {key_prefix}: {', '.join(missing)}")
        for field in missing:
            selected[field] = st.selectbox(
                f"Map {key_prefix} field '{field}'",
                options=list(raw_df.columns),
                key=f"{key_prefix}_{field}",
            )

    return selected


def _show_chart(chart_path: Path, caption: str) -> None:
    if chart_path.exists():
        st.image(str(chart_path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing chart: {caption}")


def _to_datetime(df: pd.DataFrame, col: str = "stay_date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _interactive_line_chart(
    df: pd.DataFrame,
    y_columns: list[str],
    title: str,
    y_title: str,
    x_col: str = "stay_date",
) -> alt.Chart:
    chart_df = _to_datetime(df, x_col)
    melted = chart_df.melt(id_vars=[x_col], value_vars=y_columns, var_name="series", value_name="value")
    base = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_col}:T", title="Stay Date"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("series:N", title="Series"),
            tooltip=[alt.Tooltip(f"{x_col}:T"), alt.Tooltip("series:N"), alt.Tooltip("value:Q", format=",.2f")],
        )
        .properties(title=title, height=320)
        .interactive()
    )
    return base


def _interactive_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_title: str,
) -> alt.Chart:
    chart_df = _to_datetime(df, x_col)
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:T", title="Stay Date"),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            tooltip=[alt.Tooltip(f"{x_col}:T"), alt.Tooltip(f"{y_col}:Q", format=",.2f")],
        )
        .properties(title=title, height=320)
        .interactive()
    )


st.title("Hotel RMS - Forward Pricing Simulator")
st.write("Upload historical + future on-books reports, run demand-aware pricing simulation, and review explainable recommendations.")

with st.sidebar:
    st.header("Pricing Configuration")
    use_manual_rooms_available = st.checkbox("Manually set total rooms for all dates", value=False)
    manual_rooms_available = None
    if use_manual_rooms_available:
        manual_rooms_available = int(
            st.number_input("Manual rooms available", min_value=0, value=100, step=1)
        )

    rate_floor = st.number_input("Rate floor", min_value=0.0, value=99.0, step=1.0)
    rate_ceiling = st.number_input("Rate ceiling", min_value=0.0, value=399.0, step=1.0)
    max_change_pct = st.slider("Max daily change %", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    high_threshold = st.slider("High occupancy threshold %", min_value=50.0, max_value=100.0, value=85.0)
    low_threshold = st.slider("Low occupancy threshold %", min_value=0.0, max_value=80.0, value=50.0)
    elasticity = st.number_input("Elasticity", min_value=0.2, max_value=3.0, value=1.2, step=0.1)
    default_current_rate = st.number_input("Default current rate fallback", min_value=0.0, value=120.0, step=1.0)
    use_interactive_charts = st.checkbox("Use interactive charts", value=True)
    output_base = st.text_input("Output folder", value="outputs")

st.subheader("Uploads")
historical_file = st.file_uploader("Historical PMS report (CSV/XLSX)", type=["csv", "xlsx", "xls"])
future_file = st.file_uploader("Future on-books report (CSV/XLSX)", type=["csv", "xlsx", "xls"])
events_file = st.file_uploader("Optional events.csv", type=["csv"])
budget_file = st.file_uploader("Optional budget (CSV/XLSX)", type=["csv", "xlsx", "xls"])

historical_mapping: Dict[str, str] = {}
future_mapping: Dict[str, str] = {}

if historical_file is not None:
    hist_preview = _read_uploaded_table(historical_file)
    st.caption(f"Historical columns: {', '.join([str(c) for c in hist_preview.columns])}")
    historical_required = ["stay_date", "rooms_sold", "room_revenue"]
    if not use_manual_rooms_available:
        historical_required.insert(1, "rooms_available")
    historical_mapping = _mapping_ui(hist_preview, historical_required, "historical")
    with st.expander("Historical preview", expanded=False):
        st.dataframe(hist_preview.head(40), use_container_width=True)

    with st.expander("Optional historical pricing mappings", expanded=False):
        optional_choices = ["(auto)"] + list(hist_preview.columns)
        hist_current_rate_choice = st.selectbox(
            "Map historical current_rate (optional)",
            options=optional_choices,
            key="hist_opt_current_rate",
        )
        hist_adr_choice = st.selectbox(
            "Map historical ADR (optional)",
            options=optional_choices,
            key="hist_opt_adr",
        )
        if hist_current_rate_choice != "(auto)":
            historical_mapping["current_rate"] = hist_current_rate_choice
        if hist_adr_choice != "(auto)":
            historical_mapping["adr"] = hist_adr_choice

if future_file is not None:
    fut_preview = _read_uploaded_table(future_file)
    st.caption(f"Future columns: {', '.join([str(c) for c in fut_preview.columns])}")
    future_required = ["stay_date", "rooms_sold"]
    if not use_manual_rooms_available:
        future_required.insert(1, "rooms_available")
    future_mapping = _mapping_ui(fut_preview, future_required, "future")
    with st.expander("Future preview", expanded=False):
        st.dataframe(fut_preview.head(40), use_container_width=True)

    with st.expander("Optional future pricing mappings", expanded=False):
        optional_choices = ["(auto)"] + list(fut_preview.columns)
        fut_current_rate_choice = st.selectbox(
            "Map future current_rate (optional)",
            options=optional_choices,
            key="fut_opt_current_rate",
        )
        fut_adr_choice = st.selectbox(
            "Map future ADR (optional)",
            options=optional_choices,
            key="fut_opt_adr",
        )
        if fut_current_rate_choice != "(auto)":
            future_mapping["current_rate"] = fut_current_rate_choice
        if fut_adr_choice != "(auto)":
            future_mapping["adr"] = fut_adr_choice

run_clicked = st.button("Run Pricing Simulation", type="primary")

if run_clicked:
    if historical_file is None:
        st.error("Upload a historical PMS report first.")
    else:
        try:
            workspace_root = Path(__file__).parent
            uploads_dir = workspace_root / "outputs" / "_uploads"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            historical_path = _save_uploaded_file(historical_file, uploads_dir / f"{timestamp}_{historical_file.name}")

            future_path = None
            if future_file is not None:
                future_path = str(_save_uploaded_file(future_file, uploads_dir / f"{timestamp}_{future_file.name}"))

            events_path = None
            if events_file is not None:
                events_path = str(_save_uploaded_file(events_file, uploads_dir / f"{timestamp}_{events_file.name}"))

            budget_path = None
            if budget_file is not None:
                budget_path = str(_save_uploaded_file(budget_file, uploads_dir / f"{timestamp}_{budget_file.name}"))

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
                        "output_dir": output_base,
                        "interactive": False,
                        "column_mapping": historical_mapping or None,
                        "future_column_mapping": future_mapping or None,
                        "rate_floor": float(rate_floor),
                        "rate_ceiling": float(rate_ceiling),
                        "max_change_pct": float(max_change_pct / 100.0),
                        "high_threshold": float(high_threshold / 100.0),
                        "low_threshold": float(low_threshold / 100.0),
                        "elasticity": float(elasticity),
                        "default_current_rate": float(default_current_rate),
                        "manual_rooms_available": manual_rooms_available,
                    },
                )
                status.update(label="Simulation complete", state="complete")

            progress.progress(100, text="Completed")
            st.success(f"Outputs created in: {output_paths['output_dir']}")

            # Summary panel
            budget_summary = summary.get("budget_summary", {})
            forecast_metrics = summary.get("forecast_metrics", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Revenue to date", f"${budget_summary.get('actual_revenue_to_date', 0.0):,.0f}")
            c2.metric("Month-end forecast", f"${budget_summary.get('month_end_forecast', 0.0):,.0f}")
            c3.metric("Budget variance", f"${budget_summary.get('variance_to_budget_abs', 0.0):,.0f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Required ADR remaining", f"${budget_summary.get('required_adr_remaining', 0.0):,.2f}")
            c5.metric("Forecast MAE", f"{forecast_metrics.get('mae', float('nan')):.2f}")
            c6.metric("Projected uplift", f"${summary.get('projected_uplift_vs_baseline', 0.0):,.0f}")

            # YoY panel
            yoy_df, yoy_summary = _build_yoy_outputs(output_paths, workspace_root)
            if len(yoy_df) > 0:
                st.subheader("Year-over-Year (YoY)")
                y1, y2, y3 = st.columns(3)
                y1.metric(
                    "Occupancy Change",
                    f"{yoy_summary.get('occupancy_change_pct', 0.0):+.1f}%",
                    f"{yoy_summary.get('avg_current_occupancy_pct', 0.0):.1f}% vs {yoy_summary.get('avg_stly_occupancy_pct', 0.0):.1f}%",
                )
                y2.metric(
                    "ADR Change",
                    f"{yoy_summary.get('adr_change_pct', 0.0):+.1f}%",
                    f"${yoy_summary.get('avg_current_adr', 0.0):,.2f} vs ${yoy_summary.get('avg_stly_adr', 0.0):,.2f}",
                )
                y3.metric(
                    "Revenue Change",
                    f"{yoy_summary.get('revenue_change_pct', 0.0):+.1f}%",
                    f"${yoy_summary.get('total_current_revenue', 0.0):,.0f} vs ${yoy_summary.get('total_stly_revenue', 0.0):,.0f}",
                )
            else:
                st.info("YoY data not available for this run. Add comparable STLY files under data/historical to enable YoY.")

            # Tables
            st.subheader("Outputs")
            tabs = st.tabs([
                "Forecast",
                "Rate Recommendations",
                "Top Raise",
                "Top Rescue",
                "Top Monitor",
                "Evaluation",
                "YoY",
            ])

            with tabs[0]:
                df = _safe_read_csv(output_paths.get("forecast", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[1]:
                df = _safe_read_csv(output_paths.get("rate_recommendations", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[2]:
                df = _safe_read_csv(output_paths.get("top_raise_opportunities", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[3]:
                df = _safe_read_csv(output_paths.get("top_rescue_dates", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[4]:
                df = _safe_read_csv(output_paths.get("top_monitor_dates", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[5]:
                df = _safe_read_csv(output_paths.get("evaluation_metrics", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[6]:
                st.dataframe(yoy_df if len(yoy_df) > 0 else pd.DataFrame(), use_container_width=True)

            # Charts
            st.subheader("Charts")
            run_dir = Path(output_paths["output_dir"])
            if use_interactive_charts:
                rate_df = _safe_read_csv(output_paths.get("rate_recommendations", ""))
                priority_df = _safe_read_csv(output_paths.get("top_monitor_dates", ""))
                forecast_vs_actual_df = _safe_read_csv(output_paths.get("forecast_vs_actual", ""))

                ch1, ch2 = st.columns(2)
                with ch1:
                    if rate_df is not None and {"stay_date", "current_rate", "recommended_rate"}.issubset(rate_df.columns):
                        st.altair_chart(
                            _interactive_line_chart(
                                rate_df,
                                y_columns=["current_rate", "recommended_rate"],
                                title="Current vs Recommended Rate",
                                y_title="Rate",
                            ),
                            use_container_width=True,
                        )
                    else:
                        _show_chart(run_dir / "current_vs_recommended_rate.png", "Current vs Recommended Rate")

                    if priority_df is not None and {"stay_date", "priority_score"}.issubset(priority_df.columns):
                        st.altair_chart(
                            _interactive_line_chart(
                                priority_df,
                                y_columns=["priority_score"],
                                title="Priority Score by Date",
                                y_title="Priority Score",
                            ),
                            use_container_width=True,
                        )
                    else:
                        _show_chart(run_dir / "priority_score_by_date.png", "Priority Score by Date")

                    if len(yoy_df) > 0 and {"calendar_date", "Current OCC %", "STLY OCC %"}.issubset(yoy_df.columns):
                        yoy_chart_df = yoy_df[["calendar_date", "Current OCC %", "STLY OCC %"]].copy()
                        yoy_melted = yoy_chart_df.melt(
                            id_vars=["calendar_date"],
                            value_vars=["Current OCC %", "STLY OCC %"],
                            var_name="series",
                            value_name="value",
                        )
                        yoy_chart = (
                            alt.Chart(yoy_melted)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("calendar_date:N", title="Calendar Date (MM-DD)"),
                                y=alt.Y("value:Q", title="Occupancy %"),
                                color=alt.Color("series:N", title="Series"),
                                tooltip=[
                                    alt.Tooltip("calendar_date:N", title="Calendar Date"),
                                    alt.Tooltip("series:N", title="Series"),
                                    alt.Tooltip("value:Q", title="Occupancy %", format=",.2f"),
                                ],
                            )
                            .properties(title="YoY Occupancy Comparison", height=320)
                            .interactive()
                        )
                        st.altair_chart(yoy_chart, use_container_width=True)

                with ch2:
                    if rate_df is not None and {"stay_date", "uplift_vs_current"}.issubset(rate_df.columns):
                        st.altair_chart(
                            _interactive_bar_chart(
                                rate_df,
                                x_col="stay_date",
                                y_col="uplift_vs_current",
                                title="Expected Revenue Uplift",
                                y_title="Uplift",
                            ),
                            use_container_width=True,
                        )
                    else:
                        _show_chart(run_dir / "expected_revenue_uplift.png", "Expected Revenue Uplift")

                    if forecast_vs_actual_df is not None and {
                        "stay_date",
                        "actual_rooms_sold",
                        "baseline_rooms_sold",
                        "enhanced_rooms_sold",
                    }.issubset(forecast_vs_actual_df.columns):
                        st.altair_chart(
                            _interactive_line_chart(
                                forecast_vs_actual_df,
                                y_columns=["actual_rooms_sold", "baseline_rooms_sold", "enhanced_rooms_sold"],
                                title="Forecast vs Actual (Rooms Sold)",
                                y_title="Rooms Sold",
                            ),
                            use_container_width=True,
                        )
                    else:
                        _show_chart(run_dir / "forecast_vs_actual.png", "Forecast vs Actual")
            else:
                ch1, ch2 = st.columns(2)
                with ch1:
                    _show_chart(run_dir / "current_vs_recommended_rate.png", "Current vs Recommended Rate")
                    _show_chart(run_dir / "priority_score_by_date.png", "Priority Score by Date")
                    if len(yoy_df) > 0:
                        st.info("YoY Occupancy Comparison is available when interactive charts are enabled.")
                with ch2:
                    _show_chart(run_dir / "expected_revenue_uplift.png", "Expected Revenue Uplift")
                    _show_chart(run_dir / "forecast_vs_actual.png", "Forecast vs Actual")

            # Downloads
            st.subheader("Downloads")
            for key in [
                "forecast",
                "rate_recommendations",
                "top_raise_opportunities",
                "top_rescue_dates",
                "top_monitor_dates",
                "evaluation_metrics",
                "baseline_vs_new_policy",
                "yoy_comparison",
            ]:
                file_path = output_paths.get(key)
                if file_path and Path(file_path).exists():
                    st.download_button(
                        f"Download {Path(file_path).name}",
                        data=Path(file_path).read_bytes(),
                        file_name=Path(file_path).name,
                        mime="text/csv",
                        key=f"download_{key}",
                    )

            zip_bytes = _build_zip_bytes(Path(output_paths["output_dir"]))
            st.download_button(
                "Download all outputs (zip)",
                data=zip_bytes,
                file_name=f"hotel_rms_outputs_{timestamp}.zip",
                mime="application/zip",
            )

        except Exception as ex:
            st.error(f"Run failed: {ex}")
