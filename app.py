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
from src.dataset_manager import (
    delete_dataset,
    get_dataset_info,
    list_datasets,
    load_dataset,
    save_dataset,
)
from src.schema import auto_map_columns
from src.tailored import (
    ALLOWED_UPDATE_FREQUENCIES,
    build_daily_median_rate_table,
    default_tailored_settings,
    update_daily_median_rates,
    update_median_rate,
    validate_tailored_settings,
)
from src.yoy import summarize_yoy


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


def _tailored_state_key(name: str) -> str:
    return f"tailored_{name}"


def _tailored_pending_state_key() -> str:
    return "_tailored_pending_session_update"


def _initialize_tailored_session(settings: Dict | None = None) -> None:
    defaults = default_tailored_settings()
    merged = {**defaults, **(settings or {})}
    for key, value in merged.items():
        state_key = _tailored_state_key(key)
        normalized_value = "" if key in {"median_rate", "global_median_rate_fallback"} and value in (None, "") else value
        if settings is not None or state_key not in st.session_state:
            st.session_state[state_key] = normalized_value


def _current_tailored_settings() -> Dict[str, object]:
    defaults = default_tailored_settings()
    current: Dict[str, object] = {}
    for key, default_value in defaults.items():
        current[key] = st.session_state.get(_tailored_state_key(key), default_value)
    return current


def _format_optional_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def _format_optional_timestamp(value: str | None) -> str:
    if not value:
        return "Not updated yet"
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def _queue_tailored_session_update(settings: Dict[str, object]) -> None:
    st.session_state[_tailored_pending_state_key()] = dict(settings)


def _apply_pending_tailored_session_update() -> None:
    pending = st.session_state.pop(_tailored_pending_state_key(), None)
    if pending:
        _initialize_tailored_session(pending)


_apply_pending_tailored_session_update()
_initialize_tailored_session()


def _prepare_tailored_future_preview(raw_df: pd.DataFrame | None, mapping: Dict[str, str] | None) -> pd.DataFrame:
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


def _build_yoy_outputs(
    output_paths: Dict[str, str],
    pipeline_yoy_summary: Dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    existing_yoy = _safe_read_csv(output_paths.get("yoy_comparison", ""))
    if existing_yoy is None or len(existing_yoy) == 0:
        return pd.DataFrame(), dict(pipeline_yoy_summary or {})

    return existing_yoy, dict(pipeline_yoy_summary or summarize_yoy(existing_yoy))


def _format_yoy_change(value: float | None, prefix: str = "", suffix: str = "%") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{prefix}{value:+.1f}{suffix}"


def _format_yoy_pair(current: float | None, prior: float | None, currency: bool = False, decimals: int = 1) -> str:
    if current is None or pd.isna(current) or prior is None or pd.isna(prior):
        return "No comparable prior-year data"

    if currency:
        return f"${current:,.{decimals}f} vs ${prior:,.{decimals}f}"
    return f"{current:.{decimals}f}% vs {prior:.{decimals}f}%"


def _mapping_ui(raw_df: pd.DataFrame, required: list[str], key_prefix: str) -> Dict[str, str]:
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


def _merge_saved_mapping_with_auto(raw_df: pd.DataFrame, saved_mapping: Dict[str, str] | None) -> Dict[str, str]:
    merged = dict(saved_mapping or {})
    auto_mapping = auto_map_columns(raw_df)
    for key, value in auto_mapping.items():
        merged.setdefault(key, value)
    return merged


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
    # ============================================================================
    # DATASET MANAGEMENT
    # ============================================================================
    st.header("Datasets")
    
    dataset_tab1, dataset_tab2, dataset_tab3 = st.tabs(["Load", "Save", "Manage"])
    
    # Load Dataset Tab
    with dataset_tab1:
        st.subheader("Load Dataset")
        saved_datasets = list_datasets()
        if saved_datasets:
            selected_dataset = st.selectbox(
                "Select saved dataset",
                options=saved_datasets,
                key="load_dataset_select"
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
                    _initialize_tailored_session(tailored_settings)
                    st.session_state.load_dataset_success = True
                else:
                    st.error("Failed to load dataset")
            
            if "load_dataset_success" in st.session_state and st.session_state.load_dataset_success:
                dataset_info = get_dataset_info(selected_dataset)
                if dataset_info:
                    st.success(f"✓ Loaded: {selected_dataset}")
                    st.caption(f"Created: {dataset_info.get('created_at', 'N/A')[:10]}")
                    st.caption(f"Rows: {dataset_info.get('rows_historical', 0)} hist / {dataset_info.get('rows_future', 0)} future")
        else:
            st.info("No saved datasets yet. Save one using the 'Save' tab.")
    
    # Save Dataset Tab
    with dataset_tab2:
        st.subheader("Save Current Dataset")
        dataset_name = st.text_input(
            "Dataset name",
            key="save_dataset_name",
            placeholder="e.g., Q2_2025_April"
        )
        
        if st.button("Save Dataset", key="save_dataset_btn"):
            if "historical_df" in st.session_state and "future_df" in st.session_state:
                tailored_settings, tailored_errors = validate_tailored_settings(_current_tailored_settings())
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
                        st.success(f"✓ Saved dataset: {dataset_name}")
                    else:
                        st.error("Failed to save dataset")
            else:
                st.warning("Load or upload data first before saving")
    
    # Manage Datasets Tab
    with dataset_tab3:
        st.subheader("Manage Datasets")
        saved_datasets = list_datasets()
        if saved_datasets:
            dataset_to_delete = st.selectbox(
                "Select dataset to delete",
                options=saved_datasets,
                key="delete_dataset_select"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Delete", key="delete_dataset_btn", use_container_width=True):
                    if delete_dataset(dataset_to_delete):
                        st.success(f"✓ Deleted: {dataset_to_delete}")
                        st.rerun()
                    else:
                        st.error("Failed to delete dataset")
            
            with col2:
                if st.button("ℹ️ Info", key="info_dataset_btn", use_container_width=True):
                    info = get_dataset_info(dataset_to_delete)
                    if info:
                        st.json(info)
            
            st.divider()
            st.subheader("All Datasets")
            for ds_name in saved_datasets:
                info = get_dataset_info(ds_name)
                with st.expander(f"📁 {ds_name}"):
                    st.caption(f"Created: {info.get('created_at', 'N/A')[:10]}")
                    st.caption(f"Updated: {info.get('updated_at', 'N/A')[:10]}")
                    st.caption(f"Rows: {info.get('rows_historical', 0)} historical / {info.get('rows_future', 0)} future")
                    if info.get('has_events'):
                        st.caption("✓ Has events data")
                    if info.get('has_budget'):
                        st.caption("✓ Has budget data")
                    if info.get('has_tailored_settings'):
                        st.caption("✓ Has tailored model settings")
        else:
            st.info("No datasets to manage yet")
    
    st.divider()
    
    # ============================================================================
    # PRICING CONFIGURATION
    # ============================================================================
    st.header("Pricing Configuration")
    use_manual_rooms_available = st.checkbox(
        "Manually set total rooms for all dates", 
        value=st.session_state.get("use_manual_rooms_available", False)
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
    
    # Update session state with current values
    st.session_state.use_manual_rooms_available = use_manual_rooms_available
    st.session_state.manual_rooms_available = manual_rooms_available

    rate_floor = st.number_input("Rate floor", min_value=0.0, value=99.0, step=1.0)
    rate_ceiling = st.number_input("Rate ceiling", min_value=0.0, value=399.0, step=1.0)
    max_change_pct = st.slider("Max daily change %", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    high_threshold = st.slider("High occupancy threshold %", min_value=50.0, max_value=100.0, value=85.0)
    low_threshold = st.slider("Low occupancy threshold %", min_value=0.0, max_value=80.0, value=50.0)
    elasticity = st.number_input("Elasticity", min_value=0.2, max_value=3.0, value=1.2, step=0.1)
    default_current_rate = st.number_input("Default current rate fallback", min_value=0.0, value=120.0, step=1.0)
    use_interactive_charts = st.checkbox("Use interactive charts", value=True)
    output_base = st.text_input("Output folder", value="outputs")

    st.divider()

    st.header("Tailored Model")
    current_property_type = str(st.session_state.get(_tailored_state_key("property_type"), TAILORED_PROPERTY_TYPES[0]))
    property_options = TAILORED_PROPERTY_TYPES if current_property_type in TAILORED_PROPERTY_TYPES else [current_property_type, *TAILORED_PROPERTY_TYPES]
    st.selectbox(
        "Property type",
        options=property_options,
        index=property_options.index(current_property_type),
        key=_tailored_state_key("property_type"),
    )

    current_segment_focus = str(st.session_state.get(_tailored_state_key("segment_focus"), TAILORED_SEGMENT_OPTIONS[0]))
    segment_options = TAILORED_SEGMENT_OPTIONS if current_segment_focus in TAILORED_SEGMENT_OPTIONS else [current_segment_focus, *TAILORED_SEGMENT_OPTIONS]
    st.selectbox(
        "Segment focus",
        options=segment_options,
        index=segment_options.index(current_segment_focus),
        key=_tailored_state_key("segment_focus"),
    )

    tc1, tc2 = st.columns(2)
    with tc1:
        st.slider("Baseline occupancy sensitivity", min_value=0.0, max_value=2.0, step=0.1, key=_tailored_state_key("baseline_occupancy_sensitivity"))
        st.slider("ADR sensitivity", min_value=0.0, max_value=2.0, step=0.1, key=_tailored_state_key("adr_sensitivity"))
        st.slider("RevPAR priority", min_value=0.0, max_value=2.0, step=0.1, key=_tailored_state_key("revpar_priority"))
        st.slider("Rooms sold priority", min_value=0.0, max_value=2.0, step=0.1, key=_tailored_state_key("rooms_sold_priority"))
        st.slider("Revenue priority", min_value=0.0, max_value=2.0, step=0.1, key=_tailored_state_key("revenue_priority"))
    with tc2:
        st.slider("Demand adjustment factor", min_value=0.0, max_value=2.0, step=0.1, key=_tailored_state_key("demand_adjustment_factor"))
        st.slider("Seasonality adjustment factor", min_value=0.0, max_value=2.0, step=0.1, key=_tailored_state_key("seasonality_adjustment_factor"))
        st.slider("Event or compression-night impact", min_value=0.0, max_value=2.0, step=0.1, key=_tailored_state_key("event_impact_factor"))
        st.number_input("Minimum acceptable rate", min_value=0.0, step=1.0, key=_tailored_state_key("minimum_acceptable_rate"))
        st.number_input("Maximum recommended rate", min_value=0.0, step=1.0, key=_tailored_state_key("maximum_recommended_rate"))

    st.text_input(
        "Global Median Rate Fallback",
        key=_tailored_state_key("global_median_rate_fallback"),
        placeholder="Used only when a date-level median is unavailable",
    )
    st.caption("This global value is only used when a forecast date does not have a manual or dataset-derived daily median.")

    median_frequency_options = list(ALLOWED_UPDATE_FREQUENCIES.keys())
    current_frequency = str(st.session_state.get(_tailored_state_key("median_rate_update_frequency"), "Manual only"))
    st.selectbox(
        "Median rate update frequency",
        options=median_frequency_options,
        index=median_frequency_options.index(current_frequency) if current_frequency in median_frequency_options else median_frequency_options.index("Manual only"),
        key=_tailored_state_key("median_rate_update_frequency"),
    )
    st.caption(
        "Last median rate update: "
        + _format_optional_timestamp(st.session_state.get(_tailored_state_key("median_rate_last_updated")))
    )

    median_col1, median_col2 = st.columns(2)
    with median_col1:
        if st.button("Clear Global Fallback", key="tailored_clear_global_fallback", use_container_width=True):
            refreshed = dict(_current_tailored_settings())
            refreshed["global_median_rate_fallback"] = None
            refreshed["median_rate"] = None
            refreshed["median_rate_last_updated"] = None
            _queue_tailored_session_update(refreshed)
            st.rerun()
    with median_col2:
        if st.button("Update Global Fallback Timestamp", key="tailored_update_median_timestamp", use_container_width=True):
            current_settings = _current_tailored_settings()
            median_text = str(current_settings.get("global_median_rate_fallback", "")).strip()
            if not median_text:
                st.error("Enter a global median fallback before updating its timestamp.")
            else:
                validated_settings, tailored_errors = validate_tailored_settings(current_settings)
                if tailored_errors:
                    st.error("Tailored Model settings: " + "; ".join(tailored_errors))
                else:
                    refreshed = update_median_rate(validated_settings, float(validated_settings["global_median_rate_fallback"]))
                    _queue_tailored_session_update(refreshed)
                    st.rerun()

st.subheader("Uploads")

# Check if dataset was loaded from session state
if "historical_df" in st.session_state and "future_df" in st.session_state:
    st.info(f"✓ Using loaded dataset: **{st.session_state.get('loaded_dataset_name', 'Loaded Dataset')}**")
    historical_file = None
    future_file = None
    events_file = None
    budget_file = None
    hist_preview = st.session_state.historical_df
    fut_preview = st.session_state.future_df
    
    # Fill missing mapping fields from auto-detection while preserving saved choices.
    historical_mapping = _merge_saved_mapping_with_auto(
        hist_preview,
        st.session_state.get("historical_mapping", {}),
    )
    future_mapping = _merge_saved_mapping_with_auto(
        fut_preview,
        st.session_state.get("future_mapping", {}),
    )
    
    # Update session state with completed mappings
    st.session_state.historical_mapping = historical_mapping
    st.session_state.future_mapping = future_mapping
    
    if st.button("Clear Loaded Dataset", key="clear_dataset_btn"):
        for key in ["historical_df", "future_df", "events_df", "budget_df", "loaded_dataset_name", 
                    "historical_mapping", "future_mapping", "load_dataset_success",
                    "use_manual_rooms_available", "manual_rooms_available", "tailored_daily_median_editor"]:
            if key in st.session_state:
                del st.session_state[key]
        for key in list(st.session_state.keys()):
            if key.startswith("tailored_"):
                del st.session_state[key]
        _initialize_tailored_session()
        st.rerun()
else:
    # Standard file upload
    historical_file = st.file_uploader("Historical PMS report (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    future_file = st.file_uploader("Future on-books report (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    events_file = st.file_uploader("Optional events.csv", type=["csv"])
    budget_file = st.file_uploader("Optional budget (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    
    hist_preview = None
    fut_preview = None
    historical_mapping: Dict[str, str] = {}
    future_mapping: Dict[str, str] = {}
    
    if historical_file is not None:
        hist_preview = _read_uploaded_table(historical_file)
        st.session_state.historical_df = hist_preview
        st.caption(f"Historical columns: {', '.join([str(c) for c in hist_preview.columns])}")
    
    if future_file is not None:
        fut_preview = _read_uploaded_table(future_file)
        st.session_state.future_df = fut_preview
        st.caption(f"Future columns: {', '.join([str(c) for c in fut_preview.columns])}")
    
    if events_file is not None:
        events_preview = _read_uploaded_table(events_file)
        st.session_state.events_df = events_preview
    
    if budget_file is not None:
        budget_preview = _read_uploaded_table(budget_file)
        st.session_state.budget_df = budget_preview

if hist_preview is not None:
    historical_required = ["stay_date", "rooms_sold", "room_revenue"]
    if not use_manual_rooms_available:
        historical_required.insert(1, "rooms_available")
    historical_mapping = _mapping_ui(hist_preview, historical_required, "historical")
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
    future_mapping = _mapping_ui(fut_preview, future_required, "future")
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

tailored_future_preview = _prepare_tailored_future_preview(
    fut_preview,
    future_mapping if "future_mapping" in locals() else st.session_state.get("future_mapping", {}),
)
if len(tailored_future_preview) > 0:
    st.subheader("Daily Median Rates")
    st.caption("Edit manual daily medians by forecast date. Blank manual values are allowed and will fall back to the dataset-derived or global median.")

    daily_median_df = build_daily_median_rate_table(
        tailored_future_preview,
        _current_tailored_settings(),
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
            "suggested_dataset_median_rate": st.column_config.NumberColumn("Suggested dataset median rate", format="$%.2f"),
            "manual_daily_median_rate": st.column_config.NumberColumn("Manual median rate", min_value=0.01, step=1.0, format="$%.2f"),
            "global_median_fallback": st.column_config.NumberColumn("Global median fallback", format="$%.2f"),
            "final_median_rate_used": st.column_config.NumberColumn("Final median rate used", format="$%.2f"),
            "median_rate_source": "Median-rate source",
            "last_median_update_timestamp": "Last updated timestamp",
        },
    )

    updated_tailored_settings = update_daily_median_rates(
        _current_tailored_settings(),
        edited_daily_median_df,
    )
    st.session_state[_tailored_state_key("daily_median_rates")] = updated_tailored_settings.get("daily_median_rates", [])

action_col1, action_col2 = st.columns(2)
with action_col1:
    run_clicked = st.button("Run Pricing Simulation", type="primary", use_container_width=True)
with action_col2:
    refresh_tailored_clicked = st.button("Refresh Tailored Recommendation", use_container_width=True)
st.caption("Tailored refresh reuses the data already loaded in session, so you can update the median rate without re-importing files.")

if run_clicked or refresh_tailored_clicked:
    if hist_preview is None:
        st.error("Upload or load a historical PMS report first.")
    elif fut_preview is None:
        st.error("Upload or load a future on-books report first.")
    else:
        try:
            tailored_settings, tailored_errors = validate_tailored_settings(_current_tailored_settings())
            if tailored_errors:
                st.error("Tailored Model settings: " + "; ".join(tailored_errors))
                st.stop()

            workspace_root = Path(__file__).parent
            uploads_dir = workspace_root / "outputs" / "_uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Handle file uploads (save them) or loaded datasets (use existing paths)
            if historical_file is not None:
                historical_path = _save_uploaded_file(historical_file, uploads_dir / f"{timestamp}_{historical_file.name}")
            else:
                # For loaded datasets, create a temporary CSV in uploads
                historical_path = uploads_dir / f"{timestamp}_historical_loaded.csv"
                hist_preview.to_csv(historical_path, index=False)

            future_path = None
            if future_file is not None:
                future_path = str(_save_uploaded_file(future_file, uploads_dir / f"{timestamp}_{future_file.name}"))
            else:
                # For loaded datasets, create a temporary CSV in uploads
                future_path = str(uploads_dir / f"{timestamp}_future_loaded.csv")
                fut_preview.to_csv(future_path, index=False)

            events_path = None
            if events_file is not None:
                events_path = str(_save_uploaded_file(events_file, uploads_dir / f"{timestamp}_{events_file.name}"))
            elif "events_df" in st.session_state and st.session_state.events_df is not None:
                events_path = str(uploads_dir / f"{timestamp}_events_loaded.csv")
                st.session_state.events_df.to_csv(events_path, index=False)

            budget_path = None
            if budget_file is not None:
                budget_path = str(_save_uploaded_file(budget_file, uploads_dir / f"{timestamp}_{budget_file.name}"))
            elif "budget_df" in st.session_state and st.session_state.budget_df is not None:
                budget_path = str(uploads_dir / f"{timestamp}_budget_loaded.csv")
                st.session_state.budget_df.to_csv(budget_path, index=False)

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
                        "tailored_settings": tailored_settings,
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
            st.caption(
                "Baseline model rows: "
                f"{summary.get('baseline_rows', 0)} "
                f"(unavailable: {summary.get('baseline_unavailable_rows', 0)})"
            )
            if summary.get("using_uploaded_comparison", False):
                st.caption("YoY and baseline comparison are using the two uploaded datasets as the prior-year/current-year pair.")

            tailored_summary = summary.get("tailored_summary", {})
            st.subheader("Tailored Model")
            t1, t2, t3 = st.columns(3)
            t1.metric("Avg final median used", _format_optional_currency(tailored_summary.get("avg_final_median_rate_used")))
            t2.metric("Manual daily median dates", f"{int(tailored_summary.get('manual_daily_median_dates', 0))}")
            t3.metric("Global fallback dates", f"{int(tailored_summary.get('global_fallback_median_dates', 0))}")
            st.caption(
                "Median update frequency: "
                f"{tailored_summary.get('median_rate_update_frequency', 'Manual only')}"
                + " | Last update: "
                + _format_optional_timestamp(tailored_summary.get("median_rate_last_updated"))
            )
            st.caption(
                "Daily median sources: "
                f"dataset-derived={int(tailored_summary.get('dataset_derived_daily_median_dates', 0))}, "
                f"missing={int(tailored_summary.get('missing_median_dates', 0))}"
            )

            # YoY panel
            yoy_df, yoy_summary = _build_yoy_outputs(
                output_paths,
                summary.get("yoy_summary", {}),
            )
            if len(yoy_df) > 0:
                st.subheader("Year-over-Year (YoY)")
                y1, y2, y3 = st.columns(3)
                y1.metric(
                    "Occupancy Change",
                    _format_yoy_change(yoy_summary.get("occupancy_change_pct")),
                    _format_yoy_pair(
                        yoy_summary.get("avg_current_occupancy_pct"),
                        yoy_summary.get("avg_stly_occupancy_pct"),
                    ),
                )
                y2.metric(
                    "ADR Change",
                    _format_yoy_change(yoy_summary.get("adr_change_pct")),
                    _format_yoy_pair(
                        yoy_summary.get("avg_current_adr"),
                        yoy_summary.get("avg_stly_adr"),
                        currency=True,
                        decimals=2,
                    ),
                )
                y3.metric(
                    "Revenue Change",
                    _format_yoy_change(yoy_summary.get("revenue_change_pct")),
                    _format_yoy_pair(
                        yoy_summary.get("total_current_revenue"),
                        yoy_summary.get("total_stly_revenue"),
                        currency=True,
                        decimals=0,
                    ),
                )
                st.caption(
                    "YoY row status: "
                    f"matched={yoy_summary.get('matched_rows', 0)}, "
                    f"missing prior-year={yoy_summary.get('missing_rows', 0)}, "
                    f"incomplete prior-year={yoy_summary.get('incomplete_rows', 0)}"
                )
                if not yoy_summary.get("has_comparable_data", False):
                    st.info("No comparable prior-year stay dates were found for this run, so YoY change values are shown as N/A.")
            else:
                st.info("YoY data not available for this run. Add comparable STLY files under data/historical to enable YoY.")

            # Tables
            st.subheader("Outputs")
            tabs = st.tabs([
                "Forecast",
                "Baseline",
                "Tailored Model",
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
                df = _safe_read_csv(output_paths.get("baseline_recommendations", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[2]:
                tailored_summary_df = _safe_read_csv(output_paths.get("tailored_model_summary", ""))
                tailored_results_df = _safe_read_csv(output_paths.get("tailored_model_results", ""))
                if tailored_summary_df is not None:
                    st.dataframe(tailored_summary_df, use_container_width=True)
                st.dataframe(tailored_results_df if tailored_results_df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[3]:
                df = _safe_read_csv(output_paths.get("rate_recommendations", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[4]:
                df = _safe_read_csv(output_paths.get("top_raise_opportunities", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[5]:
                df = _safe_read_csv(output_paths.get("top_rescue_dates", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[6]:
                df = _safe_read_csv(output_paths.get("top_monitor_dates", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[7]:
                df = _safe_read_csv(output_paths.get("evaluation_metrics", ""))
                st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
            with tabs[8]:
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
                "baseline_recommendations",
                "tailored_model_results",
                "tailored_model_summary",
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
