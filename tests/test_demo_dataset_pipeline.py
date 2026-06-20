from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pandas as pd
import main as pipeline_main
import ui.dataset_panel as dataset_panel

from main import run_pipeline
from src.ingest import read_table_source
from src.schema import auto_map_columns
from ui.upload_panel import merge_saved_mapping_with_auto


def test_loaded_dataset_mapping_preserves_non_auto_columns() -> None:
    raw_df = read_table_source(Path(__file__).resolve().parents[1] / "data" / "future_on_books_sample.csv")
    raw_df = raw_df.rename(columns={"rooms_sold_to_date": "Booked Rooms"})

    mapping = merge_saved_mapping_with_auto(
        raw_df,
        {
            "stay_date": "stay_date",
            "rooms_available": "rooms_available",
            "rooms_sold": "Booked Rooms",
        },
    )

    assert mapping["rooms_sold"] == "Booked Rooms"
    assert mapping["current_rate"] == "current_rate"


def test_demo_payload_uses_project_relative_required_files() -> None:
    payload, warnings = dataset_panel.load_demo_dataset_payload()

    assert len(payload["historical_df"]) > 0
    assert len(payload["future_df"]) > 0
    assert {"stay_date", "rooms_available", "rooms_sold", "room_revenue"}.issubset(payload["historical_mapping"])
    assert {"stay_date", "rooms_available", "rooms_sold"}.issubset(payload["future_mapping"])
    assert not any("sample_data.csv" in warning for warning in warnings)


def test_demo_payload_warns_when_optional_files_are_missing(tmp_path, monkeypatch) -> None:
    source_data_dir = Path(__file__).resolve().parents[1] / "data"
    copyfile(source_data_dir / "sample_data.csv", tmp_path / "sample_data.csv")
    copyfile(source_data_dir / "future_on_books_sample.csv", tmp_path / "future_on_books_sample.csv")
    monkeypatch.setattr(dataset_panel, "DEMO_DATA_DIR", tmp_path)

    payload, warnings = dataset_panel.load_demo_dataset_payload()

    assert len(payload["historical_df"]) > 0
    assert len(payload["future_df"]) > 0
    assert payload["events_df"] is None
    assert payload["budget_df"] is None
    assert any("events_sample.csv" in warning for warning in warnings)
    assert any("budget_daily_sample.csv" in warning for warning in warnings)


def test_demo_dataset_runs_through_pipeline(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_main,
        "current_month_context",
        lambda: (2026, 6, pd.Timestamp("2026-06-20")),
    )

    data_dir = Path(__file__).resolve().parents[1] / "data"
    hist_df = read_table_source(data_dir / "sample_data.csv")
    future_df = read_table_source(data_dir / "future_on_books_sample.csv")
    events_df = read_table_source(data_dir / "events_sample.csv")
    budget_path = data_dir / "budget_daily_sample.csv"
    if not budget_path.exists():
        budget_path = data_dir / "budget_monthly_sample.csv"

    hist_path = tmp_path / "historical_loaded.csv"
    future_path = tmp_path / "future_loaded.csv"
    events_path = tmp_path / "events_loaded.csv"
    hist_df.to_csv(hist_path, index=False)
    future_df.to_csv(future_path, index=False)
    events_df.to_csv(events_path, index=False)

    output_paths, summary = run_pipeline(
        input_path=str(hist_path),
        future_path=str(future_path),
        events_path=str(events_path),
        budget_path=str(budget_path),
        config={
            "output_dir": str(tmp_path / "outputs"),
            "interactive": False,
            "column_mapping": auto_map_columns(hist_df),
            "future_column_mapping": auto_map_columns(future_df),
            "rate_floor": 99.0,
            "rate_ceiling": 399.0,
            "max_change_pct": 0.10,
            "high_threshold": 0.85,
            "low_threshold": 0.50,
            "elasticity": 1.2,
            "default_current_rate": 120.0,
            "manual_rooms_available": None,
            "tailored_settings": {},
        },
    )

    assert Path(output_paths["forecast"]).exists()
    assert "budget_summary" in summary
    assert "month_end_forecast" in summary["budget_summary"]
