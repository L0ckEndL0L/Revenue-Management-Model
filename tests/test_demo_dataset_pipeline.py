from __future__ import annotations

from pathlib import Path

from main import run_pipeline
from src.ingest import read_table_source
from src.schema import auto_map_columns


def test_demo_dataset_runs_through_pipeline(tmp_path) -> None:
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
