from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.dataset_manager as dataset_manager
from main import run_pipeline


def _historical_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_date": pd.to_datetime([
                "2026-05-20",
                "2026-05-21",
                "2026-05-22",
                "2026-05-23",
                "2026-05-24",
                "2026-05-25",
            ]),
            "rooms_available": [100, 100, 100, 100, 100, 100],
            "rooms_sold": [76, 82, 88, 91, 85, 79],
            "room_revenue": [9120.0, 10332.0, 11440.0, 12285.0, 10880.0, 9560.0],
            "current_rate": [120.0, 126.0, 130.0, 135.0, 128.0, 121.0],
        }
    )


def _future_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
            "rooms_available": [100, 100, 100],
            "rooms_sold": [72, 84, 20],
            "room_revenue": [9000.0, 10920.0, None],
            "current_rate": [125.0, 130.0, None],
        }
    )


def test_save_and_load_dataset_preserves_daily_median_settings(tmp_path, monkeypatch) -> None:
    datasets_dir = tmp_path / "datasets"
    metadata_file = datasets_dir / "datasets_metadata.json"
    monkeypatch.setattr(dataset_manager, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(dataset_manager, "METADATA_FILE", metadata_file)

    tailored_settings = {
        "property_type": "Luxury",
        "segment_focus": "Revenue",
        "global_median_rate_fallback": 155.0,
        "median_rate_update_frequency": "Every hour",
        "median_rate_last_updated": "2026-06-06T09:15:00",
        "daily_median_rates": [
            {
                "stay_date": "2026-07-01",
                "manual_daily_median_rate": 165.0,
                "last_median_update_timestamp": "2026-06-06T09:20:00",
            },
            {
                "stay_date": "2026-07-02",
                "manual_daily_median_rate": None,
                "last_median_update_timestamp": None,
            },
        ],
    }

    saved = dataset_manager.save_dataset(
        name="Tailored Dataset",
        historical_df=_historical_df(),
        future_df=_future_df(),
        tailored_settings=tailored_settings,
    )

    assert saved is True

    loaded = dataset_manager.load_dataset("Tailored Dataset")
    loaded_tailored_settings = loaded[-1]
    dataset_info = dataset_manager.get_dataset_info("Tailored Dataset")

    assert loaded_tailored_settings["global_median_rate_fallback"] == 155.0
    assert loaded_tailored_settings["daily_median_rates"][0]["manual_daily_median_rate"] == 165.0
    assert loaded_tailored_settings["daily_median_rates"][0]["stay_date"] == "2026-07-01"
    assert dataset_info["has_tailored_settings"] is True


def test_run_pipeline_creates_date_level_tailored_exports(tmp_path) -> None:
    hist_path = tmp_path / "historical.csv"
    future_path = tmp_path / "future.csv"
    _historical_df().to_csv(hist_path, index=False)
    _future_df().to_csv(future_path, index=False)

    output_paths, summary = run_pipeline(
        input_path=str(hist_path),
        future_path=str(future_path),
        config={
            "output_dir": str(tmp_path / "outputs"),
            "interactive": False,
            "allow_overbooking": False,
            "tailored_settings": {
                "property_type": "Resort",
                "segment_focus": "Revenue",
                "global_median_rate_fallback": 150.0,
                "median_rate_update_frequency": "Every 2 hours",
                "median_rate_last_updated": "2026-06-06T10:00:00",
                "minimum_acceptable_rate": 100.0,
                "maximum_recommended_rate": 250.0,
                "daily_median_rates": [
                    {
                        "stay_date": "2026-07-01",
                        "manual_daily_median_rate": 160.0,
                        "last_median_update_timestamp": "2026-06-06T10:05:00",
                    }
                ],
            },
        },
    )

    tailored_results_path = Path(output_paths["tailored_model_results"])
    tailored_summary_path = Path(output_paths["tailored_model_summary"])

    assert tailored_results_path.exists()
    assert tailored_summary_path.exists()

    tailored_results = pd.read_csv(tailored_results_path)
    tailored_summary = pd.read_csv(tailored_summary_path)

    assert {
        "stay_date",
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
        "reasoning_notes",
        "last_median_update_timestamp",
        "median_rate_update_frequency",
    }.issubset(tailored_results.columns)
    assert {
        "manual_daily_median_dates",
        "dataset_derived_daily_median_dates",
        "global_fallback_median_dates",
        "missing_median_dates",
        "avg_final_median_rate_used",
        "avg_tailored_recommendation",
        "avg_difference_from_median",
    }.issubset(tailored_summary.columns)
    assert "tailored_summary" in summary