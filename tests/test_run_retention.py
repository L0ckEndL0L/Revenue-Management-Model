from __future__ import annotations

from pathlib import Path

import pytest

from src.run_retention import cleanup_old_run_directories


def test_cleanup_old_run_directories_keeps_newest_and_ignores_other_folders(tmp_path: Path) -> None:
    run_names = [
        "run_20260101_010101",
        "run_20260102_010101",
        "run_20260103_010101",
        "run_20260104_010101",
    ]
    for name in run_names:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "result.csv").write_text("value\n1\n", encoding="utf-8")
    unrelated = tmp_path / "area_demand_mock_2026"
    unrelated.mkdir()

    removed = cleanup_old_run_directories(tmp_path, keep_latest=2)

    assert sorted(path.name for path in removed) == run_names[:2]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "area_demand_mock_2026",
        "run_20260103_010101",
        "run_20260104_010101",
    ]


def test_cleanup_old_run_directories_rejects_zero_retention(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        cleanup_old_run_directories(tmp_path, keep_latest=0)
