"""Safe retention cleanup for timestamped pipeline output directories."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
from pathlib import Path


RUN_DIRECTORY_PATTERN = re.compile(r"^run_\d{8}_\d{6}$")


def _clear_readonly_and_retry(function, path: str, _error) -> None:
    """Allow cleanup of read-only OneDrive placeholders and their contents."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def cleanup_old_run_directories(output_base: str | Path, keep_latest: int = 10) -> list[Path]:
    """Delete older timestamped run directories and return the removed paths."""
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")

    output_root = Path(output_base).expanduser().resolve()
    if not output_root.exists() or not output_root.is_dir():
        return []

    candidates: list[Path] = []
    for child in output_root.iterdir():
        if not child.is_dir() or not RUN_DIRECTORY_PATTERN.fullmatch(child.name):
            continue
        resolved_child = child.resolve()
        if resolved_child.parent != output_root:
            continue
        candidates.append(resolved_child)

    candidates.sort(key=lambda path: path.name, reverse=True)
    removed: list[Path] = []
    for old_run in candidates[keep_latest:]:
        try:
            shutil.rmtree(old_run, onexc=_clear_readonly_and_retry)
        except OSError:
            # A syncing or open file should not turn a completed model run into a failure.
            continue
        if not old_run.exists():
            removed.append(old_run)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove old timestamped RMS run directories safely.")
    parser.add_argument("--output", default="outputs", help="Output directory containing run_YYYYMMDD_HHMMSS folders")
    parser.add_argument("--keep", type=int, default=10, help="Number of newest completed runs to keep")
    args = parser.parse_args()
    removed = cleanup_old_run_directories(args.output, args.keep)
    print(f"Removed {len(removed)} old run directories; kept the newest {args.keep}.")


if __name__ == "__main__":
    main()
