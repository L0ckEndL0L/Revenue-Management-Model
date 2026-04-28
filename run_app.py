"""
run_app.py
Entry point for the Hotel Revenue Management System (RMS) Streamlit application.

Usage:
    python run_app.py

This script launches the interactive Streamlit web interface.  It works as a
convenient single-command alternative to calling ``streamlit run app.py``
directly from the terminal.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit UI for the Hotel RMS application."""
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        print(f"[ERROR] Could not find app.py at: {app_path}")
        sys.exit(1)

    print("Starting Hotel RMS Streamlit application...")
    print(f"  App:  {app_path}")
    print("  Open the URL shown below in your browser.\n")

    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        cwd=str(app_path.parent),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
