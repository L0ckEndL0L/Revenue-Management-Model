"""
explain.py
Explainability helpers and Week 3 output writers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable explanation text for each recommendation."""
    result = df.copy()

    def _explain(row: pd.Series) -> str:
        parts = []

        if row.get("decision_tag") == "aggressive_increase":
            parts.append(
                f"occupancy {row['occupancy'] * 100:.1f}% above high threshold"
            )
        elif row.get("decision_tag") == "moderate_decrease":
            parts.append(
                f"occupancy {row['occupancy'] * 100:.1f}% below low threshold"
            )
        elif row.get("has_historical"):
            # Historical data was matched for this date
            pv = row.get("pace_variance", 0.0)
            if abs(pv) < 0.001:  # Essentially zero
                parts.append("occupancy matched STLY")
            else:
                direction = "above" if pv > 0 else "below"
                parts.append(f"occupancy {abs(pv * 100):.1f}% {direction} STLY")
        else:
            parts.append("no historical benchmark; occupancy-threshold fallback")

        if pd.notna(row.get("event_name")) and row.get("event_name") != "":
            event_pct = row.get("event_pct", 0.0) * 100
            parts.append(
                f"event '{row['event_name']}' added {event_pct:.1f}% adjustment"
            )

        change_pct = row.get("applied_change_pct", 0.0) * 100
        direction_word = "Increased" if change_pct > 0 else "Decreased" if change_pct < 0 else "Held"

        return f"{direction_word} {abs(change_pct):.1f}% due to " + " and ".join(parts) + "."

    result["explanation"] = result.apply(_explain, axis=1)
    return result


def export_recommendations_csv(df: pd.DataFrame, output_path: str) -> None:
    """Export Week 3 recommendation CSV with required columns and formatting."""
    export_cols = [
        "stay_date",
        "current_adr",
        "recommended_adr",
        "occupancy",
        "stly_occupancy",
        "pace_variance",
        "event_name",
        "explanation",
    ]

    out = df[export_cols].copy()
    out["stay_date"] = pd.to_datetime(out["stay_date"]).dt.strftime("%Y-%m-%d")
    out["current_adr"] = out["current_adr"].round(2)
    out["recommended_adr"] = out["recommended_adr"].round(2)
    out["occupancy"] = out["occupancy"].round(4)
    out["stly_occupancy"] = out["stly_occupancy"].round(4)
    out["pace_variance"] = out["pace_variance"].round(4)

    out.to_csv(output_path, index=False)


def export_pricing_summary(summary: dict, output_path: str) -> None:
    """Write pricing summary text report."""
    content = (
        "HOTEL RMS - WEEK 3 PRICING SUMMARY\n"
        "===================================\n"
        f"total dates analyzed: {summary['total_dates_analyzed']}\n"
        f"number of increases: {summary['number_of_increases']}\n"
        f"number of decreases: {summary['number_of_decreases']}\n"
        f"avg % change: {summary['avg_pct_change'] * 100:.2f}%\n"
    )
    Path(output_path).write_text(content, encoding="utf-8")


def plot_adr_vs_recommendation(df: pd.DataFrame, output_path: str) -> None:
    """Plot current ADR vs recommended ADR by stay_date."""
    plot_df = df.sort_values("stay_date").copy()

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["stay_date"], plot_df["current_adr"], label="Current ADR", linewidth=2)
    plt.plot(
        plot_df["stay_date"],
        plot_df["recommended_adr"],
        label="Recommended ADR",
        linewidth=2,
        linestyle="--",
    )
    plt.title("Current ADR vs Recommended ADR")
    plt.xlabel("Stay Date")
    plt.ylabel("ADR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
