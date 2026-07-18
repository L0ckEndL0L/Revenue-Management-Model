"""
pricing.py
Rule-based pricing recommendation engine for the Hotel RMS.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.elasticity import expected_rooms_sold


@dataclass(frozen=True)
class PricingConfig:
    """Configuration for rule-based pricing and guardrails."""

    high_threshold: float = 0.85
    low_threshold: float = 0.50
    base_increase_pct: float = 0.03
    base_decrease_pct: float = -0.03
    aggressive_increase_pct: float = 0.08
    moderate_decrease_pct: float = -0.06
    floor_rate: float = 40.0
    ceiling_rate: float = 600.0
    max_daily_change_pct: float = 0.12
    weekend_premium_min: float = 1.0
    weekend_premium_max: float = 1.15


def _clip_change(change_pct: float, max_abs_change: float) -> float:
    return float(np.clip(change_pct, -max_abs_change, max_abs_change))


def generate_rate_recommendations(
    pace_df: pd.DataFrame,
    config: PricingConfig,
) -> pd.DataFrame:
    """
    Generate recommended ADR values using occupancy/STLY and thresholds.

    Expects columns:
      stay_date, current_adr, occupancy, stly_occupancy, pace_variance, has_historical, event_pct
    """
    required_cols = {
        "stay_date",
        "current_adr",
        "occupancy",
        "stly_occupancy",
        "pace_variance",
        "has_historical",
        "event_pct",
    }
    missing = required_cols - set(pace_df.columns)
    if missing:
        raise ValueError(f"Missing required pricing columns: {sorted(missing)}")

    rows = []

    for row in pace_df.itertuples(index=False):
        current_adr = float(row.current_adr)
        occupancy = float(row.occupancy)
        stly_occupancy = (
            float(row.stly_occupancy) if pd.notna(row.stly_occupancy) else np.nan
        )
        pace_variance = float(row.pace_variance) if pd.notna(row.pace_variance) else np.nan
        has_historical = bool(row.has_historical)
        event_pct = float(row.event_pct)

        base_change = 0.0
        decision_tag = "hold"

        # Historical comparison rule if available.
        if has_historical and pd.notna(pace_variance):
            if pace_variance > 0:
                base_change += config.base_increase_pct
                decision_tag = "increase"
            elif pace_variance < 0:
                base_change += config.base_decrease_pct
                decision_tag = "decrease"

        # Occupancy threshold overrides/intensifies change.
        if occupancy >= config.high_threshold:
            base_change = max(base_change, config.aggressive_increase_pct)
            decision_tag = "aggressive_increase"
        elif occupancy <= config.low_threshold:
            base_change = min(base_change, config.moderate_decrease_pct)
            decision_tag = "moderate_decrease"
        elif not has_historical:
            # fallback logic if no historical benchmark
            base_change = 0.0
            decision_tag = "hold"

        # Apply event adjustment.
        total_change = base_change + event_pct

        # Guardrail: cap daily movement.
        capped_change = _clip_change(total_change, config.max_daily_change_pct)

        # Convert to ADR with guards.
        recommended = current_adr * (1.0 + capped_change)
        recommended = max(recommended, 0.0)
        recommended = max(recommended, config.floor_rate)
        recommended = min(recommended, config.ceiling_rate)

        # Reconcile realized pct after floor/ceiling.
        realized_change = 0.0
        if current_adr > 0:
            realized_change = (recommended / current_adr) - 1.0
            realized_change = _clip_change(realized_change, config.max_daily_change_pct)

        rows.append(
            {
                "stay_date": row.stay_date,
                "current_adr": current_adr,
                "recommended_adr": recommended,
                "occupancy": occupancy,
                "stly_occupancy": stly_occupancy,
                "pace_variance": pace_variance,
                "decision_tag": decision_tag,
                "applied_change_pct": realized_change,
            }
        )

    return pd.DataFrame(rows).sort_values("stay_date").reset_index(drop=True)


def _pace_signal(pace_variance: float | None) -> str:
    if pace_variance is None or pd.isna(pace_variance):
        return "unknown"
    if pace_variance > 0.01:
        return "ahead"
    if pace_variance < -0.01:
        return "behind"
    return "in_line"


def _event_signal(impact_level: str | None) -> str:
    if impact_level is None or pd.isna(impact_level):
        return "none"
    normalized = str(impact_level).strip().lower()
    if normalized in {"low", "medium", "high"}:
        return {"low": "low", "medium": "med", "high": "high"}[normalized]
    return "none"


def _confidence_label(signal_strength: float) -> str:
    if signal_strength >= 2.0:
        return "high"
    if signal_strength >= 1.0:
        return "med"
    return "low"


def simulate_elasticity_pricing(
    future_df: pd.DataFrame,
    config: PricingConfig,
    elasticity: float = 1.2,
    candidate_pct_steps: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15),
    budget_gap: float = 0.0,
    required_adr_remaining: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate candidate rates and choose revenue-maximizing recommendation per date.

    Expects columns:
      stay_date, rooms_available, rooms_sold, current_rate, forecast_rooms_sold,
      forecast_occ, pace_variance, impact_level
    """
    required_cols = {
        "stay_date",
        "rooms_available",
        "rooms_sold",
        "current_rate",
        "forecast_rooms_sold",
        "forecast_occ",
    }
    missing = required_cols - set(future_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for elasticity pricing: {sorted(missing)}")

    rows = []
    simulation_rows = []

    for row in future_df.sort_values("stay_date").itertuples(index=False):
        stay_date = row.stay_date
        rooms_available = float(row.rooms_available)
        on_books = float(row.rooms_sold)
        current_rate = max(float(row.current_rate), 0.01)
        base_demand = float(row.forecast_rooms_sold)
        forecast_occ = float(row.forecast_occ)
        pace_variance = getattr(row, "pace_variance", np.nan)
        impact_level = getattr(row, "impact_level", None)
        event_pct = float(getattr(row, "event_pct", 0.0) or 0.0)
        dow = pd.Timestamp(stay_date).dayofweek

        min_rate = max(config.floor_rate, current_rate * (1.0 - config.max_daily_change_pct))
        max_rate = min(config.ceiling_rate, current_rate * (1.0 + config.max_daily_change_pct))
        constraint_notes = []

        if dow in [4, 5]:
            weekend_min = current_rate * config.weekend_premium_min
            weekend_max = current_rate * config.weekend_premium_max
            min_rate = max(min_rate, weekend_min)
            max_rate = min(max_rate, weekend_max)
            constraint_notes.append("dow_band")

        best_rate = current_rate
        best_rooms = expected_rooms_sold(base_demand, current_rate, current_rate, elasticity, on_books, rooms_available)
        best_revenue = best_rate * best_rooms

        for pct in candidate_pct_steps:
            candidate_rate = current_rate * (1.0 + pct + event_pct)
            candidate_rate = float(np.clip(candidate_rate, min_rate, max_rate))

            applied_constraints = []
            if candidate_rate <= config.floor_rate + 1e-9:
                applied_constraints.append("floor")
            if candidate_rate >= config.ceiling_rate - 1e-9:
                applied_constraints.append("ceiling")
            if abs((candidate_rate / current_rate) - 1.0) >= config.max_daily_change_pct - 1e-6:
                applied_constraints.append("max_change")
            applied_constraints.extend(constraint_notes)

            expected_sold = expected_rooms_sold(
                base_demand=base_demand,
                candidate_rate=candidate_rate,
                current_rate=current_rate,
                elasticity=elasticity,
                on_books=on_books,
                rooms_available=rooms_available,
            )
            expected_revenue = candidate_rate * expected_sold

            simulation_rows.append(
                {
                    "stay_date": stay_date,
                    "candidate_rate": candidate_rate,
                    "expected_rooms_sold": expected_sold,
                    "expected_revenue": expected_revenue,
                    "constraints": "|".join(dict.fromkeys(applied_constraints)),
                }
            )

            if expected_revenue > best_revenue:
                best_rate = candidate_rate
                best_rooms = expected_sold
                best_revenue = expected_revenue

        baseline_rooms = expected_rooms_sold(base_demand, current_rate, current_rate, elasticity, on_books, rooms_available)
        baseline_revenue = baseline_rooms * current_rate
        uplift = best_revenue - baseline_revenue

        pace_signal = _pace_signal(pace_variance)
        event_signal = _event_signal(impact_level)
        budget_signal = f"gap=${budget_gap:,.0f}; required_adr=${required_adr_remaining:,.2f}"

        constraint_applied = []
        if best_rate <= config.floor_rate + 1e-9:
            constraint_applied.append("floor")
        if best_rate >= config.ceiling_rate - 1e-9:
            constraint_applied.append("ceiling")
        if abs((best_rate / current_rate) - 1.0) >= config.max_daily_change_pct - 1e-6:
            constraint_applied.append("max_change")
        constraint_applied.extend(constraint_notes)
        constraint_text = "|".join(dict.fromkeys(constraint_applied)) if constraint_applied else "none"

        signal_strength = (
            (1.0 if pace_signal in {"ahead", "behind"} else 0.0)
            + (1.0 if event_signal in {"med", "high"} else 0.0)
            + (1.0 if abs(uplift) > 50 else 0.0)
        )
        confidence = _confidence_label(signal_strength)

        explanation = (
            f"Signals: pace={pace_signal}, event={event_signal}, budget={budget_signal}. "
            f"Simulated candidates around current rate ${current_rate:,.2f} with elasticity {elasticity:.2f}. "
            f"Selected ${best_rate:,.2f} maximizing expected revenue (${best_revenue:,.2f}). "
            f"Constraints applied: {constraint_text}."
        )

        rows.append(
            {
                "stay_date": stay_date,
                "current_rate": current_rate,
                "recommended_rate": best_rate,
                "expected_rooms_sold_at_recommended": best_rooms,
                "expected_revenue_at_recommended": best_revenue,
                "uplift_vs_current": uplift,
                "pace_signal": pace_signal,
                "event_signal": event_signal,
                "budget_signal": budget_signal,
                "constraint_applied": constraint_text,
                "confidence": confidence,
                "explanation": explanation,
                "forecast_occ": forecast_occ,
                "on_books": on_books,
                "rooms_available": rooms_available,
            }
        )

    reco_columns = [
        "stay_date",
        "current_rate",
        "recommended_rate",
        "expected_rooms_sold_at_recommended",
        "expected_revenue_at_recommended",
        "uplift_vs_current",
        "pace_signal",
        "event_signal",
        "budget_signal",
        "constraint_applied",
        "confidence",
        "explanation",
        "forecast_occ",
        "on_books",
        "rooms_available",
    ]
    sim_columns = [
        "stay_date",
        "candidate_rate",
        "expected_rooms_sold",
        "expected_revenue",
        "constraints",
    ]

    reco_df = pd.DataFrame(rows)
    if len(reco_df) == 0:
        reco_df = pd.DataFrame(columns=reco_columns)
    else:
        reco_df = reco_df.sort_values("stay_date").reset_index(drop=True)

    sim_df = pd.DataFrame(simulation_rows)
    if len(sim_df) == 0:
        sim_df = pd.DataFrame(columns=sim_columns)
    else:
        sim_df = sim_df.sort_values(["stay_date", "candidate_rate"]).reset_index(drop=True)

    return reco_df, sim_df


def build_priority_lists(
    reco_df: pd.DataFrame,
    budget_gap: float = 0.0,
    target_occ: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build priority scoring and ranked raise/rescue/monitor lists."""
    if len(reco_df) == 0:
        empty = pd.DataFrame(columns=["stay_date", "priority_score"])
        return empty, empty, empty, empty

    df = reco_df.copy()
    df["compression_score"] = np.clip((df["forecast_occ"] - 0.80) * 5.0, 0.0, None)
    df["under_pacing_score"] = np.clip((target_occ - df["forecast_occ"]) * 5.0, 0.0, None)
    df["event_score"] = df["event_signal"].map({"none": 0.0, "low": 0.5, "med": 1.0, "high": 1.5}).fillna(0.0)
    if "budget_gap" in df.columns:
        row_budget_gap = pd.to_numeric(df["budget_gap"], errors="coerce").fillna(float(budget_gap))
        df["budget_score"] = np.clip(row_budget_gap.abs() / 50000.0, 0.0, 2.0)
    else:
        df["budget_score"] = np.clip(abs(float(budget_gap)) / 50000.0, 0.0, 2.0)

    df["priority_score"] = (
        1.2 * df["compression_score"]
        + 1.0 * df["under_pacing_score"]
        + 1.0 * df["event_score"]
        + 1.3 * df["budget_score"]
    )

    top_raise = df[df["uplift_vs_current"] > 0].sort_values("priority_score", ascending=False).head(15)
    top_rescue = df[df["forecast_occ"] < target_occ].sort_values("priority_score", ascending=False).head(15)
    top_monitor = df.sort_values("priority_score", ascending=False).head(15)

    cols = [
        "stay_date",
        "priority_score",
        "current_rate",
        "recommended_rate",
        "forecast_occ",
        "uplift_vs_current",
        "event_signal",
        "pace_signal",
        "confidence",
        "explanation",
    ]

    return (
        top_raise[cols].reset_index(drop=True),
        top_rescue[cols].reset_index(drop=True),
        top_monitor[cols].reset_index(drop=True),
        df.sort_values("stay_date").reset_index(drop=True),
    )
