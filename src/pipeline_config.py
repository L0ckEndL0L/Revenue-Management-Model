"""Pipeline configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass

from src.baseline import BaselinePricingConfig
from src.pricing import PricingConfig


@dataclass(frozen=True)
class PipelineRuntimeConfig:
    allow_overbooking: bool
    interactive: bool
    elasticity: float
    default_current_rate: object
    manual_rooms_available: int | None
    pricing_config: PricingConfig
    baseline_config: BaselinePricingConfig


def build_pipeline_config(config: dict | None) -> PipelineRuntimeConfig:
    """Normalize user config into typed pricing/runtime objects."""
    config = config or {}
    manual_rooms_available = config.get("manual_rooms_available")
    if manual_rooms_available is not None:
        manual_rooms_available = int(manual_rooms_available)

    return PipelineRuntimeConfig(
        allow_overbooking=bool(config.get("allow_overbooking", False)),
        interactive=bool(config.get("interactive", False)),
        elasticity=float(config.get("elasticity", 1.2)),
        default_current_rate=config.get("default_current_rate"),
        manual_rooms_available=manual_rooms_available,
        pricing_config=PricingConfig(
            high_threshold=float(config.get("high_threshold", 0.85)),
            low_threshold=float(config.get("low_threshold", 0.50)),
            floor_rate=float(config.get("rate_floor", 99.0)),
            ceiling_rate=float(config.get("rate_ceiling", 399.0)),
            max_daily_change_pct=float(config.get("max_change_pct", 0.10)),
            weekend_premium_min=float(config.get("weekend_premium_min", 1.0)),
            weekend_premium_max=float(config.get("weekend_premium_max", 1.15)),
        ),
        baseline_config=BaselinePricingConfig(
            high_occupancy_threshold=float(config.get("baseline_high_threshold", 0.85)),
            low_occupancy_threshold=float(config.get("baseline_low_threshold", 0.55)),
            high_occupancy_increase_pct=float(config.get("baseline_high_increase_pct", 0.05)),
            low_occupancy_decrease_pct=float(config.get("baseline_low_decrease_pct", -0.05)),
            moderate_occupancy_change_pct=float(config.get("baseline_moderate_change_pct", 0.00)),
            rate_floor=float(config.get("rate_floor", 99.0)),
            rate_ceiling=float(config.get("rate_ceiling", 399.0)),
        ),
    )
