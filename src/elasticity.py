"""
elasticity.py
Simple, explainable demand curve utilities for price-response simulation.
"""

from __future__ import annotations

import numpy as np


def expected_rooms_sold(
    base_demand: float,
    candidate_rate: float,
    current_rate: float,
    elasticity: float,
    on_books: float,
    rooms_available: float,
) -> float:
    """
    Compute expected rooms sold under a candidate rate.

    Demand curve:
      expected = base_demand * (candidate_rate / current_rate) ** (-elasticity)

    The result is bounded between on-books and available rooms.
    """
    base = max(float(base_demand), 0.0)
    rate = max(float(candidate_rate), 0.01)
    reference_rate = max(float(current_rate), 0.01)
    e = max(float(elasticity), 0.01)

    demand = base * ((rate / reference_rate) ** (-e))

    lower_bound = max(float(on_books), 0.0)
    upper_bound = max(float(rooms_available), 0.0)
    if upper_bound < lower_bound:
        upper_bound = lower_bound

    return float(np.clip(demand, lower_bound, upper_bound))
