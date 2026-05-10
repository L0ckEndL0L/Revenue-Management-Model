from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.elasticity import expected_rooms_sold


def test_expected_rooms_sold_respects_bounds() -> None:
    value = expected_rooms_sold(
        base_demand=100,
        candidate_rate=200,
        current_rate=150,
        elasticity=1.2,
        on_books=50,
        rooms_available=80,
    )

    assert 50 <= value <= 80


def test_expected_rooms_sold_handles_small_rates() -> None:
    value = expected_rooms_sold(
        base_demand=20,
        candidate_rate=0,
        current_rate=0,
        elasticity=0,
        on_books=0,
        rooms_available=100,
    )

    assert value >= 0
