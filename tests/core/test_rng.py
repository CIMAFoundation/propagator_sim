from __future__ import annotations

import numpy as np

from propagator.core import BoundaryConditions, Propagator  # type: ignore


def run_simulation(seed: int | None, seconds: int = 4 * 3600) -> np.ndarray:
    size = 64
    propagator = Propagator(
        veg=np.full((size, size), 5, dtype=np.int32),
        dem=np.zeros((size, size), dtype=np.float32),
        realizations=4,
        do_spotting=False,
        seed=seed,
        out_of_bounds_mode="ignore",
    )
    propagator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=np.full((size, size), 10.0, dtype=np.float32),
            wind_dir=np.full((size, size), 45.0, dtype=np.float32),
            wind_speed=np.full((size, size), 10.0, dtype=np.float32),
            ignitions=[(size // 2, size // 2)],
        )
    )
    propagator.step(seconds=seconds)
    return propagator.get_arrival_time()


def test_same_seed_is_reproducible():
    first = run_simulation(seed=123)
    second = run_simulation(seed=123)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_diverge():
    first = run_simulation(seed=123)
    other = run_simulation(seed=987)
    assert (first != other).any()


def test_reseed_makes_rollback_deterministic():
    size = 64
    propagator = Propagator(
        veg=np.full((size, size), 5, dtype=np.int32),
        dem=np.zeros((size, size), dtype=np.float32),
        realizations=4,
        do_spotting=False,
        seed=42,
        out_of_bounds_mode="ignore",
    )
    propagator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=np.full((size, size), 10.0, dtype=np.float32),
            wind_dir=np.full((size, size), 45.0, dtype=np.float32),
            wind_speed=np.full((size, size), 10.0, dtype=np.float32),
            ignitions=[(size // 2, size // 2)],
        )
    )
    propagator.step(seconds=3600)
    checkpoint = propagator.checkpoint()

    propagator.reseed(7)
    propagator.step(seconds=3600)
    first = propagator.get_arrival_time()

    propagator.restore(checkpoint)
    propagator.reseed(7)
    propagator.step(seconds=3600)
    second = propagator.get_arrival_time()

    np.testing.assert_array_equal(first, second)
