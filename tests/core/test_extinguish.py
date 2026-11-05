"""`extinguish` behaves the same on both cores: certain and stochastic
success, reproducibility, boundary conditions and checkpoints."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from propagator.core import BoundaryConditions, Propagator  # type: ignore
from propagator.core.constants import (  # type: ignore
    FUEL_SYSTEM_LEGACY_DICT,
)

SIZE = 48
CORES = ["numba", "rust"]


def make_simulator(core: str, realizations: int = 8, seed: int = 7):
    veg = np.full((SIZE, SIZE), 5, dtype=np.int32)
    dem = np.zeros((SIZE, SIZE), dtype=np.float32)
    kwargs = dict(
        veg=veg,
        dem=dem,
        realizations=realizations,
        do_spotting=False,
        seed=seed,
        out_of_bounds_mode="ignore",
    )
    if core == "rust":
        pytest.importorskip("propagator_rust")
        from propagator.rust_core import Propagator as RustPropagator

        simulator = RustPropagator(
            fuels_dict=FUEL_SYSTEM_LEGACY_DICT, **kwargs
        )
    else:
        simulator = Propagator(**kwargs)
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=np.full((SIZE, SIZE), 10.0, dtype=np.float32),
            wind_dir=np.full((SIZE, SIZE), 45.0, dtype=np.float32),
            wind_speed=np.full((SIZE, SIZE), 15.0, dtype=np.float32),
            ignitions=[(SIZE // 2, SIZE // 2)],
        )
    )
    return simulator


def burnt(simulator) -> np.ndarray:
    return simulator.get_output().fire_probability > 0


@pytest.mark.parametrize("core", CORES)
def test_certain_success_on_the_burnt_cells_stops_the_fire(core):
    sim = make_simulator(core)
    sim.step(seconds=1800)
    before = sim.get_output().fire_probability.copy()
    # only the burnt cells: the pending events into their unburnt
    # neighbours are cancelled because they come *from* masked cells
    dropped = sim.extinguish(before > 0)
    assert dropped > 0
    assert sim.next_time() is None
    sim.step(seconds=3600)
    np.testing.assert_array_equal(sim.get_output().fire_probability, before)


@pytest.mark.parametrize("core", CORES)
def test_zero_probability_changes_nothing(core):
    a = make_simulator(core)
    a.step(seconds=1800)
    assert a.extinguish(np.zeros((SIZE, SIZE), dtype=np.float32)) == 0
    a.step(seconds=3600)
    reference = a.get_output().fire_probability.copy()

    b = make_simulator(core)
    b.step(seconds=1800)
    b.step(seconds=3600)
    np.testing.assert_array_equal(b.get_output().fire_probability, reference)


@pytest.mark.parametrize("core", CORES)
def test_partial_success_is_stochastic_and_reproducible(core):
    def run(p: float):
        sim = make_simulator(core)
        sim.step(seconds=1800)
        mask = np.full((SIZE, SIZE), p, dtype=np.float32)
        dropped = sim.extinguish(mask)
        next_time = sim.next_time()
        sim.step(seconds=3600)
        return dropped, next_time, sim.get_output().fire_probability.copy()

    all_dropped, all_next, _ = run(1.0)
    half_dropped, half_next, half_fire = run(0.5)
    assert all_next is None
    # one draw per cell and realization: part of the front survives
    assert 0 < half_dropped < all_dropped
    assert half_next is not None
    # same seed, same outcome
    _, _, again = run(0.5)
    np.testing.assert_array_equal(again, half_fire)


@pytest.mark.parametrize("core", CORES)
def test_pending_ignitions_on_cleared_cells_are_dropped(core):
    sim = make_simulator(core)
    cell = (5, 5)
    sim.set_boundary_conditions(BoundaryConditions(time=600, ignitions=[cell]))
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[cell] = True
    sim.extinguish(mask)
    sim.step(seconds=1200)
    assert not burnt(sim)[cell]


def test_numba_rejects_a_grid_of_the_wrong_shape():
    sim = make_simulator("numba")
    with pytest.raises(ValueError):
        sim.extinguish(np.ones((SIZE, SIZE + 1), dtype=np.float32))


def test_numba_checkpoint_keeps_event_sources(tmp_path):
    sim = make_simulator("numba")
    sim.step(seconds=1800)
    mask = burnt(sim)
    path = tmp_path / "cp.npz"
    sim.checkpoint().save(path)

    from propagator.core.checkpoint import PropagatorCheckpoint

    restored = make_simulator("numba")
    restored.restore(PropagatorCheckpoint.load(path))
    restored.extinguish(mask)
    # sources survived the round trip: spread out of the burnt cells is
    # cancelled, so nothing is left
    assert restored.next_time() is None


def test_numba_checkpoint_without_sources_still_loads():
    # version-2 checkpoints have no front_src: every event counts as
    # external, so only spread *into* masked cells is cancelled
    sim = make_simulator("numba")
    sim.step(seconds=1800)
    mask = burnt(sim)
    old = dataclasses.replace(sim.checkpoint(), front_src=None, version=2)

    restored = make_simulator("numba")
    restored.restore(old)
    restored.extinguish(mask)
    assert restored.next_time() is not None
