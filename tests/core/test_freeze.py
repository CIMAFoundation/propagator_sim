from __future__ import annotations

import numpy as np
import pytest

from propagator.core import (  # type: ignore
    BoundaryConditions,
    Propagator,
    PropagatorCheckpoint,
)
from propagator.core.numba import TILE_SIZE  # type: ignore


def make_propagator(
    tmp_path, size: int = 128, seed: int | None = None, freeze: bool = True
) -> Propagator:
    propagator = Propagator(
        veg=np.full((size, size), 5, dtype=np.int32),
        dem=np.zeros((size, size), dtype=np.float32),
        realizations=4,
        do_spotting=False,
        seed=seed,
        freeze_dir=tmp_path / "store" if freeze else None,
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
    return propagator


def outputs_snapshot(propagator: Propagator):
    out = propagator.get_output()
    return (
        out.fire_probability.copy(),
        out.min_arrival_time.copy(),
        out.ros_max.copy(),
        out.fli_max.copy(),
        propagator.get_fire().copy(),
        propagator.get_arrival_time().copy(),
        propagator.get_ros().copy(),
        propagator.get_fireline_int().copy(),
    )


def run_until_frozen(propagator: Propagator, max_hours: int = 48) -> int:
    """Advance until at least one tile freezes (interior burned out)."""
    for _ in range(max_hours):
        propagator.step(seconds=3600)
        frozen = propagator.freeze_inactive_tiles()
        if frozen:
            return frozen
    return 0


def test_freeze_preserves_outputs_and_releases_slots(tmp_path):
    propagator = make_propagator(tmp_path)
    for _ in range(24):
        propagator.step(seconds=3600)

    before = outputs_snapshot(propagator)
    live_before = int(propagator._tile_counts.sum())

    frozen = propagator.freeze_inactive_tiles()
    assert frozen > 0
    assert int(propagator._tile_counts.sum()) == live_before - frozen
    assert len(propagator._tile_store) == frozen

    after = outputs_snapshot(propagator)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)


def test_freeze_does_not_change_seeded_evolution(tmp_path):
    # thread RNGs are process-global: run each simulation to completion
    # with a reseed right before it, instead of interleaving
    frozen_run = make_propagator(tmp_path)
    frozen_run.reseed(11)
    for _ in range(12):
        frozen_run.step(seconds=3600)
        frozen_run.freeze_inactive_tiles()

    plain_run = make_propagator(tmp_path, freeze=False)
    plain_run.reseed(11)
    for _ in range(12):
        plain_run.step(seconds=3600)

    assert len(frozen_run._tile_store) > 0
    np.testing.assert_array_equal(frozen_run.get_fire(), plain_run.get_fire())
    np.testing.assert_array_equal(
        frozen_run.get_arrival_time(), plain_run.get_arrival_time()
    )


def test_ignition_into_frozen_tile_thaws_it(tmp_path):
    propagator = make_propagator(tmp_path)
    frozen = run_until_frozen(propagator)
    assert frozen > 0

    # find a frozen tile and re-ignite a cell inside it
    key = next(iter(propagator._tile_store.keys()))
    realization = key[0]
    tile_row, tile_col = propagator._tile_local_pos(key)
    row = (tile_row << 5) + TILE_SIZE // 2
    col = (tile_col << 5) + TILE_SIZE // 2

    count_before = len(propagator._tile_store)
    slot = propagator._state_tile_slot(realization, row, col)
    assert slot[0] >= 0
    assert len(propagator._tile_store) == count_before - 1


def test_incremental_checkpoint_references_frozen_tiles(tmp_path):
    propagator = make_propagator(tmp_path)
    frozen = run_until_frozen(propagator)
    assert frozen > 0

    reference = outputs_snapshot(propagator)
    frozen_before = len(propagator._tile_store)
    live_before = int(propagator._tile_counts.sum())

    checkpoint = propagator.checkpoint()
    # incremental: nothing is thawed, nothing is copied into RAM
    assert len(propagator._tile_store) == frozen_before
    assert int(propagator._tile_counts.sum()) == live_before
    assert len(checkpoint.frozen_index) == frozen_before

    # keep simulating, including thaw/refreeze churn, then roll back
    propagator.step(seconds=6 * 3600)
    propagator.freeze_inactive_tiles()

    propagator.restore(checkpoint)
    assert len(propagator._tile_store) == frozen_before
    for left, right in zip(reference, outputs_snapshot(propagator)):
        np.testing.assert_array_equal(left, right)

    # the checkpoint survives repeated restores
    propagator.step(seconds=6 * 3600)
    propagator.restore(checkpoint)
    for left, right in zip(reference, outputs_snapshot(propagator)):
        np.testing.assert_array_equal(left, right)


def test_checkpoint_with_frozen_tiles_save_load(tmp_path):
    propagator = make_propagator(tmp_path)
    frozen = run_until_frozen(propagator)
    assert frozen > 0
    reference = outputs_snapshot(propagator)

    checkpoint = propagator.checkpoint()
    path = tmp_path / "state.npz"
    checkpoint.save(path)
    assert (tmp_path / "state.tiles").exists()

    loaded = PropagatorCheckpoint.load(path)
    assert len(loaded.frozen_index) == frozen

    # resume WITH a store: records are imported, stay frozen
    with_store = Propagator.from_checkpoint(
        loaded, freeze_dir=tmp_path / "store2"
    )
    assert len(with_store._tile_store) == frozen
    for left, right in zip(reference, outputs_snapshot(with_store)):
        np.testing.assert_array_equal(left, right)

    # resume WITHOUT a store: records are materialized into the pools
    without_store = Propagator.from_checkpoint(loaded)
    assert without_store._tile_store is None
    for left, right in zip(reference, outputs_snapshot(without_store)):
        np.testing.assert_array_equal(left, right)


def test_expand_keeps_frozen_tiles_valid(tmp_path):
    size = 128
    propagator = make_propagator(tmp_path, size=size)
    frozen = run_until_frozen(propagator)
    assert frozen > 0
    before_fire = propagator.get_fire().copy()

    margin = TILE_SIZE
    new_size = size + 2 * margin
    propagator.expand(
        veg=np.full((new_size, new_size), 5, dtype=np.int32),
        dem=np.zeros((new_size, new_size), dtype=np.float32),
        origin=(-margin, -margin),
    )

    overlap = slice(margin, margin + size)
    np.testing.assert_array_equal(
        propagator.get_fire()[:, overlap, overlap], before_fire
    )


def test_freeze_requires_freeze_dir(tmp_path):
    propagator = make_propagator(tmp_path, freeze=False)
    with pytest.raises(RuntimeError, match="freeze_dir"):
        propagator.freeze_inactive_tiles()
