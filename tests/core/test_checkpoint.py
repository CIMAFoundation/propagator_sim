from __future__ import annotations

import numpy as np
import pytest

from propagator.core import (  # type: ignore
    BoundaryConditions,
    Propagator,
    PropagatorCheckpoint,
    PropagatorOutOfBoundsError,
)
from propagator.core.numba import TILE_SIZE  # type: ignore


def make_running_propagator(
    size: int = 64,
    realizations: int = 4,
    origin: tuple[int, int] = (0, 0),
) -> Propagator:
    veg = np.full((size, size), 5, dtype=np.int32)
    dem = np.zeros((size, size), dtype=np.float32)
    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=realizations,
        do_spotting=False,
        origin=origin,
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


def state_summary(propagator: Propagator):
    fold = propagator._fold_state()
    return (
        propagator.time,
        fold.count.copy(),
        propagator.get_fire().copy(),
        propagator.get_arrival_time().copy(),
        propagator._front_sizes.copy(),
        propagator._tile_counts.copy(),
    )


def assert_same_state(a, b) -> None:
    assert a[0] == b[0]
    for left, right in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(left, right)


def test_checkpoint_restore_rolls_back_state():
    propagator = make_running_propagator()
    propagator.step(seconds=600)

    checkpoint = propagator.checkpoint()
    reference = state_summary(propagator)

    propagator.step(seconds=1800)
    assert propagator.time == 2400
    assert propagator.get_fire().sum() > reference[2].sum()

    propagator.restore(checkpoint)
    assert_same_state(state_summary(propagator), reference)

    # the simulation must be able to continue after a rollback, and the
    # checkpoint must stay valid for repeated restores
    propagator.step(seconds=1800)
    assert propagator.get_fire().sum() > reference[2].sum()
    propagator.restore(checkpoint)
    assert_same_state(state_summary(propagator), reference)


def test_checkpoint_is_isolated_from_live_state():
    propagator = make_running_propagator()
    propagator.step(seconds=600)
    checkpoint = propagator.checkpoint()
    burned_at_checkpoint = int(checkpoint.tile_counts.sum())

    propagator.step(seconds=3600)

    assert int(checkpoint.tile_counts.sum()) == burned_at_checkpoint
    assert checkpoint.time == 600


def test_checkpoint_save_load_roundtrip(tmp_path):
    propagator = make_running_propagator()
    propagator.step(seconds=900)
    # leave a pending boundary condition in the queue
    propagator.set_boundary_conditions(
        BoundaryConditions(
            time=7200,
            wind_speed=np.full((64, 64), 25.0, dtype=np.float32),
        )
    )
    checkpoint = propagator.checkpoint()

    path = tmp_path / "state.npz"
    checkpoint.save(path)
    loaded = PropagatorCheckpoint.load(path)

    assert loaded.time == checkpoint.time
    assert loaded.origin == checkpoint.origin
    assert loaded.realizations == checkpoint.realizations
    np.testing.assert_array_equal(loaded.veg, checkpoint.veg)
    np.testing.assert_array_equal(loaded.front_sizes, checkpoint.front_sizes)
    np.testing.assert_array_equal(loaded.tile_idx, checkpoint.tile_idx)
    np.testing.assert_array_equal(loaded.tile_flags, checkpoint.tile_flags)
    assert len(loaded.scheduler_events) == len(checkpoint.scheduler_events)

    resumed = Propagator.from_checkpoint(loaded)
    assert_same_state(state_summary(resumed), state_summary(propagator))
    assert resumed.scheduler.next_time() == 7200

    resumed.out_of_bounds_mode = "ignore"
    resumed.step(seconds=3600)
    assert resumed.time == 900 + 3600


def test_from_checkpoint_grows_domain():
    size = 64
    propagator = make_running_propagator(size=size, origin=(320, 320))
    propagator.out_of_bounds_mode = "raise"

    # run until the fire hits the boundary; the kernel halts with the
    # boundary event still in the heap
    with pytest.raises(PropagatorOutOfBoundsError):
        for _ in range(500):
            propagator.step(seconds=3600)
    checkpoint = propagator.checkpoint()
    assert any(checkpoint.front_sizes > 0)
    old_summary = state_summary(propagator)

    # grow by one tile row/col on every side
    margin = TILE_SIZE
    new_size = size + 2 * margin
    new_origin = (320 - margin, 320 - margin)
    grown = Propagator.from_checkpoint(
        checkpoint,
        veg=np.full((new_size, new_size), 5, dtype=np.int32),
        dem=np.zeros((new_size, new_size), dtype=np.float32),
        origin=new_origin,
        out_of_bounds_mode="ignore",
    )

    assert grown.time == checkpoint.time
    assert grown.world_bounds() == (
        320 - margin,
        320 - margin,
        320 + size + margin - 1,
        320 + size + margin - 1,
    )

    # state in the overlap is bit-identical, re-anchored by the margin
    overlap = slice(margin, margin + size)
    np.testing.assert_array_equal(
        grown.get_fire()[:, overlap, overlap], old_summary[2]
    )
    np.testing.assert_array_equal(
        grown.get_arrival_time()[:, overlap, overlap], old_summary[3]
    )
    # weather fields were padded to the new grid
    assert grown.moisture.shape == (new_size, new_size)

    # resuming spreads the fire beyond the old grid bounds
    grown.step(seconds=6 * 3600)
    fire = grown.get_fire()
    outside = fire.copy()
    outside[:, overlap, overlap] = 0
    assert outside.sum() > 0


def test_expand_grows_in_place():
    size = 64
    propagator = make_running_propagator(size=size, origin=(320, 320))
    propagator.out_of_bounds_mode = "raise"

    with pytest.raises(PropagatorOutOfBoundsError):
        for _ in range(500):
            propagator.step(seconds=3600)
    before = state_summary(propagator)
    tile_flags_before = propagator._tile_flags
    scheduler_before = propagator.scheduler

    margin = TILE_SIZE
    new_size = size + 2 * margin
    propagator.expand(
        veg=np.full((new_size, new_size), 5, dtype=np.int32),
        dem=np.zeros((new_size, new_size), dtype=np.float32),
        origin=(320 - margin, 320 - margin),
    )

    # in-place: pools and scheduler objects are the same, nothing copied
    assert propagator._tile_flags is tile_flags_before
    assert propagator.scheduler is scheduler_before
    assert propagator.time == before[0]
    assert propagator.veg.shape == (new_size, new_size)
    assert propagator.moisture.shape == (new_size, new_size)

    overlap = slice(margin, margin + size)
    np.testing.assert_array_equal(
        propagator.get_fire()[:, overlap, overlap], before[2]
    )
    np.testing.assert_array_equal(
        propagator.get_arrival_time()[:, overlap, overlap], before[3]
    )

    # the halted boundary events resume and cross the old bounds
    propagator.out_of_bounds_mode = "ignore"
    propagator.step(seconds=6 * 3600)
    fire = propagator.get_fire()
    outside = fire.copy()
    outside[:, overlap, overlap] = 0
    assert outside.sum() > 0

    with pytest.raises(ValueError, match="multiple of"):
        propagator.expand(
            veg=np.full((new_size + 8, new_size + 8), 5, dtype=np.int32),
            dem=np.zeros((new_size + 8, new_size + 8), dtype=np.float32),
            origin=(320 - margin - 8, 320 - margin),
        )


def test_from_checkpoint_validates_alignment_and_coverage():
    propagator = make_running_propagator(size=64, origin=(320, 320))
    propagator.step(seconds=600)
    checkpoint = propagator.checkpoint()

    veg = np.full((128, 128), 5, dtype=np.int32)
    dem = np.zeros((128, 128), dtype=np.float32)

    with pytest.raises(ValueError, match="multiple of"):
        Propagator.from_checkpoint(
            checkpoint, veg=veg, dem=dem, origin=(320 - 7, 320)
        )
    with pytest.raises(ValueError, match="contain"):
        Propagator.from_checkpoint(
            checkpoint, veg=veg, dem=dem, origin=(320 + TILE_SIZE, 320)
        )
    with pytest.raises(ValueError, match="contain"):
        Propagator.from_checkpoint(
            checkpoint,
            veg=np.full((32, 32), 5, dtype=np.int32),
            dem=np.zeros((32, 32), dtype=np.float32),
            origin=(320, 320),
        )


def test_restore_rejects_mismatched_grid():
    propagator = make_running_propagator(size=64)
    propagator.step(seconds=600)
    checkpoint = propagator.checkpoint()

    other = make_running_propagator(size=96)
    with pytest.raises(ValueError, match="grid"):
        other.restore(checkpoint)

    shifted = make_running_propagator(size=64, origin=(TILE_SIZE, 0))
    with pytest.raises(ValueError, match="origin"):
        shifted.restore(checkpoint)
