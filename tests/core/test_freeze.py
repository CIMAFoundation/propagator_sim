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


def test_freeze_preserves_mean_outputs(tmp_path):
    propagator = make_propagator(tmp_path)
    for _ in range(24):
        propagator.step(seconds=3600)

    before = propagator.get_output()
    assert propagator.freeze_inactive_tiles() > 0
    after = propagator.get_output()

    # float sums are accumulated in a different order through the cache:
    # equal up to floating-point associativity
    for name in (
        "mean_arrival_time",
        "ros_mean",
        "fli_mean",
        "flame_height_mean",
        "flame_height_max",
    ):
        np.testing.assert_allclose(
            getattr(after, name),
            getattr(before, name),
            rtol=1e-9,
            equal_nan=True,
        )


def test_fold_uses_cache_not_disk(tmp_path):
    propagator = make_propagator(tmp_path)
    frozen = run_until_frozen(propagator)
    assert frozen > 0

    store = propagator._tile_store
    reads = {"count": 0}
    original_read_bytes = store.read_bytes

    def counting_read_bytes(offset):
        reads["count"] += 1
        return original_read_bytes(offset)

    store.read_bytes = counting_read_bytes  # type: ignore[method-assign]

    # blocks were built incrementally at freeze time: no disk reads
    propagator.get_output()
    propagator.get_output()
    assert reads["count"] == 0

    # a thaw dirties one position; the next fold re-reads only the
    # remaining records at that position
    key = next(iter(store.keys()))
    tile_row, tile_col = propagator._tile_local_pos(key)
    propagator._thaw_tile(key[0], tile_row, tile_col)
    propagator.get_output()
    assert reads["count"] <= propagator.realizations


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


def _rust_island_run(tmp_path, freeze: bool):
    """Rust core: grass with a non-fuel ring enclosing a fuel island, a
    fire burning around it, then a batch of ignitions inside the island
    (all realizations, plus a second cell in realization 1)."""
    pytest.importorskip("propagator_rust")
    from propagator.core.constants import (  # type: ignore
        FUEL_SYSTEM_LEGACY_DICT,
    )
    from propagator.rust_core import Propagator as RustPropagator

    size, realizations = 128, 6
    veg = np.full((size, size), 5, dtype=np.int32)
    lo, hi = 36, 59
    veg[lo, lo : hi + 1] = 0
    veg[hi, lo : hi + 1] = 0
    veg[lo : hi + 1, lo] = 0
    veg[lo : hi + 1, hi] = 0
    simulator = RustPropagator(
        veg=veg,
        dem=np.zeros((size, size), dtype=np.float32),
        realizations=realizations,
        fuels_dict=FUEL_SYSTEM_LEGACY_DICT,
        do_spotting=False,
        seed=3,
        freeze_dir=tmp_path / "rust_store" if freeze else None,
        out_of_bounds_mode="ignore",
    )
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=5.0,
            wind_dir=0.0,
            wind_speed=10.0,
            ignitions=[(110, 20)],
        )
    )
    ignition_time = 40 * 3600 + 1800
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=ignition_time,
            ignitions=[(48, 48, r) for r in range(realizations)]
            + [(40, 55, 1)],
        )
    )
    frozen = 0
    for hour in range(1, 81):
        simulator.step(seconds=3600)
        if freeze:
            frozen += simulator.freeze_inactive_tiles()
        if hour == 40:
            # the island has not burnt yet
            assert simulator.get_output().fire_probability[48, 48] == 0
    return simulator.get_output(), frozen


def test_rust_ignition_inside_frozen_island_matches_plain_run(tmp_path):
    plain, _ = _rust_island_run(tmp_path, freeze=False)
    frozen, frozen_count = _rust_island_run(tmp_path, freeze=True)
    assert frozen_count > 0
    assert frozen.fire_probability[48, 48] == 1.0
    assert frozen.fire_probability[40, 55] > 0
    for name in (
        "fire_probability",
        "min_arrival_time",
        "mean_arrival_time",
        "ros_max",
        "ros_mean",
        "fli_max",
    ):
        np.testing.assert_array_equal(
            getattr(frozen, name), getattr(plain, name), err_msg=name
        )


def test_rust_freeze_diagnostic_reports_counts():
    pytest.importorskip("propagator_rust")
    from propagator.core.constants import (  # type: ignore
        FUEL_SYSTEM_LEGACY_DICT,
    )
    from propagator.rust_core import Propagator as RustPropagator

    size = 128
    simulator = RustPropagator(
        veg=np.full((size, size), 5, dtype=np.int32),
        dem=np.zeros((size, size), dtype=np.float32),
        realizations=4,
        fuels_dict=FUEL_SYSTEM_LEGACY_DICT,
        do_spotting=True,
        seed=1,
        out_of_bounds_mode="ignore",
    )
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=5.0,
            wind_dir=0.0,
            wind_speed=10.0,
            ignitions=[(64, 64)],
        )
    )
    simulator.step(seconds=4 * 3600)
    counts = simulator.freeze_diagnostic()
    assert set(counts) == {
        "allocated",
        "strict_freezable",
        "reachability_freezable",
        "blocked_by_strict",
    }
    assert counts["allocated"] > 0
    assert counts["strict_freezable"] <= counts["reachability_freezable"]
    assert (
        counts["blocked_by_strict"]
        == counts["reachability_freezable"] - counts["strict_freezable"]
    )


def _rust_spotting_run(
    tmp_path, fuzzy: float | None, trim_margin: int | None = None
):
    pytest.importorskip("propagator_rust")
    from propagator.core.constants import (  # type: ignore
        FUEL_SYSTEM_LEGACY_DICT,
    )
    from propagator.rust_core import Propagator as RustPropagator

    size = 128
    simulator = RustPropagator(
        veg=np.full((size, size), 5, dtype=np.int32),
        dem=np.zeros((size, size), dtype=np.float32),
        realizations=4,
        fuels_dict=FUEL_SYSTEM_LEGACY_DICT,
        do_spotting=True,
        seed=12,
        freeze_dir=tmp_path / f"fuzzy-{fuzzy}" if fuzzy is not None else None,
        out_of_bounds_mode="ignore",
        fuzzy_freeze_threshold=fuzzy,
        fuzzy_freeze_margin=0,
    )
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=4.0,
            wind_dir=270.0,
            wind_speed=30.0,
            ignitions=[(64, 20)],
        )
    )
    for _ in range(36):
        simulator.step(seconds=600)
        if trim_margin is not None:
            simulator.trim_edge_cache(trim_margin)
        if fuzzy is not None:
            simulator.freeze_inactive_tiles()
    return simulator


def test_rust_fuzzy_freeze_is_result_neutral(tmp_path):
    plain = _rust_spotting_run(tmp_path, None).get_output()
    fuzzy_sim = _rust_spotting_run(tmp_path, 0.0)
    fuzzy = fuzzy_sim.get_output()
    stats = fuzzy_sim.freeze_stats()
    assert set(stats) == {
        "dead_tiles",
        "cold_tiles",
        "fault_ins",
        "fault_in_bytes",
        "live_tiles",
        "live_tile_bytes",
        "edge_cache_bytes",
        "fold_cache_bytes",
    }
    assert stats["live_tile_bytes"] == stats["live_tiles"] * 13312
    assert stats["fault_ins"] > 0
    assert stats["fault_in_bytes"] > 0
    for name in (
        "fire_probability",
        "spotting_receiving_probability",
        "min_arrival_time",
        "mean_arrival_time",
        "ros_mean",
        "fli_max",
    ):
        np.testing.assert_array_equal(
            getattr(fuzzy, name), getattr(plain, name), err_msg=name
        )


def test_rust_fuzzy_freeze_requires_freeze_dir():
    pytest.importorskip("propagator_rust")
    from propagator.core.constants import (  # type: ignore
        FUEL_SYSTEM_LEGACY_DICT,
    )
    from propagator.rust_core import Propagator as RustPropagator

    with pytest.raises(ValueError, match="freeze_dir"):
        RustPropagator(
            veg=np.full((64, 64), 5, dtype=np.int32),
            dem=np.zeros((64, 64), dtype=np.float32),
            realizations=1,
            fuels_dict=FUEL_SYSTEM_LEGACY_DICT,
            fuzzy_freeze_threshold=0.9,
        )


def test_rust_edge_cache_trim_is_result_neutral(tmp_path):
    plain_sim = _rust_spotting_run(tmp_path, None)
    plain = plain_sim.get_output()
    trimmed_sim = _rust_spotting_run(tmp_path, None, trim_margin=0)
    trimmed = trimmed_sim.get_output()
    assert trimmed_sim.edge_cache_bytes() < plain_sim.edge_cache_bytes()
    for name in ("fire_probability", "min_arrival_time", "ros_max", "fli_max"):
        np.testing.assert_array_equal(
            getattr(plain, name), getattr(trimmed, name), err_msg=name
        )
