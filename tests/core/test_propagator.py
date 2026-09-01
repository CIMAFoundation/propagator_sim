from __future__ import annotations

import numpy as np
import pytest

from propagator.core import BoundaryConditions, Propagator  # type: ignore
from propagator.core.models import PropagatorStats, UpdateBatch  # type: ignore
from propagator.core.scheduler import SchedulerEvent  # type: ignore


def make_propagator(realizations: int = 2) -> Propagator:
    veg = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dem = np.zeros_like(veg, dtype=np.float32)
    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=realizations,
        do_spotting=False,
    )
    base = np.full_like(veg, 0.2, dtype=np.float32)
    propagator.moisture = base.copy()
    propagator.wind_dir = np.zeros_like(base)
    propagator.wind_speed = np.full_like(base, 5.0)
    return propagator


def test_spotting_state_allocated_only_when_enabled():
    base_veg = np.array([[1, 2], [3, 4]], dtype=np.int32)
    base_dem = np.zeros_like(base_veg, dtype=np.float32)

    no_spotting = Propagator(
        veg=base_veg,
        dem=base_dem,
        realizations=2,
        do_spotting=False,
    )
    assert no_spotting.spotting_generation is None
    assert no_spotting.spotting_receiving is None
    np.testing.assert_allclose(
        no_spotting.compute_spotting_generation_probability(),
        np.zeros(base_veg.shape, dtype=np.float32),
    )
    np.testing.assert_allclose(
        no_spotting.compute_spotting_receiving_probability(),
        np.zeros(base_veg.shape, dtype=np.float32),
    )

    with_spotting = Propagator(
        veg=base_veg,
        dem=base_dem,
        realizations=2,
        do_spotting=True,
    )
    assert with_spotting.spotting_generation is not None
    assert with_spotting.spotting_receiving is not None


def test_zero_front_capacity_raises_instead_of_silently_dropping_ignitions():
    """Regression test: `front_capacity=0` used to make every `_front_push`
    (including ignitions) silently no-op, leaving `_front_sizes` at zero
    so `_propagate_until` returned immediately without ever reaching the
    overflow check -- the simulation looked "done" with nothing burned."""
    veg = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dem = np.zeros_like(veg, dtype=np.float32)
    with pytest.raises(ValueError):
        Propagator(veg=veg, dem=dem, realizations=2, front_capacity=0)


def test_negative_front_capacity_factor_raises():
    veg = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dem = np.zeros_like(veg, dtype=np.float32)
    with pytest.raises(ValueError):
        Propagator(
            veg=veg, dem=dem, realizations=2, front_capacity_factor=-1.0
        )


def test_disabling_spotting_does_not_mutate_shared_default_fuel_system():
    """Regression test: `Propagator(do_spotting=False)` used to call
    `self.fuels.disable_spotting()` in place on whatever fuel system it
    was given. Since the default fuel system is `FUEL_SYSTEM_LEGACY`, a
    single shared module-level instance, this permanently disabled
    spotting on it — silently breaking spotting for every subsequently
    created `Propagator(do_spotting=True)` in the same process that also
    relied on the default fuel system."""
    from propagator.core import FUEL_SYSTEM_LEGACY

    base_veg = np.array(
        [[5, 5], [5, 5]], dtype=np.int32
    )  # conifers: spotting-capable
    base_dem = np.zeros_like(base_veg, dtype=np.float32)
    spotting_before = list(FUEL_SYSTEM_LEGACY.spotting)

    Propagator(veg=base_veg, dem=base_dem, realizations=1, do_spotting=False)

    assert list(FUEL_SYSTEM_LEGACY.spotting) == spotting_before

    with_spotting = Propagator(
        veg=base_veg, dem=base_dem, realizations=1, do_spotting=True
    )
    assert any(with_spotting.fuels.spotting)


def test_compute_fire_probability_and_means():
    propagator = make_propagator(realizations=2)

    propagator.fire = np.array(
        [
            [[1, 0], [0, 1]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int8,
    )
    propagator.ros = np.array(
        [
            [[0.8, 0.0], [0.0, 1.2]],
            [[0.4, 0.6], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    propagator.fireline_int = np.array(
        [
            [[10.0, 0.0], [0.0, 20.0]],
            [[5.0, 15.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    propagator.arrival_time = np.array(
        [
            [[10, 0], [0, 20]],
            [[15, 25], [0, 0]],
        ],
        dtype=np.int32,
    )
    propagator.spotting_generation = np.array(
        [
            [[1, 0], [0, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.uint32,
    )
    propagator.spotting_receiving = np.array(
        [
            [[0, 0], [1, 1]],
            [[0, 1], [0, 0]],
        ],
        dtype=np.uint32,
    )

    prob = propagator.compute_fire_probability()
    np.testing.assert_allclose(
        prob,
        np.array(
            [
                [0.5, 0.5],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    spotting_gen_prob = propagator.compute_spotting_generation_probability()
    np.testing.assert_allclose(
        spotting_gen_prob,
        np.array(
            [
                [0.5, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    spotting_recv_prob = propagator.compute_spotting_receiving_probability()
    np.testing.assert_allclose(
        spotting_recv_prob,
        np.array(
            [
                [0.0, 1.0],
                [0.5, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    min_arrival = propagator.compute_arrival_time_min()
    np.testing.assert_allclose(
        min_arrival,
        np.array(
            [
                [10.0, 20.0],
                [15.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    mean_arrival = propagator.compute_arrival_time_mean()
    np.testing.assert_allclose(
        mean_arrival,
        np.array(
            [
                [10.0, 20.0],
                [20.0, np.nan],
            ],
            dtype=np.float32,
        ),
        equal_nan=True,
    )

    ros_max = propagator.compute_ros_max()
    np.testing.assert_allclose(
        ros_max,
        np.array(
            [
                [0.8, 1.2],
                [0.6, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    ros_mean = propagator.compute_ros_mean()
    np.testing.assert_allclose(
        ros_mean,
        np.array(
            [
                [0.8, 1.2],
                [0.5, np.nan],
            ],
            dtype=np.float32,
        ),
        equal_nan=True,
    )

    fli_max = propagator.compute_fireline_int_max()
    np.testing.assert_allclose(
        fli_max,
        np.array(
            [
                [10.0, 20.0],
                [15.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    def byram(intensity):
        return 0.0775 * intensity**0.46 if intensity > 0 else 0.0

    flame_length_max = propagator.compute_flame_length_max()
    np.testing.assert_allclose(
        flame_length_max,
        np.array(
            [
                [byram(10.0), byram(20.0)],
                [byram(15.0), 0.0],
            ],
            dtype=np.float32,
        ),
    )

    flame_length_mean = propagator.compute_flame_length_mean()
    np.testing.assert_allclose(
        flame_length_mean,
        np.array(
            [
                [byram(10.0), byram(20.0)],
                [(byram(5.0) + byram(15.0)) / 2, np.nan],
            ],
            dtype=np.float32,
        ),
        equal_nan=True,
    )

    fli_mean = propagator.compute_fireline_int_mean()
    np.testing.assert_allclose(
        fli_mean,
        np.array(
            [
                [10.0, 20.0],
                [10.0, np.nan],
            ],
            dtype=np.float32,
        ),
        equal_nan=True,
    )


def test_get_output_includes_spotting_probabilities():
    propagator = make_propagator(realizations=2)
    propagator.time = 60
    propagator.fire = np.array(
        [[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=np.int8
    )
    propagator.spotting_generation = np.array(
        [[[1, 0], [0, 1]], [[0, 0], [0, 0]]], dtype=np.uint32
    )
    propagator.spotting_receiving = np.array(
        [[[0, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=np.uint32
    )
    propagator.arrival_time = np.array(
        [[[5, 0], [0, 0]], [[0, 9], [0, 0]]], dtype=np.int32
    )

    output = propagator.get_output()

    np.testing.assert_allclose(
        output.spotting_generation_probability,
        np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        output.spotting_receiving_probability,
        np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        output.min_arrival_time,
        np.array([[5.0, 0.0], [9.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        output.mean_arrival_time,
        np.array([[5.0, np.nan], [9.0, np.nan]], dtype=np.float32),
        equal_nan=True,
    )


def test_sample_cells_reports_reached_and_unreached_and_out_of_bounds():
    propagator = make_propagator(realizations=2)
    propagator.time = 60
    propagator.fire = np.array(
        [[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=np.int8
    )
    propagator.arrival_time = np.array(
        [[[5, 0], [0, 0]], [[0, 9], [0, 0]]], dtype=np.int32
    )

    samples = propagator.sample_cells(
        [
            ("burned", 0, 0),
            ("unburned", 1, 1),
            ("outside", 99, 99),
        ]
    )
    by_key = {s.key: s for s in samples}

    assert by_key["burned"].reached is True
    assert by_key["burned"].row == 0
    assert by_key["burned"].col == 0
    np.testing.assert_allclose(by_key["burned"].min_arrival_time, 5.0)
    np.testing.assert_allclose(by_key["burned"].mean_arrival_time, 5.0)

    assert by_key["unburned"].reached is False
    assert np.isnan(by_key["unburned"].min_arrival_time)
    assert np.isnan(by_key["unburned"].mean_arrival_time)

    assert by_key["outside"].reached is False
    assert np.isnan(by_key["outside"].min_arrival_time)


def test_passing_precomputed_grids_does_not_change_results():
    """`get_output` hands its already-reduced grids to `sample_cells` and
    `compute_flame_length_max` instead of letting them redo the full
    (rows, cols, realizations) reductions. That is purely an
    optimization: results must be identical either way."""
    propagator = make_propagator(realizations=2)
    propagator.fire = np.array(
        [[[1, 0], [0, 1]], [[1, 1], [0, 0]]], dtype=np.int8
    )
    propagator.arrival_time = np.array(
        [[[10, 0], [0, 20]], [[15, 25], [0, 0]]], dtype=np.int32
    )
    propagator.fireline_int = np.array(
        [[[10.0, 0.0], [0.0, 20.0]], [[5.0, 15.0], [0.0, 0.0]]],
        dtype=np.float32,
    )
    cells = [("a", 0, 0), ("b", 1, 1), ("outside", 99, 99)]

    np.testing.assert_allclose(
        propagator.compute_flame_length_max(
            fli_max=propagator.compute_fireline_int_max()
        ),
        propagator.compute_flame_length_max(),
    )
    with_precomputed = propagator.sample_cells(
        cells,
        fire_probability=propagator.compute_fire_probability(),
        min_arrival=propagator.compute_arrival_time_min(),
        mean_arrival=propagator.compute_arrival_time_mean(),
    )
    without = propagator.sample_cells(cells)

    # compared field by field: an unreached cell reports NaN arrival
    # times, and NaN != NaN would make plain dataclass equality fail even
    # for identical results
    assert len(with_precomputed) == len(without)
    for a, b in zip(with_precomputed, without):
        assert (a.key, a.row, a.col, a.reached) == (
            b.key,
            b.row,
            b.col,
            b.reached,
        )
        np.testing.assert_allclose(
            [a.min_arrival_time, a.mean_arrival_time],
            [b.min_arrival_time, b.mean_arrival_time],
            equal_nan=True,
        )


def test_flame_length_is_derived_lazily_and_cached():
    """`get_output` no longer computes the flame-length products eagerly:
    the mean needs a full (rows, cols, realizations) transient that
    consumers which never read it (the web UI) shouldn't pay for on every
    frame. The values must still match the direct computation, and be
    computed only once per snapshot."""
    propagator = make_propagator(realizations=2)
    propagator.fire = np.array(
        [[[1, 0], [0, 1]], [[1, 1], [0, 0]]], dtype=np.int8
    )
    propagator.fireline_int = np.array(
        [[[10.0, 0.0], [0.0, 20.0]], [[5.0, 15.0], [0.0, 0.0]]],
        dtype=np.float32,
    )

    calls = {"n": 0}
    real_mean = propagator.compute_flame_length_mean

    def counting_mean():
        calls["n"] += 1
        return real_mean()

    propagator.compute_flame_length_mean = counting_mean
    output = propagator.get_output()
    assert calls["n"] == 0, "must not be computed until read"

    np.testing.assert_allclose(
        output.flame_length_mean, real_mean(), equal_nan=True
    )
    assert calls["n"] == 1
    # second read is served from the cache, not recomputed
    output.flame_length_mean
    assert calls["n"] == 1

    np.testing.assert_allclose(
        output.flame_length_max,
        propagator.compute_flame_length_max(),
        equal_nan=True,
    )


def test_flame_length_raises_clearly_on_a_hand_built_output():
    from propagator.core.models import PropagatorOutput, PropagatorStats

    grid = np.zeros((2, 2), dtype=np.float32)
    output = PropagatorOutput(
        time=0,
        fire_probability=grid,
        spotting_generation_probability=grid,
        spotting_receiving_probability=grid,
        mean_arrival_time=grid,
        min_arrival_time=grid,
        ros_mean=grid,
        ros_max=grid,
        fli_mean=grid,
        fli_max=grid,
        stats=PropagatorStats(0, 0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(AttributeError):
        output.flame_length_mean


def test_byram_flame_length_does_not_mutate_its_input():
    """It computes in place inside one transient buffer to avoid a second
    full copy of the 3D intensity array; that buffer must be its own, not
    the caller's array."""
    from propagator.core.propagator import _byram_flame_length

    fireline_int = np.array([-5.0, 0.0, 10.0, 20.0], dtype=np.float32)
    original = fireline_int.copy()

    result = _byram_flame_length(fireline_int)

    np.testing.assert_array_equal(fireline_int, original)
    assert result is not fireline_int
    np.testing.assert_allclose(
        result,
        np.array([0.0, 0.0, 0.0775 * 10.0**0.46, 0.0775 * 20.0**0.46]),
        rtol=1e-6,
    )


def test_get_output_sample_cells_defaults_to_empty():
    propagator = make_propagator(realizations=1)
    propagator.fire = np.zeros((2, 2, 1), dtype=np.int8)
    propagator.arrival_time = np.zeros((2, 2, 1), dtype=np.int32)

    output = propagator.get_output()
    assert output.poi_arrival == ()

    output_with_samples = propagator.get_output(sample_cells=[("p", 0, 0)])
    assert len(output_with_samples.poi_arrival) == 1
    assert output_with_samples.poi_arrival[0].key == "p"


def test_compute_stats_counts_active_and_thresholds():
    propagator = make_propagator(realizations=2)

    updates = UpdateBatch(
        rows=np.array([0, 1], dtype=np.int32),
        cols=np.array([0, 1], dtype=np.int32),
        realizations=np.array([0, 1], dtype=np.int32),
        rates_of_spread=np.array([0.3, 0.4], dtype=np.float32),
        fireline_intensities=np.array([1.0, 2.0], dtype=np.float32),
    )
    propagator._schedule_ignitions(1, updates)

    values = np.array(
        [
            [0.2, 0.75],
            [0.51, 0.9],
        ],
        dtype=np.float32,
    )

    stats = propagator.compute_stats(values)

    assert isinstance(stats, PropagatorStats)
    assert stats.n_active == 2
    cell_area = propagator.cellsize**2
    assert stats.area_mean == pytest.approx(2.36 * cell_area)
    assert stats.area_50 == 3 * cell_area
    assert stats.area_75 == 2 * cell_area
    assert stats.area_90 == 1 * cell_area


def test_set_boundary_conditions_enqueue_event():
    propagator = make_propagator(realizations=1)

    boundary = BoundaryConditions(
        time=3,
        moisture=np.full((2, 2), 30.0, dtype=np.float32),
        wind_dir=np.array(
            [
                [0.0, 90.0],
                [180.0, 270.0],
            ],
            dtype=np.float32,
        ),
        wind_speed=np.full((2, 2), 12.0, dtype=np.float32),
        ignitions=np.array(
            [
                [True, False],
                [False, False],
            ],
            dtype=bool,
        ),
        additional_moisture=np.full((2, 2), 5.0, dtype=np.float32),
        vegetation_changes=np.array(
            [
                [np.nan, 2.0],
                [3.0, np.nan],
            ],
            dtype=np.float32,
        ),
    )

    propagator.set_boundary_conditions(boundary)
    time, event = propagator.scheduler.pop()

    assert time == 3
    np.testing.assert_allclose(
        event.moisture, np.full((2, 2), 0.3, dtype=np.float32)
    )
    expected_wind_dir = np.radians(
        [
            [0.0, 90.0],
            [180.0, 270.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(event.wind_dir, expected_wind_dir)
    np.testing.assert_allclose(
        event.wind_speed, np.full((2, 2), 12.0, dtype=np.float32)
    )
    np.testing.assert_allclose(
        event.additional_moisture, np.full((2, 2), 0.05, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        event.vegetation_changes,
        np.array(
            [
                [np.nan, 2.0],
                [3.0, np.nan],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        event.updates.rows, np.array([0], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        event.updates.cols, np.array([0], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        event.updates.realizations, np.array([0], dtype=np.int32)
    )


def test_decay_actions_moisture_exponential():
    propagator = make_propagator(realizations=1)
    propagator.actions_moisture = np.full((2, 2), 0.5, dtype=np.float32)

    propagator._decay_actions_moisture(time_delta=5 * 60, decay_factor=0.1)

    expected_value = 0.5 * (1 - 0.1) ** 5
    np.testing.assert_allclose(
        propagator.actions_moisture,
        np.full((2, 2), expected_value, dtype=np.float32),
    )

    propagator.actions_moisture = None
    propagator._decay_actions_moisture(time_delta=5, decay_factor=0.1)
    assert propagator.actions_moisture is None


def test_front_capacity_defaults_to_twice_grid_cells():
    veg = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dem = np.zeros_like(veg, dtype=np.float32)

    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=2,
        do_spotting=False,
    )

    assert propagator._front_capacity == 2 * veg.size


def test_front_capacity_explicit_override():
    veg = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dem = np.zeros_like(veg, dtype=np.float32)

    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=2,
        do_spotting=False,
        front_capacity_factor=5.0,
        front_capacity=3,
    )

    assert propagator._front_capacity == 3


def test_front_push_sets_overflow_beyond_capacity():
    veg = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dem = np.zeros_like(veg, dtype=np.float32)

    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=1,
        do_spotting=False,
        front_capacity=1,
    )

    propagator._front_push(
        realization=0, time=0, row=0, col=0, ros=0.0, fli=0.0
    )
    assert propagator._front_overflow[0] == 0

    propagator._front_push(
        realization=0, time=1, row=1, col=1, ros=0.0, fli=0.0
    )
    assert propagator._front_overflow[0] == 1


def test_front_heap_overflow_raises_runtime_error():
    """A front queue that overflows capacity must fail loudly instead of
    silently dropping pending spread updates."""
    from propagator.core.numba.models import FuelSystem

    veg = np.array([[5, 5], [5, 5]], dtype=np.int32)
    dem = np.zeros_like(veg, dtype=np.float32)

    # single-fuel system with certain (p=1) transition: every neighbour
    # of a burning cell is scheduled, deterministically
    fuels = FuelSystem(1)
    fuels.add_fuel(5, "conifers", 1.0, 1.0, 20000.0, 0.0, -9999.0)
    fuels.add_transition_probability(5, 5, 1.0)

    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=1,
        do_spotting=False,
        fuels=fuels,
        front_capacity=2,
    )
    propagator.moisture = np.zeros((2, 2), dtype=np.float32)
    propagator.wind_dir = np.zeros((2, 2), dtype=np.float32)
    propagator.wind_speed = np.zeros((2, 2), dtype=np.float32)

    propagator.set_boundary_conditions(
        BoundaryConditions(time=0, ignitions=[(0, 0)])
    )

    with pytest.raises(RuntimeError, match="front queue overflowed"):
        propagator.step()


def _make_two_fuel_propagator(non_vegetated_at, ignite_at, realizations=5):
    """A 9x9, all-fuel-5 (conifers) grid except for one cell set to fuel 3
    (non-vegetated, burn=False), placed away from the grid edge. Only the
    5<->3 transitions are given a certain (p=1) probability (every other
    transition, including 5->5, stays at the default 0), so any spread
    touching the fuel-3 cell that isn't blocked by `Fuel.burn` would show
    up deterministically rather than only probabilistically."""
    from propagator.core.numba.models import FuelSystem

    veg = np.full((9, 9), 5, dtype=np.int32)
    veg[non_vegetated_at] = 3
    dem = np.zeros_like(veg, dtype=np.float32)
    fuels = FuelSystem(2)
    fuels.add_fuel(5, "conifers", 1.0, 1.0, 20000.0, 0.0, -9999.0)
    fuels.add_fuel(
        3, "non-vegetated", 1.0, 1.0, 20000.0, 0.0, -9999.0, False, 0.0, False
    )
    fuels.add_transition_probability(5, 3, 1.0)
    fuels.add_transition_probability(3, 5, 1.0)

    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=realizations,
        do_spotting=False,
        fuels=fuels,
    )
    propagator.moisture = np.zeros(veg.shape, dtype=np.float32)
    propagator.wind_dir = np.zeros(veg.shape, dtype=np.float32)
    propagator.wind_speed = np.zeros(veg.shape, dtype=np.float32)
    propagator.set_boundary_conditions(
        BoundaryConditions(time=0, ignitions=[ignite_at])
    )
    return propagator


def test_unburnable_fuel_does_not_ignite_from_a_burning_neighbour():
    """Regression test: the propagation kernel used to only check
    `veg == NO_FUEL` to decide whether a destination cell could catch
    fire, ignoring `Fuel.burn` entirely -- so a fuel explicitly marked
    non-combustible (e.g. the fuel a firefighting action neutralizes a
    cell with) could still ignite via the fuel table's own residual
    transition probability."""
    propagator = _make_two_fuel_propagator(
        non_vegetated_at=(4, 5), ignite_at=(4, 4)
    )

    steps = 0
    while propagator.next_time() is not None and steps < 20:
        propagator.step()
        steps += 1

    assert np.all(propagator.fire[4, 4, :] == 1)
    assert np.all(propagator.fire[4, 5, :] == 0)


def test_unburnable_fuel_does_not_propagate_outward():
    """Regression test: a forced ignition on a non-burnable cell (as
    happens when a firefighting action neutralizes the ignition point)
    must never spread to a burnable neighbour, even though the source
    cell itself is marked as burning."""
    propagator = _make_two_fuel_propagator(
        non_vegetated_at=(4, 4), ignite_at=(4, 4)
    )

    steps = 0
    while propagator.next_time() is not None and steps < 20:
        propagator.step()
        steps += 1

    assert np.all(propagator.fire[4, 4, :] == 1)
    assert np.all(propagator.fire[4, 5, :] == 0)


def test_step_applies_event():
    propagator = make_propagator(realizations=1)
    propagator.actions_moisture = np.full((2, 2), 0.5, dtype=np.float32)

    event = SchedulerEvent(
        moisture=np.full((2, 2), 0.2, dtype=np.float32),
        additional_moisture=np.full((2, 2), 0.05, dtype=np.float32),
        wind_dir=np.full((2, 2), 1.1, dtype=np.float32),
        wind_speed=np.full((2, 2), 8.0, dtype=np.float32),
        vegetation_changes=np.array(
            [
                [np.nan, 6.0],
                [5.0, np.nan],
            ],
            dtype=np.float32,
        ),
    )
    propagator.scheduler.add_event(180, event)

    propagator.step()

    assert propagator.time == 180
    np.testing.assert_allclose(
        propagator.moisture, np.full((2, 2), 0.2, dtype=np.float32)
    )
    expected_actions = 0.5 * (1 - 0.01) ** 3 + 0.05
    np.testing.assert_allclose(
        propagator.actions_moisture,
        np.full((2, 2), expected_actions, dtype=np.float32),
    )
    np.testing.assert_allclose(
        propagator.wind_dir, np.full((2, 2), 1.1, dtype=np.float32)
    )
    np.testing.assert_allclose(
        propagator.wind_speed, np.full((2, 2), 8.0, dtype=np.float32)
    )
    assert propagator.veg[0, 1] == 6.0
    assert propagator.veg[1, 0] == 5.0
    assert propagator.next_time() is None
