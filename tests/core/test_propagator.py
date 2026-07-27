from __future__ import annotations

import numpy as np
import pytest

from propagator.core import BoundaryConditions, Propagator  # type: ignore
from propagator.core.models import PropagatorStats, UpdateBatch  # type: ignore
from propagator.core.numba import FLAG_SPOT_GEN, FLAG_SPOT_RECV  # type: ignore
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


def test_spotting_probabilities_default_to_zero():
    base_veg = np.array([[1, 2], [3, 4]], dtype=np.int32)
    base_dem = np.zeros_like(base_veg, dtype=np.float32)

    for do_spotting in (False, True):
        propagator = Propagator(
            veg=base_veg,
            dem=base_dem,
            realizations=2,
            do_spotting=do_spotting,
        )
        np.testing.assert_allclose(
            propagator.compute_spotting_generation_probability(),
            np.zeros(base_veg.shape, dtype=np.float32),
        )
        np.testing.assert_allclose(
            propagator.compute_spotting_receiving_probability(),
            np.zeros(base_veg.shape, dtype=np.float32),
        )
        assert (
            propagator.get_spotting_generation().shape == (2,) + base_veg.shape
        )
        assert not propagator.get_spotting_receiving().any()


def ignite_cell(
    propagator: Propagator,
    realization: int,
    row: int,
    col: int,
    time: int,
    ros: float,
    fli: float,
) -> None:
    """Burn a single cell at a given time through the update machinery."""
    updates = UpdateBatch(
        rows=np.array([row], dtype=np.int32),
        cols=np.array([col], dtype=np.int32),
        realizations=np.array([realization], dtype=np.int32),
        rates_of_spread=np.array([ros], dtype=np.float32),
        fireline_intensities=np.array([fli], dtype=np.float32),
    )
    propagator._apply_updates(updates, new_time=time)


def mark_spotting(
    propagator: Propagator,
    realization: int,
    row: int,
    col: int,
    *,
    gen: bool = False,
    recv: bool = False,
) -> None:
    """Set spotting flags on a cell's tile slot (test seam)."""
    tile, local_row, local_col = propagator._state_tile_slot(
        realization, row, col
    )
    if gen:
        propagator._tile_flags[realization, tile, local_row, local_col] |= (
            FLAG_SPOT_GEN
        )
    if recv:
        propagator._tile_flags[realization, tile, local_row, local_col] |= (
            FLAG_SPOT_RECV
        )


def test_compute_fire_probability_and_means():
    propagator = make_propagator(realizations=2)

    # burn state per realization:
    # r0: (0,0) t=10 ros=0.8 fli=10, (1,0) t=15 ros=0.4 fli=5
    # r1: (0,1) t=20 ros=1.2 fli=20, (1,0) t=25 ros=0.6 fli=15
    ignite_cell(propagator, 0, 0, 0, time=10, ros=0.8, fli=10.0)
    ignite_cell(propagator, 0, 1, 0, time=15, ros=0.4, fli=5.0)
    ignite_cell(propagator, 1, 0, 1, time=20, ros=1.2, fli=20.0)
    ignite_cell(propagator, 1, 1, 0, time=25, ros=0.6, fli=15.0)

    # spotting flags:
    # gen: r0 at (0,0) and (1,0); r1 at (1,0)
    # recv: r0 and r1 at (0,1); r1 at (1,0)
    mark_spotting(propagator, 0, 0, 0, gen=True)
    mark_spotting(propagator, 0, 1, 0, gen=True)
    mark_spotting(propagator, 1, 1, 0, gen=True)
    mark_spotting(propagator, 0, 0, 1, recv=True)
    mark_spotting(propagator, 1, 0, 1, recv=True)
    mark_spotting(propagator, 1, 1, 0, recv=True)

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
    # burn (0,0) in r0 at t=5 and (1,0) in r1 at t=9
    ignite_cell(propagator, 0, 0, 0, time=5, ros=0.0, fli=0.0)
    ignite_cell(propagator, 1, 1, 0, time=9, ros=0.0, fli=0.0)
    propagator.time = 60
    # gen: r0 at (0,0), r1 at (0,1); recv: r1 at (0,0), r0 at (1,0)
    mark_spotting(propagator, 0, 0, 0, gen=True)
    mark_spotting(propagator, 1, 0, 1, gen=True)
    mark_spotting(propagator, 1, 0, 0, recv=True)
    mark_spotting(propagator, 0, 1, 0, recv=True)

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


def make_wide_propagator(realizations: int = 2) -> Propagator:
    veg = np.full((128, 128), 4, dtype=np.int32)
    dem = np.zeros_like(veg, dtype=np.float32)
    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=realizations,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )
    base = np.full_like(veg, 0.1, dtype=np.float32)
    propagator.moisture = base.copy()
    propagator.wind_dir = np.zeros_like(base)
    propagator.wind_speed = np.full_like(base, 10.0)
    return propagator


def test_boundary_proximity_reports_only_nearby_edges():
    propagator = make_wide_propagator(realizations=2)

    # a pending event well inside the domain: no edge is close
    propagator._front_push(0, 10, 64, 64, 0.0, 0.0)
    assert propagator.boundary_proximity(4) == (False, False, False, False)

    # ... one 3 cells from the west edge: only west is reported
    propagator._front_push(0, 20, 64, 3, 0.0, 0.0)
    assert propagator.boundary_proximity(4) == (False, False, True, False)

    # every realization's heap is scanned, and margins accumulate per edge
    propagator._front_push(1, 20, 125, 64, 0.0, 0.0)
    assert propagator.boundary_proximity(4) == (False, True, True, False)

    # a margin below 1 is treated as 1: a cell in the boundary ring counts
    propagator._front_push(1, 30, 0, 64, 0.0, 0.0)
    assert propagator.boundary_proximity(0) == (True, False, False, False)


def test_boundary_proximity_predicts_before_the_front_halts():
    propagator = make_wide_propagator(realizations=1)
    # ignite 3 cells from the west edge: the front is close to it from the
    # start, before propagation ever halts on the boundary
    propagator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=np.full((128, 128), 10.0, dtype=np.float32),
            wind_dir=np.zeros((128, 128), dtype=np.float32),
            wind_speed=np.full((128, 128), 10.0, dtype=np.float32),
            ignitions=[(64, 3)],
        )
    )
    propagator.step(seconds=1)

    assert propagator.boundary_proximity(4) == (False, False, True, False)
    # nothing halted, so boundary_pressure stays silent
    assert propagator.boundary_pressure() == (False, False, False, False)


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


def test_apply_updates_updates_state():
    propagator = make_propagator(realizations=1)

    updates = UpdateBatch(
        rows=np.array([0], dtype=np.int32),
        cols=np.array([1], dtype=np.int32),
        realizations=np.array([0], dtype=np.int32),
        rates_of_spread=np.array([2.5], dtype=np.float32),
        fireline_intensities=np.array([7.5], dtype=np.float32),
    )

    future_time = 5
    propagator._apply_updates(updates, new_time=future_time)

    assert propagator.get_fire()[0, 0, 1] == 1
    assert propagator.get_arrival_time()[0, 0, 1] == future_time
    assert propagator.get_ros()[0, 0, 1] == pytest.approx(2.5)
    assert propagator.get_fireline_int()[0, 0, 1] == pytest.approx(7.5)

    time = propagator.time
    assert time == future_time


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
