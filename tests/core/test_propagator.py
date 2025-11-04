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


def test_compute_stats_counts_active_and_thresholds():
    propagator = make_propagator(realizations=2)

    updates = UpdateBatch(
        rows=np.array([0, 1], dtype=np.int32),
        cols=np.array([0, 1], dtype=np.int32),
        realizations=np.array([0, 1], dtype=np.int32),
        rates_of_spread=np.array([0.3, 0.4], dtype=np.float32),
        fireline_intensities=np.array([1.0, 2.0], dtype=np.float32),
    )
    propagator.scheduler.add_event(1, SchedulerEvent(updates=updates))

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


def test_apply_updates_schedules_follow_up(monkeypatch):
    propagator = make_propagator(realizations=1)

    updates = UpdateBatch(
        rows=np.array([0], dtype=np.int32),
        cols=np.array([1], dtype=np.int32),
        realizations=np.array([0], dtype=np.int32),
        rates_of_spread=np.array([2.5], dtype=np.float32),
        fireline_intensities=np.array([7.5], dtype=np.float32),
    )

    future_time = 5
    stub = (
        np.array([future_time], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.4], dtype=np.float32),
        np.array([12.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "propagator.core.propagator.next_updates_fn", lambda *_, **__: stub
    )

    propagator._apply_updates(future_time, updates)

    assert propagator.fire[0, 1, 0] == 1
    assert propagator.ros[0, 1, 0] == pytest.approx(2.5)
    assert propagator.fireline_int[0, 1, 0] == pytest.approx(7.5)

    time = propagator.time
    assert time == future_time


def test_step_applies_event(monkeypatch):
    propagator = make_propagator(realizations=1)
    propagator.actions_moisture = np.full((2, 2), 0.5, dtype=np.float32)

    stub = (
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )
    monkeypatch.setattr(
        "propagator.core.propagator.next_updates_fn", lambda *_, **__: stub
    )

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
