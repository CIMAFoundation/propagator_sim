from __future__ import annotations

import numpy as np

from propagator.core.constants import NO_FUEL
from propagator.core.models import UpdateBatch, UpdateBatchWithTime
from propagator.core.scheduler import Scheduler, SchedulerEvent


def test_push_updates_groups_by_time():
    scheduler = Scheduler(realizations=2)

    batch = UpdateBatchWithTime.from_tuple(
        (
            np.array([2, 1, 1], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
            np.array([0, 1, 1], dtype=np.int32),
            np.array([0.2, 0.5, 0.6], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        )
    )

    scheduler.push_updates(batch)

    assert len(scheduler) == 2
    assert scheduler.next_time() == 1

    time_one, event_one = scheduler.pop()
    assert time_one == 1
    np.testing.assert_array_equal(
        event_one.updates.rows, np.array([1, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        event_one.updates.cols, np.array([2, 3], dtype=np.int32)
    )

    time_two, event_two = scheduler.pop()
    assert time_two == 2
    np.testing.assert_array_equal(
        event_two.updates.rows, np.array([0], dtype=np.int32)
    )


def test_add_event_merges_updates_and_actions():
    scheduler = Scheduler(realizations=2)

    event_a = SchedulerEvent(
        updates=UpdateBatch(
            rows=np.array([0], dtype=np.int32),
            cols=np.array([1], dtype=np.int32),
            realizations=np.array([0], dtype=np.int32),
            rates_of_spread=np.array([0.7], dtype=np.float32),
            fireline_intensities=np.array([8.0], dtype=np.float32),
        ),
        moisture=np.full((1, 1), 0.3, dtype=np.float32),
        additional_moisture=np.full((1, 1), 0.1, dtype=np.float32),
        vegetation_changes=np.full((1, 1), 2.0, dtype=np.float32),
    )
    scheduler.add_event(4, event_a)

    event_b = SchedulerEvent(
        updates=UpdateBatch(
            rows=np.array([1], dtype=np.int32),
            cols=np.array([2], dtype=np.int32),
            realizations=np.array([1], dtype=np.int32),
            rates_of_spread=np.array([1.2], dtype=np.float32),
            fireline_intensities=np.array([15.0], dtype=np.float32),
        ),
        wind_dir=np.full((1, 1), 0.9, dtype=np.float32),
        wind_speed=np.full((1, 1), 12.0, dtype=np.float32),
        additional_moisture=np.full((1, 1), 0.2, dtype=np.float32),
        vegetation_changes=np.full((1, 1), NO_FUEL, dtype=np.float32),
    )
    scheduler.add_event(4, event_b)

    time, merged = scheduler.pop()
    assert time == 4

    np.testing.assert_array_equal(
        merged.updates.rows, np.array([0, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        merged.updates.cols, np.array([1, 2], dtype=np.int32)
    )
    np.testing.assert_allclose(
        merged.additional_moisture, np.array([[0.3]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        merged.wind_dir, np.array([[0.9]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        merged.wind_speed, np.array([[12.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        merged.vegetation_changes, np.array([[2.0]], dtype=np.float32)
    )


def test_active_returns_unique_realizations():
    scheduler = Scheduler(realizations=3)

    scheduled = UpdateBatchWithTime.from_tuple(
        (
            np.array([1, 2, 2], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0.2, 0.3, 0.4], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        )
    )

    scheduler.push_updates(scheduled)

    active = scheduler.active()
    np.testing.assert_array_equal(active, np.array([0, 1, 2], dtype=np.int32))
