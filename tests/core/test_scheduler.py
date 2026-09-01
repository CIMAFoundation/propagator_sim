from __future__ import annotations

import numpy as np

from propagator.core.constants import NO_FUEL
from propagator.core.models import UpdateBatch
from propagator.core.scheduler import Scheduler, SchedulerEvent


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
        merged.vegetation_changes, np.array([[NO_FUEL]], dtype=np.float32)
    )


def test_add_event_treats_nan_as_no_vegetation_change():
    scheduler = Scheduler(realizations=1)
    scheduler.add_event(
        4,
        SchedulerEvent(
            vegetation_changes=np.array([[2.0, np.nan]], dtype=np.float32)
        ),
    )
    scheduler.add_event(
        4,
        SchedulerEvent(
            vegetation_changes=np.array([[np.nan, 0.0]], dtype=np.float32)
        ),
    )

    _, merged = scheduler.pop()
    np.testing.assert_array_equal(
        merged.vegetation_changes, np.array([[2.0, 0.0]], dtype=np.float32)
    )
