from datetime import datetime

import numpy as np

from propagator.core.models import (
    PropagatorStats,
    UpdateBatch,
)


def test_update_batch_extend_concatenates_fields():
    base = UpdateBatch(
        rows=np.array([0], dtype=np.int32),
        cols=np.array([1], dtype=np.int32),
        realizations=np.array([0], dtype=np.int32),
        rates_of_spread=np.array([0.5], dtype=np.float32),
        fireline_intensities=np.array([10.0], dtype=np.float32),
    )
    extra = UpdateBatch(
        rows=np.array([2, 3], dtype=np.int32),
        cols=np.array([4, 5], dtype=np.int32),
        realizations=np.array([1, 2], dtype=np.int32),
        rates_of_spread=np.array([1.1, 1.2], dtype=np.float32),
        fireline_intensities=np.array([20.0, 30.0], dtype=np.float32),
    )

    base.extend(extra)

    np.testing.assert_array_equal(
        base.rows, np.array([0, 2, 3], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        base.cols, np.array([1, 4, 5], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        base.realizations, np.array([0, 1, 2], dtype=np.int32)
    )
    np.testing.assert_allclose(
        base.rates_of_spread,
        np.array([0.5, 1.1, 1.2], dtype=np.float32),
    )
    np.testing.assert_allclose(
        base.fireline_intensities,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )


def test_propagator_stats_to_dict_contains_expected_fields():
    stats = PropagatorStats(
        n_active=3,
        area_mean=4.5,
        area_50=3.0,
        area_75=2.0,
        area_90=1.0,
    )

    result = stats.to_dict(c_time=7, ref_date=datetime(2024, 1, 1))

    assert result["c_time"] == 7
    assert result["ref_date"] == "2024-01-01T00:00:00"
    assert result["n_active"] == 3
    assert result["area_mean"] == 4.5
    assert result["area_50"] == 3.0
    assert result["area_75"] == 2.0
    assert result["area_90"] == 1.0
