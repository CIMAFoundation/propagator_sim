import numpy as np

from propagator.core import Propagator
from propagator.core.numba_backend import (
    fuelsystem_from_dict as numba_fuelsystem_from_dict,
)
from propagator.core.numba_backend import (
    get_p_moisture_fn as numba_get_p_moisture_fn,
)
from propagator.core.numba_backend import (
    get_p_time_fn as numba_get_p_time_fn,
)
from propagator.core.numba_backend import (
    next_updates_fn as numba_next_updates_fn,
)
from propagator.core.numpy_backend import (
    fuelsystem_from_dict as numpy_fuelsystem_from_dict,
)
from propagator.core.numpy_backend import (
    get_p_moisture_fn as numpy_get_p_moisture_fn,
)
from propagator.core.numpy_backend import (
    get_p_time_fn as numpy_get_p_time_fn,
)
from propagator.core.numpy_backend import (
    next_updates_fn as numpy_next_updates_fn,
)


def test_numpy_next_updates_matches_numba():
    fuels_dict = {
        1: {
            "name": "test",
            "v0": 60.0,
            "d0": 1.0,
            "d1": 0.0,
            "hhv": 18000.0,
            "spread_probability": {1: 10.0},
            "spotting": False,
        }
    }

    rows = np.array([0], dtype=np.int32)
    cols = np.array([0], dtype=np.int32)
    realizations = np.array([0], dtype=np.int32)
    cellsize = 1.0
    time = 0
    veg = np.ones((2, 2), dtype=np.int32)
    dem = np.zeros((2, 2), dtype=np.float64)
    fire = np.zeros((2, 2, 1), dtype=np.int8)
    fire[0, 0, 0] = 1
    moisture = np.zeros((2, 2), dtype=np.float64)
    wind_dir = np.zeros((2, 2), dtype=np.float64)
    wind_speed = np.zeros((2, 2), dtype=np.float64)

    np.random.seed(0)
    numba_fs = numba_fuelsystem_from_dict(fuels_dict)
    numba_time_fn = numba_get_p_time_fn("standard")
    numba_moist_fn = numba_get_p_moisture_fn("trucchia")
    nb_updates = numba_next_updates_fn(
        rows,
        cols,
        realizations,
        cellsize,
        time,
        veg,
        dem,
        fire,
        moisture,
        wind_dir,
        wind_speed,
        numba_fs,
        numba_time_fn,
        numba_moist_fn,
    )

    np.random.seed(0)
    numpy_fs = numpy_fuelsystem_from_dict(fuels_dict)
    numpy_time_fn = numpy_get_p_time_fn("standard")
    numpy_moist_fn = numpy_get_p_moisture_fn("trucchia")
    np_updates = numpy_next_updates_fn(
        rows,
        cols,
        realizations,
        cellsize,
        time,
        veg,
        dem,
        fire,
        moisture,
        wind_dir,
        wind_speed,
        numpy_fs,
        numpy_time_fn,
        numpy_moist_fn,
    )

    for numba_arr, numpy_arr in zip(nb_updates, np_updates):
        np.testing.assert_allclose(numba_arr, numpy_arr, equal_nan=True)


def test_propagator_uses_numpy_backend():
    veg = np.ones((2, 2), dtype=np.int32)
    dem = np.zeros((2, 2), dtype=np.float64)

    propagator = Propagator(
        veg=veg,
        dem=dem,
        backend="numpy",
        do_spotting=False,
    )

    assert propagator.backend == "numpy"
    assert propagator.backend_module.__name__ == "propagator.core.numpy"
    assert propagator._next_updates_fn is numpy_next_updates_fn
    assert isinstance(propagator.fuels, propagator.backend_module.FuelSystem)
    assert callable(propagator.p_time_fn)
    assert callable(propagator.p_moist_fn)
    assert not np.any(propagator.fuels.spotting)
