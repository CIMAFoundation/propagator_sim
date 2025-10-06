import numpy as np

from propagator.core import FUEL_SYSTEM_LEGACY, Propagator

fuel_system = FUEL_SYSTEM_LEGACY


def test_propagator_basic():
    # Minimal 2x2 grid, 1 realization, no spotting
    veg = np.array([[1, 1], [1, 1]], dtype=np.int32)
    dem = np.zeros((2, 2), dtype=np.float32)
    propagator = Propagator(
        veg=veg, dem=dem, realizations=1, do_spotting=False, fuels=fuel_system
    )
    # Check initial fire probability is all zeros
    assert np.all(propagator.compute_fire_probability() == 0)
    # Check initial RoS max is all zeros
    assert np.all(propagator.compute_ros_max() == 0)


def test_get_moisture_no_actions():
    veg = np.ones((2, 2), dtype=np.int32)
    dem = np.zeros((2, 2), dtype=np.float32)
    propagator = Propagator(
        veg=veg, dem=dem, realizations=1, do_spotting=False, fuels=fuel_system
    )
    # Manually set moisture
    propagator.moisture = np.full((2, 2), 0.1)
    propagator.actions_moisture = None
    result = propagator._get_moisture()
    np.testing.assert_allclose(result, np.full((2, 2), 0.1))


def test_get_moisture_with_actions():
    veg = np.ones((2, 2), dtype=np.int32)
    dem = np.zeros((2, 2), dtype=np.float32)
    propagator = Propagator(
        veg=veg, dem=dem, realizations=1, do_spotting=False, fuels=fuel_system
    )
    propagator.moisture = np.full((2, 2), 0.1)
    propagator.actions_moisture = np.full((2, 2), 0.05)
    result = propagator._get_moisture()
    np.testing.assert_allclose(result, np.full((2, 2), 0.15))
