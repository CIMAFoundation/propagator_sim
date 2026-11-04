"""Flame height and radiant reach outputs agree with their models on
both cores."""

from __future__ import annotations

import numpy as np
import pytest

from propagator.core import BoundaryConditions, Propagator  # type: ignore
from propagator.core.constants import (  # type: ignore
    FUEL_SYSTEM_LEGACY_DICT,
)


def byram(fli: np.ndarray) -> np.ndarray:
    return np.where(fli > 0, 0.0775 * np.maximum(fli, 0) ** 0.46, 0.0)


def reach(fli: np.ndarray) -> np.ndarray:
    return np.where(fli > 0, np.clip(4 * byram(fli), 10.0, 200.0), 0.0)


def make_simulator(core: str, realizations: int, size: int = 40):
    veg = np.full((size, size), 5, dtype=np.int32)
    dem = np.zeros((size, size), dtype=np.float32)
    kwargs = dict(
        veg=veg,
        dem=dem,
        realizations=realizations,
        do_spotting=False,
        seed=7,
        out_of_bounds_mode="ignore",
    )
    if core == "rust":
        pytest.importorskip("propagator_rust")
        from propagator.rust_core import Propagator as RustPropagator

        simulator = RustPropagator(
            fuels_dict=FUEL_SYSTEM_LEGACY_DICT, **kwargs
        )
    else:
        simulator = Propagator(**kwargs)
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=np.full((size, size), 10.0, dtype=np.float32),
            wind_dir=np.full((size, size), 45.0, dtype=np.float32),
            wind_speed=np.full((size, size), 15.0, dtype=np.float32),
            ignitions=[(size // 2, size // 2)],
        )
    )
    return simulator


@pytest.mark.parametrize("core", ["numba", "rust"])
@pytest.mark.parametrize("realizations", [1, 4])
def test_flame_height_outputs_follow_byram(core, realizations):
    simulator = make_simulator(core, realizations)
    simulator.step(seconds=2 * 3600)
    out = simulator.get_output()

    burnt = out.fire_probability > 0
    assert burnt.sum() > 1

    np.testing.assert_allclose(
        out.flame_height_max, byram(out.fli_max), rtol=1e-5
    )
    assert np.all(np.isnan(out.flame_height_mean[~burnt]))
    assert np.all(out.flame_height_max[~burnt] == 0)

    mean = out.flame_height_mean[burnt]
    of_mean = byram(out.fli_mean[burnt])
    if realizations == 1:
        np.testing.assert_allclose(mean, of_mean, rtol=1e-5)
    else:
        # Byram is concave: mean flame height <= flame height of the mean
        assert np.all(mean <= of_mean * (1 + 1e-5))
        assert np.all(mean <= out.flame_height_max[burnt] * (1 + 1e-5))


@pytest.mark.parametrize("core", ["numba", "rust"])
@pytest.mark.parametrize("realizations", [1, 4])
def test_radiant_reach_outputs_follow_flame_height(core, realizations):
    simulator = make_simulator(core, realizations)
    simulator.step(seconds=2 * 3600)
    out = simulator.get_output()

    burnt = out.fire_probability > 0
    assert burnt.sum() > 1

    # monotonic in intensity: the max reach is the reach of the max
    np.testing.assert_allclose(
        out.radiant_reach_max, reach(out.fli_max), rtol=1e-5
    )
    assert np.all(np.isnan(out.radiant_reach_mean[~burnt]))
    assert np.all(out.radiant_reach_max[~burnt] == 0)

    mean = out.radiant_reach_mean[burnt]
    if realizations == 1:
        np.testing.assert_allclose(mean, reach(out.fli_mean[burnt]), rtol=1e-5)
    else:
        assert np.all(mean >= 0)
        assert np.all(mean <= out.radiant_reach_max[burnt] * (1 + 1e-5))
