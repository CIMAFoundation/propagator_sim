from __future__ import annotations

import pytest
from pydantic import ValidationError

from propagator.web.schemas import SimulateRequest


def base_kwargs(**overrides):
    kwargs = dict(
        center_lat=42.42,
        center_lon=12.11,
        ignition_lat=42.45,
        ignition_lon=12.25,
    )
    kwargs.update(overrides)
    return kwargs


def test_defaults_are_accepted():
    req = SimulateRequest(**base_kwargs())
    assert req.radius_km == 15.0
    assert req.time_limit_s == 6 * 3600
    assert req.time_resolution_s == 3600


def test_rejects_huge_radius_realizations_combo():
    with pytest.raises(ValidationError):
        SimulateRequest(**base_kwargs(radius_km=50, cellsize=20, realizations=50))


def test_accepts_small_high_resolution_combo():
    req = SimulateRequest(**base_kwargs(radius_km=2, cellsize=20, realizations=20))
    assert req.radius_km == 2


def test_rejects_resolution_coarser_than_limit():
    with pytest.raises(ValidationError):
        SimulateRequest(**base_kwargs(time_limit_h=1.0, time_resolution_h=2.0))


def test_rejects_out_of_range_wind_dir():
    with pytest.raises(ValidationError):
        SimulateRequest(**base_kwargs(wind_dir=400))


def test_rejects_zero_realizations():
    with pytest.raises(ValidationError):
        SimulateRequest(**base_kwargs(realizations=0))
