from __future__ import annotations

from propagator.web.schemas import SimulateRequest


def make_request(**overrides) -> SimulateRequest:
    defaults = dict(
        center_lat=42.42,
        center_lon=12.11,
        ignition_lat=42.42,
        ignition_lon=12.11,
        radius_km=1.0,
        cellsize=30.0,
        realizations=2,
        time_limit_h=2.0,
        time_resolution_h=1.0,
    )
    defaults.update(overrides)
    return SimulateRequest(**defaults)
