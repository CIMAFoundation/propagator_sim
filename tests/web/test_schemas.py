from __future__ import annotations

import pytest
from pydantic import ValidationError

from propagator.web.schemas import (
    _EDGE_CACHE_BYTES_PER_CELL,
    ActionRequest,
    SimulateRequest,
    _estimate_memory_bytes,
)


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
        SimulateRequest(
            **base_kwargs(radius_km=50, cellsize=20, realizations=50)
        )


def test_rejects_combo_under_cell_budget_but_over_memory_budget():
    # This is exactly at the cell-realizations budget, but its estimated
    # front-heap and grid-state allocation exceeds the memory budget.
    with pytest.raises(ValidationError):
        SimulateRequest(
            **base_kwargs(radius_km=50, cellsize=20, realizations=10)
        )


def test_rust_core_estimate_adds_shared_edge_cache():
    grid_cells = 1000.0**2
    numba_bytes = _estimate_memory_bytes(grid_cells, 10, "numba")
    rust_bytes = _estimate_memory_bytes(grid_cells, 10, "rust")
    # the cache is shared by all realizations: one term per padded cell
    assert rust_bytes - numba_bytes == pytest.approx(
        _EDGE_CACHE_BYTES_PER_CELL * 1024.0**2
    )
    assert _estimate_memory_bytes(
        grid_cells, 20, "rust"
    ) - _estimate_memory_bytes(grid_cells, 20, "numba") == pytest.approx(
        rust_bytes - numba_bytes
    )


def test_rust_core_rejects_combo_the_numba_core_accepts():
    # 50 km / 25 m / 5 realizations: ~2.1 GiB on numba, ~4.5 GiB on rust
    # once the fully populated edge cache is counted
    SimulateRequest(**base_kwargs(radius_km=50, cellsize=25, realizations=5))
    with pytest.raises(ValidationError):
        SimulateRequest(
            **base_kwargs(
                core="rust", radius_km=50, cellsize=25, realizations=5
            )
        )


def test_accepts_small_high_resolution_combo():
    req = SimulateRequest(
        **base_kwargs(radius_km=2, cellsize=20, realizations=20)
    )
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


def test_rejects_time_resolution_that_rounds_to_zero_seconds():
    # a time_resolution_h this small would round to time_resolution_s=0,
    # which never advances the simulation loop (see web/runner.py::run_loop)
    with pytest.raises(ValidationError):
        SimulateRequest(
            **base_kwargs(time_limit_h=1.0, time_resolution_h=0.0001)
        )


def test_action_request_accepts_valid_line():
    action = ActionRequest(
        action_type="canadair", time_h=2.0, line=[(42.4, 12.1), (42.45, 12.15)]
    )
    assert action.action_type == "canadair"


def test_action_request_rejects_invalid_action_type():
    with pytest.raises(ValidationError):
        ActionRequest(
            action_type="bulldozer",
            time_h=1.0,
            line=[(42.4, 12.1), (42.45, 12.15)],
        )


def test_action_request_rejects_single_point_line():
    with pytest.raises(ValidationError):
        ActionRequest(action_type="canadair", time_h=1.0, line=[(42.4, 12.1)])


def test_simulate_request_rejects_action_time_beyond_limit():
    with pytest.raises(ValidationError):
        SimulateRequest(
            **base_kwargs(
                time_limit_h=2.0,
                actions=[
                    {
                        "action_type": "heavy_action",
                        "time_h": 5.0,
                        "line": [[42.4, 12.1], [42.45, 12.15]],
                    }
                ],
            )
        )


def test_simulate_request_accepts_action_within_limit():
    req = SimulateRequest(
        **base_kwargs(
            time_limit_h=6.0,
            actions=[
                {
                    "action_type": "waterline_action",
                    "time_h": 2.0,
                    "line": [[42.4, 12.1], [42.45, 12.15]],
                }
            ],
        )
    )
    assert len(req.actions) == 1
