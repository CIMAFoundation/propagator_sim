"""Pydantic request/response models for the web API.

Field bounds double as the guardrails that keep a run responsive on a
local, single-user machine (see `_check_compute_budget`).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Rough compute budget: grid cells (width * height) times realizations.
# A 15 km-radius/30 m-cellsize grid (1000x1000) at 10 realizations is
# ~1e7 and runs in well under a minute; this caps combinations an order
# of magnitude above that, which already take several minutes locally.
CELL_REALIZATION_BUDGET = 2.5e8

# Peak-memory guardrail adapted to this branch's block-sparse engine. It
# includes fully populated state tiles, the dense fold used to produce web
# results, shared input/state grids, tile indexes, and the initial front heap.
_TILE_SIZE = 32
_FRONT_INITIAL_CAPACITY = 4096
_FRONT_HEAP_BYTES_PER_EVENT = 5 * 4
_TILED_STATE_BYTES_PER_CELL = 1 + 4 + 4 + 4
_FOLD_BYTES_PER_CELL = 48
_SHARED_GRID_BYTES_PER_CELL = 8 + 5 * 4
_TILE_INDEX_BYTES = 4
MAX_ESTIMATED_MEMORY_BYTES = 4 * 1024**3  # 4 GiB


def _estimate_memory_bytes(grid_cells: float, realizations: int) -> float:
    grid_side = math.ceil(math.sqrt(grid_cells))
    tiles_per_side = math.ceil(grid_side / _TILE_SIZE)
    padded_cells = (tiles_per_side * _TILE_SIZE) ** 2
    return (
        realizations
        * (
            _TILED_STATE_BYTES_PER_CELL * padded_cells
            + _TILE_INDEX_BYTES * tiles_per_side**2
            + _FRONT_HEAP_BYTES_PER_EVENT * _FRONT_INITIAL_CAPACITY
        )
        + (_FOLD_BYTES_PER_CELL + _SHARED_GRID_BYTES_PER_CELL) * grid_cells
    )


class ActionRequest(BaseModel):
    """A firefighting action (see `propagator.io.actions`), drawn as a
    line on the map and scheduled at `time_h` into the simulation."""

    action_type: Literal[
        "waterline_action", "canadair", "helicopter", "heavy_action"
    ]
    time_h: float = Field(..., ge=0)
    line: list[tuple[float, float]] = Field(
        ..., min_length=2, description="[(lat, lon), ...]"
    )


class SimulateRequest(BaseModel):
    core: Literal["numba", "rust"] = "numba"
    center_lat: float = Field(..., ge=-90, le=90)
    center_lon: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(15.0, gt=0, le=50)
    cellsize: float = Field(30.0, ge=20, le=100)

    ignition_lat: float = Field(..., ge=-90, le=90)
    ignition_lon: float = Field(..., ge=-180, le=180)

    wind_dir: float = Field(
        0.0, ge=0, lt=360, description="Degrees, clockwise from north"
    )
    wind_speed: float = Field(20.0, ge=0, description="km/h")
    moisture: float = Field(10.0, ge=0, le=100, description="Percent")

    realizations: int = Field(10, ge=1, le=50)
    do_spotting: bool = False

    time_limit_h: float = Field(6.0, gt=0, le=48)
    time_resolution_h: float = Field(1.0, gt=0, le=6)

    isochrone_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 0.9]
    )

    actions: list[ActionRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_compute_budget(self) -> "SimulateRequest":
        half_cells = (self.radius_km * 1000.0) / self.cellsize
        grid_cells = (2 * half_cells) ** 2
        cost = grid_cells * self.realizations
        if cost > CELL_REALIZATION_BUDGET:
            raise ValueError(
                "This combination of radius_km/cellsize/realizations is "
                f"too large for an interactive local run (estimated "
                f"{cost:,.0f} cell-realizations, budget "
                f"{CELL_REALIZATION_BUDGET:,.0f}). Lower the radius, "
                "increase the cellsize, or reduce realizations."
            )
        estimated_bytes = _estimate_memory_bytes(grid_cells, self.realizations)
        if estimated_bytes > MAX_ESTIMATED_MEMORY_BYTES:
            raise ValueError(
                "This combination of radius_km/cellsize/realizations would "
                "allocate an estimated "
                f"{estimated_bytes / 1024**3:.2f} GiB (budget "
                f"{MAX_ESTIMATED_MEMORY_BYTES / 1024**3:.0f} GiB) for the "
                "simulation and result state. Lower the radius, increase "
                "the cellsize, or reduce realizations."
            )
        if self.time_resolution_h > self.time_limit_h:
            raise ValueError("time_resolution_h must not exceed time_limit_h")
        if self.time_resolution_s < 1:
            raise ValueError(
                "time_resolution_h too small: must resolve to at least 1 "
                "second, or the simulation loop never advances"
            )
        for action in self.actions:
            if action.time_h > self.time_limit_h:
                raise ValueError(
                    f"action time_h={action.time_h} exceeds time_limit_h="
                    f"{self.time_limit_h}"
                )
        return self

    @property
    def time_limit_s(self) -> int:
        return int(round(self.time_limit_h * 3600))

    @property
    def time_resolution_s(self) -> int:
        return int(round(self.time_resolution_h * 3600))


class JobSummary(BaseModel):
    id: str
    status: str
    current_time_s: int
    time_limit_s: int
    warning: str | None = None
    error: str | None = None


class FrameStats(BaseModel):
    time_s: int
    n_active: int
    area_mean: float
    area_50: float
    area_75: float
    area_90: float


class JobFrames(BaseModel):
    id: str
    status: str
    bounds_wgs84: tuple[float, float, float, float] | None = None
    frame_times_s: list[int]
    stats_history: list[FrameStats]


class Isochrone(BaseModel):
    threshold: float
    coordinates: list[list[list[float]]]  # MultiLineString coordinates


class FrameOut(BaseModel):
    time_s: int
    isochrones: list[Isochrone]
    stats: FrameStats
