"""Pydantic request/response models for the web API.

Field bounds double as the guardrails that keep a run responsive on a
local, single-user machine (see `_check_compute_budget`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Rough compute budget: grid cells (width * height) times realizations.
# A 15 km-radius/30 m-cellsize grid (1000x1000) at 10 realizations is
# ~1e7 and runs in well under a minute; this caps combinations an order
# of magnitude above that, which already take several minutes locally.
CELL_REALIZATION_BUDGET = 2.5e8


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
