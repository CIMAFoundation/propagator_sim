"""Pydantic request/response models for the web API.

Field bounds double as the guardrails that keep a run responsive on a
local, single-user machine (see `_check_compute_budget`).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from propagator.io.osm_poi import DEFAULT_MAX_POIS, POI_CATEGORIES

# Rough compute budget: grid cells (width * height) times realizations.
# A 15 km-radius/30 m-cellsize grid (1000x1000) at 10 realizations is
# ~1e7 and runs in well under a minute; this caps combinations an order
# of magnitude above that, which already take several minutes locally.
CELL_REALIZATION_BUDGET = 2.5e8

# Kept in sync manually with `propagator.io.osm_poi.POI_CATEGORIES`
# (Literal needs a static value list, so it can't just reuse the tuple).
PoiCategory = Literal[
    "hospital",
    "fire_station",
    "police",
    "school",
    "emergency",
    "road",
    "building",
    "power",
]
assert set(PoiCategory.__args__) == set(POI_CATEGORIES), (
    "PoiCategory has drifted from propagator.io.osm_poi.POI_CATEGORIES"
)

# Byte-accurate memory budget, mirroring the arrays
# `Propagator.__post_init__` allocates (core/propagator.py): the
# front-event heap (5 arrays, int32/float32, sized by
# front_capacity_factor * grid_cells) plus per-cell grid state (fire,
# arrival_time, ros, fireline_int, and, if spotting is on,
# spotting_generation/receiving). Complements CELL_REALIZATION_BUDGET,
# which ignores front_capacity_factor and spotting.
DEFAULT_FRONT_CAPACITY_FACTOR = 2.0
_FRONT_HEAP_BYTES_PER_CELL = (
    5 * 4
)  # times/rows/cols (int32) + ros/fli (float32)
_GRID_STATE_BYTES_PER_CELL = (
    1 + 4 + 4 + 4
)  # fire (int8) + arrival/ros/fli (4B)
_SPOTTING_BYTES_PER_CELL = 2 * 4  # generation/receiving (uint32)
MAX_ESTIMATED_MEMORY_BYTES = 4 * 1024**3  # 4 GiB


def _estimate_memory_bytes(
    grid_cells: float, realizations: int, do_spotting: bool
) -> float:
    front_capacity = math.ceil(DEFAULT_FRONT_CAPACITY_FACTOR * grid_cells)
    per_cell = _GRID_STATE_BYTES_PER_CELL
    if do_spotting:
        per_cell += _SPOTTING_BYTES_PER_CELL
    return realizations * (
        _FRONT_HEAP_BYTES_PER_CELL * front_capacity + per_cell * grid_cells
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
    center_lat: float = Field(..., ge=-90, le=90)
    center_lon: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(10.0, gt=0, le=50)
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

    time_limit_h: float = Field(12.0, gt=0, le=48)
    time_resolution_h: float = Field(1.0, gt=0, le=6)

    isochrone_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 0.9]
    )

    actions: list[ActionRequest] = Field(default_factory=list)

    include_pois: bool = Field(
        True,
        description="Fetch OpenStreetMap points of interest (hospitals, "
        "schools, fire/police stations, power infrastructure, major "
        "roads, buildings) in the area and report fire arrival at each.",
    )
    max_pois: int = Field(
        DEFAULT_MAX_POIS,
        ge=1,
        le=5000,
        description="Cap on the number of POIs fetched for the area "
        "(higher-priority categories and those closest to the center "
        "are kept first when there would be more).",
    )
    poi_categories: list[PoiCategory] = Field(
        default_factory=lambda: list(POI_CATEGORIES),
        description="Which POI categories to fetch/report. Defaults to "
        "every category.",
    )

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
        estimated_bytes = _estimate_memory_bytes(
            grid_cells, self.realizations, self.do_spotting
        )
        if estimated_bytes > MAX_ESTIMATED_MEMORY_BYTES:
            raise ValueError(
                "This combination of radius_km/cellsize/realizations/"
                "do_spotting would allocate an estimated "
                f"{estimated_bytes / 1024**3:.2f} GiB (budget "
                f"{MAX_ESTIMATED_MEMORY_BYTES / 1024**3:.0f} GiB) for the "
                "front-event heap and per-cell grid state. Lower the "
                "radius, increase the cellsize, reduce realizations, or "
                "disable spotting."
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
    poi_warning: str | None = None
    error: str | None = None


class FrameStats(BaseModel):
    time_s: int
    n_active: int
    area_mean: float
    area_50: float
    area_75: float
    area_90: float


class POIOut(BaseModel):
    id: str
    name: str | None
    category: str
    lat: float
    lon: float
    voltage: str | None = None
    operator: str | None = None
    # Full (lat, lon) vertex list for a line/polygon POI (e.g. a power
    # line), so the map can draw its actual path instead of a single
    # point; None for a plain point POI.
    geometry: list[tuple[float, float]] | None = None


class JobFrames(BaseModel):
    id: str
    status: str
    bounds_wgs84: tuple[float, float, float, float] | None = None
    frame_times_s: list[int]
    stats_history: list[FrameStats]
    pois: list[POIOut] = Field(default_factory=list)


class Isochrone(BaseModel):
    threshold: float
    coordinates: list[list[list[float]]]  # MultiLineString coordinates


class POIArrivalOut(BaseModel):
    id: str
    name: str | None
    category: str
    lat: float
    lon: float
    voltage: str | None = None
    operator: str | None = None
    reached: bool
    arrival_time_h: float | None


class FrameOut(BaseModel):
    time_s: int
    isochrones: list[Isochrone]
    stats: FrameStats
    poi_arrival: list[POIArrivalOut] = Field(default_factory=list)
