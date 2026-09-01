"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class UpdateBatch:
    rows: npt.NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )

    cols: npt.NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )

    realizations: npt.NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )

    rates_of_spread: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )

    fireline_intensities: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )

    def __post_init__(self):
        """Validate that all component arrays have the same length."""
        n = len(self.rows)
        if not (
            len(self.cols) == n
            and len(self.realizations) == n
            and len(self.rates_of_spread) == n
            and len(self.fireline_intensities) == n
        ):
            raise ValueError("All input arrays must have the same length")

    def extend(self, other: "UpdateBatch") -> None:
        """Merge another UpdateBatch into this one."""
        self.rows = np.concatenate([self.rows, other.rows])
        self.cols = np.concatenate([self.cols, other.cols])
        self.realizations = np.concatenate(
            [self.realizations, other.realizations]
        )
        self.rates_of_spread = np.concatenate(
            [self.rates_of_spread, other.rates_of_spread]
        )
        self.fireline_intensities = np.concatenate(
            [self.fireline_intensities, other.fireline_intensities]
        )


class PropagatorError(Exception):
    """Domain-specific error raised by PROPAGATOR."""


def validate_ignitions(ignitions):
    if isinstance(ignitions, list):
        for item in ignitions:
            if not (
                isinstance(item, tuple)
                and len(item) in (2, 3)
                and all(isinstance(x, int) for x in item)
            ):
                raise ValueError(
                    "Ignition list items must be (row, col) or (row, col, realization) tuples"
                )
    elif isinstance(ignitions, np.ndarray):
        if ignitions.ndim not in (2, 3):
            raise ValueError("Ignition ndarray must be 2D or 3D boolean array")
    else:
        raise ValueError(
            "Ignitions must be either a list of tuples or a boolean ndarray"
        )


@dataclass(frozen=True)
class BoundaryConditions:
    """
    Boundary conditions applied at or after a given simulation time.


    Attributes
    ----------
    time : int
        Simulation time the conditions refer to (seconds from simulation start).
    moisture : Optional[npt.NDArray[np.floating]]
        Fuel moisture map (%).
    wind_dir : Optional[npt.NDArray[np.floating]]
        Wind direction map (weather convention, degrees clockwise, north is 0).
    wind_speed : Optional[npt.NDArray[np.floating]]
        Wind speed map (km/h).
    ignitions : Optional[
        npt.NDArray[np.bool_]
        | list[tuple[int, int] | tuple[int, int, int]]
    ]
        Ignitions to enqueue. Accepts either a boolean raster (2D applies to
        every realization; 3D maps explicit `realization` planes) or a list of
        `(row, col)` / `(row, col, realization)` tuples.
    additional_moisture : Optional[npt.NDArray[np.floating]]
        Extra moisture to add to fuel (%), can be sparse.
    vegetation_changes : Optional[npt.NDArray[np.floating]]
        Raster of vegetation type overrides (NaN to skip).
    """

    time: int
    moisture: Optional[npt.ArrayLike] = None
    wind_dir: Optional[npt.ArrayLike] = None
    wind_speed: Optional[npt.ArrayLike] = None
    ignitions: Optional[
        npt.NDArray[np.bool_] | list[tuple[int, int] | tuple[int, int, int]]
    ] = None
    additional_moisture: Optional[npt.NDArray[np.floating]] = None
    vegetation_changes: Optional[npt.NDArray[np.floating]] = None

    def __post_init__(self):
        if self.time < 0:
            raise ValueError("BoundaryConditions time must be non-negative")
        if self.ignitions is not None:
            validate_ignitions(self.ignitions)


@dataclass(frozen=True)
class PropagatorStats:
    """Summary statistics for the current simulation state."""

    n_active: int
    area_mean: float
    area_50: float
    area_75: float
    area_90: float

    def to_dict(
        self, c_time: int, ref_date: datetime
    ) -> dict[str, float | int | str]:
        """Serialize stats with the current simulation time expressed in seconds."""
        return dict(
            c_time=c_time,
            ref_date=ref_date.isoformat(),
            n_active=self.n_active,
            area_mean=self.area_mean,
            area_50=self.area_50,
            area_75=self.area_75,
            area_90=self.area_90,
        )


@dataclass(frozen=True)
class CellArrivalSample:
    """Arrival-time sample at one grid cell, keyed by an opaque string id
    supplied by the caller (e.g. an OSM POI id). Deliberately carries no
    lat/lon/name/tags: identity/geospatial meaning is an io/web-layer
    concern, keeping the core engine I/O-agnostic (see CLAUDE.md)."""

    key: str
    row: int
    col: int
    reached: bool
    min_arrival_time: float  # seconds; NaN if not reached
    mean_arrival_time: float  # seconds; NaN if not reached


@dataclass(frozen=True)
class PropagatorOutput:
    """Snapshot of simulation outputs at a given time step.

    Every field is a value captured at `time`; the object stays valid
    after the simulator advances, so consumers may collect and compare
    snapshots across the run.
    """

    time: int  # seconds from simulation start
    fire_probability: npt.NDArray[np.floating]
    spotting_generation_probability: npt.NDArray[np.floating]
    spotting_receiving_probability: npt.NDArray[np.floating]
    mean_arrival_time: npt.NDArray[np.floating]
    min_arrival_time: npt.NDArray[np.floating]
    ros_mean: npt.NDArray[np.floating]
    ros_max: npt.NDArray[np.floating]
    fli_mean: npt.NDArray[np.floating]
    fli_max: npt.NDArray[np.floating]
    flame_length_mean: npt.NDArray[np.floating]
    flame_length_max: npt.NDArray[np.floating]
    stats: PropagatorStats
    poi_arrival: tuple[CellArrivalSample, ...] = field(default_factory=tuple)
