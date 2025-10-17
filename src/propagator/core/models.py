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

# Integer coords array of shape (n, 3). We can’t encode the shape statically
# with stdlib typing, but we DO lock the dtype to integer families.
FireBehaviourUpdate = tuple[int, int, int, float, float]

UpdateBatchTuple = tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]


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

    bbox: Optional[tuple[int, int, int, int]] = field(init=False, default=None)

    def __post_init__(self):
        n = len(self.rows)
        if not (
            len(self.cols) == n
            and len(self.realizations) == n
            and len(self.rates_of_spread) == n
            and len(self.fireline_intensities) == n
        ):
            raise ValueError("All input arrays must have the same length")

        if n == 0:
            self.bbox = None
            return

        r0 = int(np.min(self.rows))
        c0 = int(np.min(self.cols))
        r1 = int(np.max(self.rows))
        c1 = int(np.max(self.cols))
        self.bbox = (r0, c0, r1, c1)

    def extend(self, other: "UpdateBatch") -> None:
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

        if self.bbox is None:
            self.bbox = other.bbox
            return

        if other.bbox is None:
            return

        r0, c0, r1, c1 = self.bbox
        or0, oc0, or1, oc1 = other.bbox
        self.bbox = (
            min(r0, or0),
            min(c0, oc0),
            max(r1, or1),
            max(c1, oc1),
        )


@dataclass(frozen=True)
class UpdateBatchWithTime:
    times: npt.NDArray[np.integer]
    rows: npt.NDArray[np.integer]
    cols: npt.NDArray[np.integer]
    realizations: npt.NDArray[np.integer]
    rates_of_spread: npt.NDArray[np.float32]
    fireline_intensities: npt.NDArray[np.float32]

    @staticmethod
    def from_tuple(data: UpdateBatchTuple) -> "UpdateBatchWithTime":
        (
            times,
            rows,
            cols,
            realizations,
            rates_of_spread,
            fireline_intensities,
        ) = data
        return UpdateBatchWithTime(
            times=times,
            rows=rows,
            cols=cols,
            realizations=realizations,
            rates_of_spread=rates_of_spread,
            fireline_intensities=fireline_intensities,
        )

    def split_by_time(self) -> dict[int, UpdateBatch]:
        result: dict[int, UpdateBatch] = {}
        for time in np.unique(self.times):
            index = self.times == time

            cols_at_time = self.cols[index]
            rows_at_time = self.rows[index]
            realizations_at_time = self.realizations[index]
            ros_at_time = self.rates_of_spread[index]
            fireline_intensity_at_time = self.fireline_intensities[index]

            update_at_time = UpdateBatch(
                rows_at_time,
                cols_at_time,
                realizations_at_time,
                ros_at_time,
                fireline_intensity_at_time,
            )
            result[time] = update_at_time

        return result


class PropagatorError(Exception):
    """Domain-specific error raised by PROPAGATOR."""


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
    ignition_mask : Optional[npt.NDArray[np.bool_]]
        Boolean mask of new ignition points.
    additional_moisture : Optional[npt.NDArray[np.floating]]
        Extra moisture to add to fuel (%), can be sparse.
    vegetation_changes : Optional[npt.NDArray[np.floating]]
        Raster of vegetation type overrides (NaN to skip).
    """

    time: int
    moisture: Optional[npt.NDArray[np.floating]] = None
    wind_dir: Optional[npt.NDArray[np.floating]] = None
    wind_speed: Optional[npt.NDArray[np.floating]] = None
    ignition_mask: Optional[npt.NDArray[np.bool_]] = None
    additional_moisture: Optional[npt.NDArray[np.floating]] = None
    vegetation_changes: Optional[npt.NDArray[np.floating]] = None


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
class PropagatorOutput:
    """Snapshot of simulation outputs at a given time step."""

    time: int  # seconds from simulation start
    fire_probability: npt.NDArray[np.floating]
    ros_mean: npt.NDArray[np.floating]
    ros_max: npt.NDArray[np.floating]
    fli_mean: npt.NDArray[np.floating]
    fli_max: npt.NDArray[np.floating]
    stats: PropagatorStats
