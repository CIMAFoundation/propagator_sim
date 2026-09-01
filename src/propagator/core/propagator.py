"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from propagator.core.constants import (
    BYRAM_FLAME_LENGTH_COEFF,
    BYRAM_FLAME_LENGTH_EXPONENT,
    CELLSIZE,
    MOISTURE_MODEL_DEFAULT,
    REALIZATIONS,
    ROS_DEFAULT,
)
from propagator.core.models import (
    BoundaryConditions,
    CellArrivalSample,
    PropagatorOutput,
    PropagatorStats,
    UpdateBatch,
)
from propagator.core.numba import (
    FUEL_SYSTEM_LEGACY,
    FuelSystem,
    advance_front_until,
    get_p_moisture_fn,
    get_p_time_fn,
)
from propagator.core.scheduler import Scheduler, SchedulerEvent

from .utils import upcast_to_ndarray


class PropagatorOutOfBoundsError(Exception):
    """Custom error for out-of-bounds updates in the Propagator."""

    pass


def _byram_flame_length(
    fireline_int: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Vectorized Byram (1959) flame length (m) from fireline
    intensity (kW/m)."""
    # Computed inside a single transient buffer: `np.maximum` already
    # copies, so the power and scale steps run in place on that copy and
    # never touch the caller's array. `compute_flame_length_mean` calls
    # this on the full (rows, cols, realizations) intensity grid once per
    # `get_output()`, where the naive
    # `COEFF * np.power(np.maximum(x, 0), EXP)` would hold two such
    # copies at once. One full-size transient is irreducible here (the
    # per-realization flame lengths have to exist to be averaged); it is
    # *not* covered by `web.schemas.MAX_ESTIMATED_MEMORY_BYTES`, which
    # budgets only the arrays `__post_init__` allocates.
    out = np.maximum(fireline_int, 0.0)
    np.power(out, BYRAM_FLAME_LENGTH_EXPONENT, out=out)
    out *= BYRAM_FLAME_LENGTH_COEFF
    return out


@dataclass
class Propagator:
    """Stochastic cellular wildfire spread simulator.

    PROPAGATOR evolves a binary fire state over a regular grid for a
    configurable number of realizations.
    Spread depends on vegetation, topography and environmental drivers
    (wind, moisture) through pluggable probability and travel-time functions.

    Attributes
    ----------

    veg : numpy.ndarray
        2D array of vegetation codes as defined in the provided FuelSystem
    dem : numpy.ndarray
        2D array of elevation values (meters above sea level).
    fuels: FuelSystem, optional
        Object defining fuels types and fire propagation
        probability between fuel types
    cellsize : float, optional
        The size of lattice (meters).
    do_spotting : bool, optional
        Whether to enable fire-spotting in the model.
    realizations : int, optional
        Number of stochastic realizations to simulate.
    front_capacity_factor : float, optional
        Multiplier applied to the number of grid cells to size the
        per-realization front (event) heap. The default of 2.0 leaves
        headroom for the pending updates piled up by spreading and,
        especially, spotting. Default is 2.0.
    front_capacity : int, optional
        Explicit front heap capacity in events per realization,
        overriding ``front_capacity_factor`` when given.
    p_time_fn: Any, optional
        The function to compute the spread time (must be jit-compiled).
        Units are compliant with other functions.
            signature: (v0: float, dh: float, angle_to: float, dist: float,
            moist: float, w_dir: float, w_speed: float) -> tuple[float, float]
    p_moist_fn: Any, optional
        The function to compute the moisture probability (must be jit-compiled)
        Units are compliant with other functions.
            signature: (moist: float) -> float

    out_of_bounds_mode: Literal["ignore", "error"], optional
        Whether to raise an error if out-of-bounds updates are detected.
        Default is "error".
    """

    # domain parameters for the simulation

    # input
    veg: npt.NDArray[np.integer]
    dem: npt.NDArray[np.floating]

    # set fuels
    fuels: FuelSystem = field(default_factory=lambda: FUEL_SYSTEM_LEGACY)

    # simulation settings
    cellsize: float = field(default=CELLSIZE)
    do_spotting: bool = field(default=False)
    realizations: int = field(default=REALIZATIONS)

    # capacity of the per-realization front (event) heap: pending spread
    # updates accumulate here (spotting in particular enqueues embers
    # from *every* burning cell, so the queue can exceed the number of
    # cells). Scaled by `front_capacity_factor`; set `front_capacity` to
    # override the computed size explicitly.
    front_capacity_factor: float = field(default=2.0)
    front_capacity: Optional[int] = field(default=None)

    # selected simulation functions
    p_time_fn: Any = field(default=get_p_time_fn(ROS_DEFAULT))
    p_moist_fn: Any = field(default=get_p_moisture_fn(MOISTURE_MODEL_DEFAULT))

    # scheduler object
    scheduler: Scheduler = field(init=False)
    _front_times: npt.NDArray[np.int32] = field(init=False)
    _front_rows: npt.NDArray[np.int32] = field(init=False)
    _front_cols: npt.NDArray[np.int32] = field(init=False)
    _front_ros: npt.NDArray[np.float32] = field(init=False)
    _front_fli: npt.NDArray[np.float32] = field(init=False)
    _front_sizes: npt.NDArray[np.int32] = field(init=False)
    _front_overflow: npt.NDArray[np.int8] = field(init=False)
    _front_capacity: int = field(init=False, default=0)

    # simulation state
    time: int = field(init=False, default=0)
    fire: npt.NDArray[np.int8] = field(init=False)
    spotting_generation: npt.NDArray[np.bool_] | None = field(init=False)
    spotting_receiving: npt.NDArray[np.bool_] | None = field(init=False)
    arrival_time: npt.NDArray[np.int32] = field(init=False)
    ros: npt.NDArray[np.float32] = field(init=False)
    fireline_int: npt.NDArray[np.float32] = field(init=False)
    moisture: npt.NDArray[np.floating] = field(init=False)
    wind_dir: npt.NDArray[np.floating] = field(init=False)
    wind_speed: npt.NDArray[np.floating] = field(init=False)
    actions_moisture: npt.NDArray[np.floating] | None = field(
        default=None, init=False
    )  # additional moisture due to fighting actions
    # (ideally it should decay over time)

    out_of_bounds_mode: Literal["ignore", "raise"] = "raise"

    def __post_init__(self):
        """Allocate internal state arrays based
        on the vegetation grid shape."""
        shape = self.veg.shape
        self.scheduler = Scheduler(realizations=self.realizations)
        if self.front_capacity_factor <= 0:
            raise ValueError(
                "front_capacity_factor must be strictly positive, got "
                f"{self.front_capacity_factor}"
            )
        if self.front_capacity is not None:
            if self.front_capacity <= 0:
                raise ValueError(
                    "front_capacity must be strictly positive, got "
                    f"{self.front_capacity}"
                )
            self._front_capacity = int(self.front_capacity)
        else:
            self._front_capacity = int(
                math.ceil(self.front_capacity_factor * self.veg.size)
            )
        front_heap_cells = self._front_capacity * self.realizations
        if front_heap_cells > np.iinfo(np.int32).max:
            raise ValueError(
                "front_capacity * realizations = "
                f"{front_heap_cells:,} overflows a 32-bit index "
                f"(limit {np.iinfo(np.int32).max:,}); lower "
                "front_capacity/front_capacity_factor or realizations"
            )
        self._front_times = np.zeros(
            (self.realizations, self._front_capacity), dtype=np.int32
        )
        self._front_rows = np.zeros(
            (self.realizations, self._front_capacity), dtype=np.int32
        )
        self._front_cols = np.zeros(
            (self.realizations, self._front_capacity), dtype=np.int32
        )
        self._front_ros = np.zeros(
            (self.realizations, self._front_capacity), dtype=np.float32
        )
        self._front_fli = np.zeros(
            (self.realizations, self._front_capacity), dtype=np.float32
        )
        self._front_sizes = np.zeros((self.realizations,), dtype=np.int32)
        self._front_overflow = np.zeros((self.realizations,), dtype=np.int8)
        self.fire = np.zeros(shape + (self.realizations,), dtype=np.int8)
        if self.do_spotting:
            self.spotting_generation = np.zeros(
                shape + (self.realizations,), dtype=np.uint32
            )
            self.spotting_receiving = np.zeros(
                shape + (self.realizations,), dtype=np.uint32
            )
        else:
            self.spotting_generation = None
            self.spotting_receiving = None
        self.arrival_time = np.zeros(
            shape + (self.realizations,), dtype=np.int32
        )
        self.ros = np.zeros(shape + (self.realizations,), dtype=np.float32)
        self.fireline_int = np.zeros(
            shape + (self.realizations,), dtype=np.float32
        )
        if not self.do_spotting:
            # Copy before mutating: `self.fuels` defaults to (or may be
            # explicitly passed as) the shared `FUEL_SYSTEM_LEGACY`
            # instance. Disabling spotting in place on that shared object
            # would silently disable spotting for every other Propagator
            # in the process that also uses the default/legacy fuels.
            self.fuels = self.fuels.copy()
            self.fuels.disable_spotting()

    def compute_fire_probability(self) -> npt.NDArray[np.floating]:
        """Return mean burn probability across realizations for each cell.

        Returns
        -------
        numpy.ndarray
            2D array with values in [0, 1].
        """
        values = np.mean(self.fire, axis=2).astype(np.float32)
        return values

    def compute_spotting_generation_probability(
        self,
    ) -> npt.NDArray[np.floating]:
        """Return per-cell spotting generation probability."""
        if self.spotting_generation is None:
            return np.zeros(self.veg.shape, dtype=np.float32)
        values = (
            np.sum(self.spotting_generation, axis=2, dtype=np.float64).astype(
                np.float32
            )
            / self.realizations
        )
        return values

    def compute_spotting_receiving_probability(
        self,
    ) -> npt.NDArray[np.floating]:
        """Return per-cell spotting receiving probability."""
        if self.spotting_receiving is None:
            return np.zeros(self.veg.shape, dtype=np.float32)
        values = (
            np.sum(self.spotting_receiving, axis=2, dtype=np.float64).astype(
                np.float32
            )
            / self.realizations
        )
        return values

    def compute_ros_max(self) -> npt.NDArray[np.floating]:
        """Return per-cell maximum Rate of Spread across realizations.

        Returns
        -------
        numpy.ndarray
            2D array with max RoS per cell.
        """
        RoS_max = self._compute_variable_max(self.ros).astype(np.float32)
        return RoS_max

    def compute_arrival_time_min(self) -> npt.NDArray[np.floating]:
        """Return per-cell minimum arrival time across realizations."""
        mask = np.sum(self.fire, axis=2) > 0
        masked = np.where(
            self.fire > 0, self.arrival_time, np.iinfo(np.int32).max
        )
        min_values = np.min(masked, axis=2).astype(np.float32)
        min_values[~mask] = 0
        return min_values

    def compute_arrival_time_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean arrival time across realizations where burned."""
        arrival_time_f = self.arrival_time.astype(np.float32)
        return self._compute_variable_mean(arrival_time_f)

    def compute_ros_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean Rate of Spread, ignoring zeros as no-spread.

        Returns
        -------
        numpy.ndarray
            2D array with mean RoS per cell.
        """
        return self._compute_variable_mean(self.ros)

    def compute_fireline_int_max(self) -> npt.NDArray[np.floating]:
        """Return per-cell maximum fireline intensity across realizations.

        Returns
        -------
        numpy.ndarray
            2D array of max intensity values.
        """
        fl_I_max = self._compute_variable_max(self.fireline_int).astype(
            np.float32
        )
        return fl_I_max

    def compute_fireline_int_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean fireline intensity,
        ignoring zeros as no-spread.

        Returns
        -------
        numpy.ndarray
            2D array of mean intensity values.
        """
        return self._compute_variable_mean(self.fireline_int)

    def compute_flame_length_max(
        self, fli_max: npt.NDArray[np.floating] | None = None
    ) -> npt.NDArray[np.floating]:
        """Return per-cell maximum Byram flame length (m) across
        realizations, derived from the maximum fireline intensity.

        Byram's flame-length relation is monotonically increasing in
        intensity, so max(flame_length) == flame_length(max(intensity)).

        Parameters
        ----------
        fli_max : numpy.ndarray, optional
            An already-computed `compute_fireline_int_max()` result to
            reuse. Only an optimization for callers that need both (see
            `get_output`); recomputed when omitted.

        Returns
        -------
        numpy.ndarray
            2D array of max flame length values (m).
        """
        if fli_max is None:
            fli_max = self.compute_fireline_int_max()
        return _byram_flame_length(fli_max).astype(np.float32)

    def compute_flame_length_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean Byram flame length (m), ignoring zeros
        as no-spread.

        Computed as the true per-realization mean of flame length (not
        the flame length of the mean intensity), consistent with
        compute_fireline_int_mean.

        Returns
        -------
        numpy.ndarray
            2D array of mean flame length values (m).
        """
        # Accumulated one realization at a time. Feeding
        # `_byram_flame_length(self.fireline_int)` to
        # `_compute_variable_mean` would be shorter, but it materializes
        # a flame-length copy of the whole (rows, cols, realizations)
        # intensity grid on every reporting frame -- hundreds of MB at
        # the grid sizes `web.schemas` accepts, and outside the memory
        # its guardrail budgets. Per realization the transient is one 2D
        # grid instead, for the same arithmetic.
        mask = self.fire > 0
        s = np.zeros(self.veg.shape, dtype=np.float64)
        for k in range(self.realizations):
            flame_length = _byram_flame_length(self.fireline_int[:, :, k])
            np.add(s, flame_length, out=s, where=mask[:, :, k])
        c = np.sum(mask, axis=2)

        out = np.full(self.veg.shape, np.nan, dtype=np.float32)
        np.divide(s, c, out=out, where=c > 0)
        return out

    def sample_cells(
        self,
        cells: list[tuple[str, int, int]],
        *,
        fire_probability: npt.NDArray[np.floating] | None = None,
        min_arrival: npt.NDArray[np.floating] | None = None,
        mean_arrival: npt.NDArray[np.floating] | None = None,
    ) -> tuple[CellArrivalSample, ...]:
        """Sample already-computed per-cell arrival time at arbitrary
        (row, col) locations, keyed by an opaque caller-supplied id.

        Generic, grid-index-only utility with no knowledge of lat/lon or
        POI semantics — that mapping is an io/web-layer concern.

        `fire_probability`/`min_arrival`/`mean_arrival` let a caller that
        has already reduced those grids pass them in instead of having
        them recomputed (see `get_output`); each is computed here when
        omitted.

        Returns
        -------
        tuple[CellArrivalSample, ...]
            One sample per requested cell, in the same order. A cell
            outside the grid bounds is reported as unreached rather than
            raising.
        """
        if not cells:
            return ()
        if fire_probability is None:
            fire_probability = self.compute_fire_probability()
        if min_arrival is None:
            min_arrival = self.compute_arrival_time_min()
        if mean_arrival is None:
            mean_arrival = self.compute_arrival_time_mean()
        height, width = fire_probability.shape
        samples = []
        for key, row, col in cells:
            if not (0 <= row < height and 0 <= col < width):
                samples.append(
                    CellArrivalSample(
                        key, row, col, False, float("nan"), float("nan")
                    )
                )
                continue
            reached = bool(fire_probability[row, col] > 0)
            samples.append(
                CellArrivalSample(
                    key,
                    row,
                    col,
                    reached,
                    float(min_arrival[row, col]) if reached else float("nan"),
                    float(mean_arrival[row, col]) if reached else float("nan"),
                )
            )
        return tuple(samples)

    def _compute_variable_mean(
        self, the_var: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Generic mean computation for a 3D variable across realizations,
        ignoring where fire has not spread.

        Parameters
        ----------
        the_var : numpy.ndarray
            3D array with shape (rows, cols, realizations).
            Variable for which to compute the mean.

        Returns
        -------
        numpy.ndarray
            2D array with mean values where fire has spread; 0 otherwise.
        """

        mask = self.fire > 0

        # Accumulate in float64 to reduce precision loss, masking the
        # reduction with `where=` rather than building a masked copy.
        # `np.sum`, not `np.nansum`: nansum internally calls
        # `_replace_nan`, which copies the whole array (plus an isnan
        # mask) for any inexact dtype, so it allocates a full
        # (rows, cols, realizations) transient no matter how the mask is
        # expressed -- measured at ~1x the input array, against ~0x for
        # np.sum. Every reporting frame reduces the ros,
        # fireline-intensity and flame-length grids through here, so that
        # is worth avoiding. Dropping NaN handling is safe because these
        # arrays cannot contain NaN: they are zero-initialised and only
        # ever written with finite values by the numba kernel. Should
        # that change, a NaN now propagates to the output instead of
        # being silently counted as zero -- the louder failure.
        s = np.sum(the_var, axis=2, dtype=np.float64, where=mask)
        c = np.sum(mask, axis=2)

        # mean where count>0; NaN otherwise
        out = np.full(self.veg.shape, np.nan, dtype=np.float32)
        np.divide(s, c, out=out, where=c > 0)
        return out

    def _compute_variable_max(
        self, the_var: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = np.sum(self.fire, axis=2) > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_values = np.nanmax(the_var, axis=2).astype(np.float32)

        max_values[~mask] = 0
        return max_values

    def compute_stats(
        self, values: npt.NDArray[np.floating]
    ) -> PropagatorStats:
        """Compute simple area-based stats and number of active cells.

        Parameters
        ----------
        values : numpy.ndarray
            Fire probability map in [0, 1].

        Returns
        -------
        PropagatorStats
            Dataclass with counters and area summaries.
        """
        n_active = int(np.sum(self._front_sizes > 0))
        cell_area = self.cellsize**2  # m^2, squared cells
        area_mean = float(np.sum(values) * cell_area)
        area_50 = float(np.sum(values >= 0.5) * cell_area)
        area_75 = float(np.sum(values >= 0.75) * cell_area)
        area_90 = float(np.sum(values >= 0.90) * cell_area)

        return PropagatorStats(
            n_active=n_active,
            area_mean=area_mean,
            area_50=area_50,
            area_75=area_75,
            area_90=area_90,
        )

    def set_boundary_conditions(
        self, boundary_condition: BoundaryConditions
    ) -> None:
        """Externally set boundary conditions at desired time in the scheduler.

        Parameters
        ----------
        boundary_condition : BoundaryConditions
            Conditions to apply.
        """
        if int(self.time) > boundary_condition.time:
            raise ValueError(
                "Boundary conditions cannot be applied in the past.\
                Please check the time of the boundary conditions."
            )

        event = SchedulerEvent()

        if boundary_condition.moisture is not None:
            # moisture is given as % we need to transform it to fraction
            moisture = upcast_to_ndarray(
                boundary_condition.moisture, self.dem.shape
            )
            event.moisture = (moisture / 100.0).astype(np.float32, copy=True)
        if boundary_condition.wind_dir is not None:
            # wind direction is given in degrees clockwise, north is 0
            # we need to transform it to radians, counter-clockwise east is 0
            wind_dir_radians = upcast_to_ndarray(
                np.radians(boundary_condition.wind_dir), self.dem.shape
            )

            event.wind_dir = wind_dir_radians.astype(np.float32, copy=True)
        if boundary_condition.wind_speed is not None:
            # wind speed is given in km/h
            wind_speed = upcast_to_ndarray(
                boundary_condition.wind_speed, self.dem.shape
            )
            event.wind_speed = wind_speed.astype(np.float32, copy=True)
        if boundary_condition.additional_moisture is not None:
            # additional moisture is given as % > transform in fraction
            event.additional_moisture = (
                boundary_condition.additional_moisture / 100.0
            ).astype(np.float32, copy=True)
        if boundary_condition.vegetation_changes is not None:
            event.vegetation_changes = boundary_condition.vegetation_changes

        if boundary_condition.ignitions is not None:
            ign_arr = boundary_condition.ignitions
            if isinstance(ign_arr, list):
                points = np.array(ign_arr, dtype=np.int32)

                if len(points.shape) == 2 and points.shape[1] == 2:
                    # 2D points, repeat for all realizations
                    points_repeated = np.repeat(
                        points, self.realizations, axis=0
                    )
                    realizations = np.tile(
                        np.arange(self.realizations), len(points)
                    )
                elif len(points.shape) == 2 and points.shape[1] == 3:
                    # 3D points with realization index
                    points_repeated = points[:, :2]
                    realizations = points[:, 2]
                else:
                    raise ValueError(
                        "Invalid ignitions format in BoundaryConditions: "
                        "If providing a list, each tuple must be either (row, col) or (row, col, realization)."
                    )

            elif isinstance(
                ign_arr, np.ndarray
            ):  # Handle ignition mask as ndarray: extract ignition points
                points = np.argwhere(ign_arr > 0)  # type: ignore

                if len(ign_arr.shape) == 2:
                    points_repeated = np.repeat(
                        points, self.realizations, axis=0
                    )
                    realizations = np.tile(
                        np.arange(self.realizations), len(points)
                    )
                else:
                    points_repeated = points
                    realizations = points[:, 2]
            else:
                raise ValueError(
                    "Invalid ignitions format in BoundaryConditions: "
                    "If providing a numpy array, expected a 2D or 3D ignition mask."
                )

            # The realization index addresses the first axis of the
            # per-realization front arrays. NumPy would either raise an
            # opaque IndexError mid-run or, for a negative index, wrap
            # around and silently ignite the wrong realization, so reject
            # out-of-range values here where the offending input is still
            # in hand.
            realizations = np.asarray(realizations)
            if realizations.size and (
                int(realizations.min()) < 0
                or int(realizations.max()) >= self.realizations
            ):
                raise ValueError(
                    "Invalid ignitions in BoundaryConditions: realization "
                    f"index out of range for {self.realizations} "
                    "realization(s)."
                )

            fireline_intensity = np.zeros_like(
                points_repeated[:, 0], dtype=np.float32
            )

            ros = np.zeros_like(points_repeated[:, 0], dtype=np.float32)
            event.updates = UpdateBatch(
                rows=points_repeated[:, 0],
                cols=points_repeated[:, 1],
                realizations=realizations,
                fireline_intensities=fireline_intensity,
                rates_of_spread=ros,
            )

        self.scheduler.add_event(boundary_condition.time, event)

    def _schedule_ignitions(
        self, time: int, updates: UpdateBatch | None
    ) -> None:
        if updates is None or len(updates.rows) == 0:
            return
        for idx in range(len(updates.rows)):
            realization = int(updates.realizations[idx])
            self._front_push(
                realization=realization,
                time=int(time),
                row=int(updates.rows[idx]),
                col=int(updates.cols[idx]),
                ros=float(updates.rates_of_spread[idx]),
                fli=float(updates.fireline_intensities[idx]),
            )

    def _front_push(
        self,
        realization: int,
        time: int,
        row: int,
        col: int,
        ros: float,
        fli: float,
    ) -> None:
        size = int(self._front_sizes[realization])
        if size >= self._front_capacity:
            self._front_overflow[realization] = 1
            return
        self._front_times[realization, size] = time
        self._front_rows[realization, size] = row
        self._front_cols[realization, size] = col
        self._front_ros[realization, size] = ros
        self._front_fli[realization, size] = fli

        idx = size
        while idx > 0:
            parent = (idx - 1) // 2
            if (
                self._front_times[realization, parent]
                <= self._front_times[realization, idx]
            ):
                break
            for arr in (
                self._front_times,
                self._front_rows,
                self._front_cols,
                self._front_ros,
                self._front_fli,
            ):
                arr[realization, parent], arr[realization, idx] = (
                    arr[realization, idx],
                    arr[realization, parent],
                )
            idx = parent
        self._front_sizes[realization] = size + 1

    def _decay_actions_moisture(
        self, time_delta: int, decay_factor: float = 0.01
    ) -> None:
        """
        Decay the actions moisture over time.

        Args:
            time_delta (int): Elapsed simulation time since last step (seconds).
            decay_factor (float): Per-minute fractional decay in [0, 1].
        """
        if self.actions_moisture is None:
            return
        k = np.clip(decay_factor, 0, 1)
        elapsed_units = max(time_delta / 60.0, 0.0)
        if elapsed_units == 0:
            return
        self.actions_moisture *= (1 - k) ** elapsed_units

    def _get_moisture(self) -> npt.NDArray[np.floating]:
        """
        Get the fuel moisture at the current time step.

        Returns:
            np.ndarray: Base moisture plus action-derived increments,
            clipped to [0, 1].
        """
        if self.actions_moisture is None:
            return self.moisture

        moisture = self.moisture + self.actions_moisture
        moisture = np.clip(moisture, 0.0, 1.0).astype(np.float32, copy=False)

        return moisture

    def _update_boundary_conditions(
        self, time_delta: int, scheduler_event: SchedulerEvent
    ) -> None:
        """Update boundary conditions at the current time step.
        Parameters
        ----------
        time_delta : int
            Elapsed simulation time since last step.
        scheduler_event : SchedulerEvent
            Event containing updated boundary conditions.
        Returns
        -------
        None
        """

        if time_delta > 0:
            self._decay_actions_moisture(time_delta)

        if scheduler_event.moisture is not None:
            self.moisture = scheduler_event.moisture.astype(
                np.float32, copy=False
            )

        if scheduler_event.additional_moisture is not None:
            if self.actions_moisture is None:
                self.actions_moisture = np.zeros_like(self.moisture)
            self.actions_moisture += scheduler_event.additional_moisture
            self.actions_moisture = np.clip(self.actions_moisture, 0.0, 1.0)
            self.actions_moisture = self.actions_moisture.astype(
                np.float32, copy=False
            )

        if scheduler_event.wind_dir is not None:
            self.wind_dir = scheduler_event.wind_dir.astype(
                np.float32, copy=False
            )

        if scheduler_event.wind_speed is not None:
            self.wind_speed = scheduler_event.wind_speed.astype(
                np.float32, copy=False
            )

    def _update_vegetation(self, scheduler_event: SchedulerEvent) -> None:
        if scheduler_event.vegetation_changes is not None:
            # mutate vegetation where needed
            mask = ~np.isnan(scheduler_event.vegetation_changes)
            self.veg[mask] = scheduler_event.vegetation_changes[mask]

    def step(
        self,
        seconds: int | None = None,
        *,
        until: int | None = None,
    ) -> None:
        """Advance the simulation to the next scheduled
        time and update state."""

        if seconds is not None and until is not None:
            raise ValueError("Provide either seconds or until, not both.")

        window = seconds if seconds is not None else until
        if window is None:
            self._step_legacy()
        else:
            if window < 0:
                raise ValueError("seconds/until must be non-negative.")
            self._step_window(window)

    def _step_legacy(self) -> None:
        next_bc_time = self.scheduler.next_time()
        next_prop_time = self._next_front_time()
        if next_bc_time is None and next_prop_time is None:
            return

        times = [
            time for time in (next_bc_time, next_prop_time) if time is not None
        ]
        new_time = min(times)

        if next_bc_time == new_time:
            new_time, scheduler_event = self.scheduler.pop()
            time_delta = new_time - self.time
            self._update_boundary_conditions(time_delta, scheduler_event)
            self._update_vegetation(scheduler_event)
            self._schedule_ignitions(new_time, scheduler_event.updates)
        else:
            if new_time > self.time:
                self._decay_actions_moisture(new_time - self.time)

        self._propagate_until(new_time)
        self.time = new_time

    def _step_window(self, window: int) -> None:
        target_time = self.time + window

        while True:
            next_bc_time = self.scheduler.next_time()
            if next_bc_time is None or next_bc_time > target_time:
                segment_end = target_time
            else:
                segment_end = next_bc_time

            segment_start = self.time
            if segment_end > segment_start:
                self._decay_actions_moisture(segment_end - segment_start)
            self._propagate_until(segment_end)

            if next_bc_time is None or next_bc_time > target_time:
                self.time = segment_end
                break

            bc_time, scheduler_event = self.scheduler.pop()
            self._update_boundary_conditions(0, scheduler_event)
            self._update_vegetation(scheduler_event)
            self._schedule_ignitions(bc_time, scheduler_event.updates)
            self.time = bc_time

    def _propagate_until(self, end_time: int) -> None:
        if int(np.sum(self._front_sizes)) == 0:
            return

        moisture = self._get_moisture()
        out_of_bounds = np.zeros((self.realizations,), dtype=np.int8)
        dummy_spotting = np.zeros((1, 1, 1), dtype=np.uint32)
        spotting_generation = (
            self.spotting_generation
            if self.spotting_generation is not None
            else dummy_spotting
        )
        spotting_receiving = (
            self.spotting_receiving
            if self.spotting_receiving is not None
            else dummy_spotting
        )

        advance_front_until(
            int(end_time),
            int(self._front_capacity),
            self._front_times,
            self._front_rows,
            self._front_cols,
            self._front_ros,
            self._front_fli,
            self._front_sizes,
            self._front_overflow,
            self.cellsize,
            self.veg,
            self.dem,
            self.fire,
            spotting_generation,
            spotting_receiving,
            self.arrival_time,
            self.ros,
            self.fireline_int,
            moisture,
            self.wind_dir,
            self.wind_speed,
            self.fuels,
            self.p_time_fn,
            self.p_moist_fn,
            out_of_bounds,
            self.do_spotting,
        )

        if int(np.sum(self._front_overflow)) > 0:
            raise RuntimeError(
                "Propagation front queue overflowed capacity; "
                "increase front_capacity or front_capacity_factor."
            )

        if (
            self.out_of_bounds_mode == "raise"
            and int(np.sum(out_of_bounds)) > 0
        ):
            raise PropagatorOutOfBoundsError("""Simulation reached the edge of the grid.
                             To ignore this error, set out_of_bounds_mode to 'ignore'.""")

    def _next_front_time(self) -> int | None:
        if int(np.sum(self._front_sizes)) == 0:
            return None
        min_time = None
        for realization in range(self.realizations):
            size = int(self._front_sizes[realization])
            if size == 0:
                continue
            time = int(self._front_times[realization, 0])
            if min_time is None or time < min_time:
                min_time = time
        return min_time

    def get_output(
        self, sample_cells: list[tuple[str, int, int]] | None = None
    ) -> PropagatorOutput:
        """Assemble the current outputs and summary stats into a dataclass.

        Args:
            sample_cells: optional list of (key, row, col) to sample the
                arrival-time grid at, e.g. for reporting when named
                points of interest are reached by the fire front. See
                `sample_cells`.

        Returns:
            PropagatorOutput: Snapshot of fire probability,
                RoS, intensity, stats.
        """
        fire_probability = self.compute_fire_probability()
        spotting_generation_probability = (
            self.compute_spotting_generation_probability()
        )
        spotting_receiving_probability = (
            self.compute_spotting_receiving_probability()
        )
        min_arrival_time = self.compute_arrival_time_min()
        mean_arrival_time = self.compute_arrival_time_mean()
        ros_max = self.compute_ros_max()
        ros_mean = self.compute_ros_mean()
        fireline_intensity_max = self.compute_fireline_int_max()
        fireline_intensity_mean = self.compute_fireline_int_mean()
        # Reuses the intensity max just computed; the mean accumulates
        # per realization (see compute_flame_length_mean) rather than
        # materializing a full 3D flame-length grid.
        flame_length_max = self.compute_flame_length_max(
            fli_max=fireline_intensity_max
        )
        flame_length_mean = self.compute_flame_length_mean()
        stats = self.compute_stats(fire_probability)
        poi_arrival = (
            self.sample_cells(
                sample_cells,
                fire_probability=fire_probability,
                min_arrival=min_arrival_time,
                mean_arrival=mean_arrival_time,
            )
            if sample_cells
            else ()
        )

        return PropagatorOutput(
            time=int(self.time),
            fire_probability=fire_probability,
            spotting_generation_probability=spotting_generation_probability,
            spotting_receiving_probability=spotting_receiving_probability,
            mean_arrival_time=mean_arrival_time,
            min_arrival_time=min_arrival_time,
            ros_mean=ros_mean,
            ros_max=ros_max,
            fli_mean=fireline_intensity_mean,
            fli_max=fireline_intensity_max,
            flame_length_mean=flame_length_mean,
            flame_length_max=flame_length_max,
            stats=stats,
            poi_arrival=poi_arrival,
        )

    def next_time(self) -> int | None:
        """
        Get the next time step.

        Returns:
            int | None: 0 at initialization; None if no more events; otherwise
            the next scheduled simulation time.
        """
        next_bc_time = self.scheduler.next_time()
        next_prop_time = self._next_front_time()

        if next_bc_time is None and next_prop_time is None:
            return None
        if next_bc_time is None:
            return next_prop_time
        if next_prop_time is None:
            return next_bc_time
        return min(next_bc_time, next_prop_time)
