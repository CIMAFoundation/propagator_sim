"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from propagator.core.checkpoint import (
    CHECKPOINT_VERSION,
    PropagatorCheckpoint,
    reanchor_event,
)
from propagator.core.constants import (
    CELLSIZE,
    MOISTURE_MODEL_DEFAULT,
    REALIZATIONS,
    ROS_DEFAULT,
)
from propagator.core.models import (
    BoundaryConditions,
    PropagatorOutput,
    PropagatorStats,
    UpdateBatch,
)
from propagator.core.numba import (
    FLAG_FIRE,
    FLAG_SPOT_GEN,
    FLAG_SPOT_RECV,
    FRONT_RESERVE,
    FUEL_SYSTEM_LEGACY,
    TILE_MASK,
    TILE_RESERVE,
    TILE_SHIFT,
    TILE_SIZE,
    FuelSystem,
    advance_front_until,
    build_fuel_index_grid,
    fold_state_tiles,
    get_p_moisture_fn,
    get_p_time_fn,
    materialize_tiles,
)
from propagator.core.scheduler import Scheduler, SchedulerEvent

from .utils import upcast_to_ndarray


class PropagatorOutOfBoundsError(Exception):
    """Custom error for out-of-bounds updates in the Propagator."""

    pass


# Initial per-realization capacity of the front event heap. The heap only
# holds the active front (a small fraction of the grid), so it starts small
# and doubles whenever the kernel reports it ran out of headroom.
FRONT_INITIAL_CAPACITY = 4096

# Initial per-realization capacity (in tiles) of the block-sparse state
# pools; doubles on demand like the front heap.
TILE_INITIAL_CAPACITY = 32


@dataclass(frozen=True)
class StateFold:
    """Per-cell aggregates of the tiled state across realizations."""

    count: npt.NDArray[np.int32]
    spot_gen_count: npt.NDArray[np.int32]
    spot_recv_count: npt.NDArray[np.int32]
    arrival_min: npt.NDArray[np.int32]
    arrival_sum: npt.NDArray[np.float64]
    ros_sum: npt.NDArray[np.float64]
    ros_max: npt.NDArray[np.float32]
    fli_sum: npt.NDArray[np.float64]
    fli_max: npt.NDArray[np.float32]


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

    # world (row, col) of local cell (0, 0): anchors the grid in an absolute
    # cell coordinate system so checkpoints can be restored onto larger,
    # shifted grids (see from_checkpoint)
    origin: tuple[int, int] = (0, 0)

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
    _fuel_idx: npt.NDArray[np.int64] = field(init=False)

    # block-sparse per-realization state: burn/spotting flags, arrival time,
    # rate of spread and fireline intensity live in on-demand 32x32 tiles
    # (see core.numba.tiles); use get_fire()/get_arrival_time()/get_ros()/
    # get_fireline_int() and the spotting getters for dense views
    _tile_idx: npt.NDArray[np.int32] = field(init=False)
    _tile_counts: npt.NDArray[np.int32] = field(init=False)
    _tile_capacity: int = field(init=False, default=0)
    _tile_flags: npt.NDArray[np.uint8] = field(init=False)
    _tile_arrival: npt.NDArray[np.int32] = field(init=False)
    _tile_ros: npt.NDArray[np.float32] = field(init=False)
    _tile_fli: npt.NDArray[np.float32] = field(init=False)

    # simulation state
    time: int = field(init=False, default=0)
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
        self._front_capacity = FRONT_INITIAL_CAPACITY
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
        tiles_h = -(-shape[0] // TILE_SIZE)
        tiles_w = -(-shape[1] // TILE_SIZE)
        self._tile_idx = np.full(
            (self.realizations, tiles_h, tiles_w), -1, dtype=np.int32
        )
        self._tile_counts = np.zeros((self.realizations,), dtype=np.int32)
        self._tile_capacity = min(TILE_INITIAL_CAPACITY, tiles_h * tiles_w)
        self._tile_flags = np.zeros(
            (self.realizations, self._tile_capacity, TILE_SIZE, TILE_SIZE),
            dtype=np.uint8,
        )
        self._tile_arrival = np.zeros(
            (self.realizations, self._tile_capacity, TILE_SIZE, TILE_SIZE),
            dtype=np.int32,
        )
        self._tile_ros = np.zeros(
            (self.realizations, self._tile_capacity, TILE_SIZE, TILE_SIZE),
            dtype=np.float32,
        )
        self._tile_fli = np.zeros(
            (self.realizations, self._tile_capacity, TILE_SIZE, TILE_SIZE),
            dtype=np.float32,
        )
        if not self.do_spotting:
            self.fuels.disable_spotting()
        self._fuel_idx = build_fuel_index_grid(self.fuels, self.veg)

    def compute_fire_probability(self) -> npt.NDArray[np.floating]:
        """Return mean burn probability across realizations for each cell.

        Returns
        -------
        numpy.ndarray
            2D array with values in [0, 1].
        """
        return self._probability(self._fold_state().count)

    def compute_spotting_generation_probability(
        self,
    ) -> npt.NDArray[np.floating]:
        """Return per-cell spotting generation probability."""
        return self._probability(self._fold_state().spot_gen_count)

    def compute_spotting_receiving_probability(
        self,
    ) -> npt.NDArray[np.floating]:
        """Return per-cell spotting receiving probability."""
        return self._probability(self._fold_state().spot_recv_count)

    def _probability(
        self, count: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.floating]:
        return (count / self.realizations).astype(np.float32)

    def compute_ros_max(self) -> npt.NDArray[np.floating]:
        """Return per-cell maximum Rate of Spread across realizations.

        Returns
        -------
        numpy.ndarray
            2D array with max RoS per cell.
        """
        return self._fold_state().ros_max

    def compute_arrival_time_min(self) -> npt.NDArray[np.floating]:
        """Return per-cell minimum arrival time across realizations."""
        return self._arrival_time_min(self._fold_state())

    def compute_arrival_time_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean arrival time across realizations where burned."""
        fold = self._fold_state()
        return self._masked_mean(fold.arrival_sum, fold.count)

    def compute_ros_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean Rate of Spread, ignoring zeros as no-spread.

        Returns
        -------
        numpy.ndarray
            2D array with mean RoS per cell.
        """
        fold = self._fold_state()
        return self._masked_mean(fold.ros_sum, fold.count)

    def compute_fireline_int_max(self) -> npt.NDArray[np.floating]:
        """Return per-cell maximum fireline intensity across realizations.

        Returns
        -------
        numpy.ndarray
            2D array of max intensity values.
        """
        return self._fold_state().fli_max

    def compute_fireline_int_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean fireline intensity,
        ignoring zeros as no-spread.

        Returns
        -------
        numpy.ndarray
            2D array of mean intensity values.
        """
        fold = self._fold_state()
        return self._masked_mean(fold.fli_sum, fold.count)

    def _fold_state(self) -> StateFold:
        """Aggregate the tiled state into per-cell 2D maps in one pass."""
        shape = self.veg.shape
        fold = StateFold(
            count=np.zeros(shape, dtype=np.int32),
            spot_gen_count=np.zeros(shape, dtype=np.int32),
            spot_recv_count=np.zeros(shape, dtype=np.int32),
            arrival_min=np.full(shape, np.iinfo(np.int32).max, dtype=np.int32),
            arrival_sum=np.zeros(shape, dtype=np.float64),
            ros_sum=np.zeros(shape, dtype=np.float64),
            ros_max=np.zeros(shape, dtype=np.float32),
            fli_sum=np.zeros(shape, dtype=np.float64),
            fli_max=np.zeros(shape, dtype=np.float32),
        )
        fold_state_tiles(
            self._tile_idx,
            self._tile_flags,
            self._tile_arrival,
            self._tile_ros,
            self._tile_fli,
            fold.count,
            fold.spot_gen_count,
            fold.spot_recv_count,
            fold.arrival_min,
            fold.arrival_sum,
            fold.ros_sum,
            fold.ros_max,
            fold.fli_sum,
            fold.fli_max,
        )
        return fold

    @staticmethod
    def _arrival_time_min(fold: StateFold) -> npt.NDArray[np.floating]:
        values = fold.arrival_min.astype(np.float32)
        values[fold.count == 0] = 0
        return values

    def _masked_mean(
        self, the_sum: npt.NDArray[np.float64], count: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.floating]:
        """Mean where fire has spread; NaN otherwise."""
        out = np.full(self.veg.shape, np.nan, dtype=np.float32)
        np.divide(the_sum, count, out=out, where=count > 0)
        return out

    def _materialize_flags(self) -> npt.NDArray[np.uint8]:
        out = np.zeros((self.realizations,) + self.veg.shape, dtype=np.uint8)
        materialize_tiles(self._tile_idx, self._tile_flags, out)
        return out

    def get_fire(self) -> npt.NDArray[np.uint8]:
        """Materialize the dense (realizations, rows, cols) burn state (0/1).

        This allocates the full dense array; intended for analysis and
        tests, not for the hot path.
        """
        return ((self._materialize_flags() & FLAG_FIRE) != 0).astype(np.uint8)

    def get_spotting_generation(self) -> npt.NDArray[np.uint8]:
        """Materialize the dense per-realization spotting generation mask."""
        return ((self._materialize_flags() & FLAG_SPOT_GEN) != 0).astype(
            np.uint8
        )

    def get_spotting_receiving(self) -> npt.NDArray[np.uint8]:
        """Materialize the dense per-realization spotting receiving mask."""
        return ((self._materialize_flags() & FLAG_SPOT_RECV) != 0).astype(
            np.uint8
        )

    def get_arrival_time(self) -> npt.NDArray[np.int32]:
        """Materialize the dense (realizations, rows, cols) arrival times.

        Cells never reached by fire are 0. This allocates the full dense
        array; intended for analysis and tests, not for the hot path.
        """
        out = np.zeros((self.realizations,) + self.veg.shape, dtype=np.int32)
        materialize_tiles(self._tile_idx, self._tile_arrival, out)
        return out

    def get_ros(self) -> npt.NDArray[np.float32]:
        """Materialize the dense (realizations, rows, cols) Rate of Spread.

        Cells never reached by fire are 0. This allocates the full dense
        array; intended for analysis and tests, not for the hot path.
        """
        out = np.zeros((self.realizations,) + self.veg.shape, dtype=np.float32)
        materialize_tiles(self._tile_idx, self._tile_ros, out)
        return out

    def get_fireline_int(self) -> npt.NDArray[np.float32]:
        """Materialize the dense (realizations, rows, cols) fireline intensity.

        Cells never reached by fire are 0. This allocates the full dense
        array; intended for analysis and tests, not for the hot path.
        """
        out = np.zeros((self.realizations,) + self.veg.shape, dtype=np.float32)
        materialize_tiles(self._tile_idx, self._tile_fli, out)
        return out

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

    def _apply_updates(
        self,
        updates: UpdateBatch,
        new_time: int | None = None,
    ) -> None:
        """Apply a batch of burning updates to the state.
        Parameters
        ----------
        updates : UpdateBatch
            Batch of updates to apply at the current time step.
        new_time : int | None
            Optional simulation time to set.
        Returns
        -------
        None
        """

        if new_time is not None:
            self.time = new_time
        rows = updates.rows
        cols = updates.cols
        realizations = updates.realizations
        ros = updates.rates_of_spread
        fireline_intensity = updates.fireline_intensities

        for index in range(len(rows)):
            realization = int(realizations[index])
            tile, local_row, local_col = self._state_tile_slot(
                realization, int(rows[index]), int(cols[index])
            )
            self._tile_flags[realization, tile, local_row, local_col] |= (
                FLAG_FIRE
            )
            self._tile_arrival[realization, tile, local_row, local_col] = int(
                self.time
            )
            self._tile_ros[realization, tile, local_row, local_col] = ros[
                index
            ]
            self._tile_fli[realization, tile, local_row, local_col] = (
                fireline_intensity[index]
            )

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
            self._grow_front()
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

    def _grow_front(self) -> None:
        """Double the capacity of the front event heap, preserving content."""
        new_capacity = self._front_capacity * 2
        for name in (
            "_front_times",
            "_front_rows",
            "_front_cols",
            "_front_ros",
            "_front_fli",
        ):
            old = getattr(self, name)
            new = np.zeros((self.realizations, new_capacity), dtype=old.dtype)
            new[:, : self._front_capacity] = old
            setattr(self, name, new)
        self._front_capacity = new_capacity

    def _grow_tiles(self) -> None:
        """Double the capacity of the state tile pools, preserving content."""
        tiles_h, tiles_w = self._tile_idx.shape[1:]
        new_capacity = min(self._tile_capacity * 2, tiles_h * tiles_w)
        for name in ("_tile_flags", "_tile_arrival", "_tile_ros", "_tile_fli"):
            old = getattr(self, name)
            new = np.zeros(
                (self.realizations, new_capacity, TILE_SIZE, TILE_SIZE),
                dtype=old.dtype,
            )
            new[:, : self._tile_capacity] = old
            setattr(self, name, new)
        self._tile_capacity = new_capacity

    def _state_tile_slot(
        self, realization: int, row: int, col: int
    ) -> tuple[int, int, int]:
        """Return (tile, local_row, local_col) for a cell, allocating the
        tile if needed."""
        tile_row = row >> TILE_SHIFT
        tile_col = col >> TILE_SHIFT
        tile = int(self._tile_idx[realization, tile_row, tile_col])
        if tile < 0:
            if int(self._tile_counts[realization]) >= self._tile_capacity:
                self._grow_tiles()
            tile = int(self._tile_counts[realization])
            self._tile_counts[realization] = tile + 1
            self._tile_idx[realization, tile_row, tile_col] = tile
        return tile, row & TILE_MASK, col & TILE_MASK

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

    def _get_simulation_bbox(self) -> tuple[int, int, int, int]:
        """Get the bounding box of the simulation area.

        Returns:
            tuple[int, int, int, int]: (row_min, col_min, row_max, col_max)
        """
        n_rows, n_cols = self.veg.shape
        return (0, 0, n_rows - 1, n_cols - 1)

    def _check_out_of_bounds(self, updates: UpdateBatch) -> None:
        # check that all updates are within bounds
        bbox = updates.get_bbox()
        if bbox is None:
            return

        update_r0, update_c0, update_r1, update_c1 = bbox
        sim_bbox = self._get_simulation_bbox()
        sim_r0, sim_c0, sim_r1, sim_c1 = sim_bbox
        n_rows, n_cols = self.veg.shape
        if (
            update_r0 <= sim_r0
            or update_c0 <= sim_c0
            or update_r1 >= n_rows - 1
            or update_c1 >= n_cols - 1
        ):
            raise PropagatorOutOfBoundsError("""Simulation reached the edge of the grid.
                             To ignore this error, set out_of_bounds_mode to 'ignore'.""")

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
            self._fuel_idx = build_fuel_index_grid(self.fuels, self.veg)

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
        # in "raise" mode the kernel suspends BEFORE igniting a boundary
        # cell (heap intact), so the caller can checkpoint, expand the
        # domain via from_checkpoint and resume without losing spread
        halt_on_boundary = self.out_of_bounds_mode == "raise"

        while True:
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
                self._fuel_idx,
                self.dem,
                self._tile_idx,
                self._tile_counts,
                int(self._tile_capacity),
                self._tile_flags,
                self._tile_arrival,
                self._tile_ros,
                self._tile_fli,
                moisture,
                self.wind_dir,
                self.wind_speed,
                self.fuels,
                self.p_time_fn,
                self.p_moist_fn,
                out_of_bounds,
                self.do_spotting,
                halt_on_boundary,
            )

            if int(np.sum(self._front_overflow)) == 0:
                break
            # the kernel suspends a realization (heap intact) when it cannot
            # guarantee headroom for the next pop: grow whatever ran out
            # and resume
            flagged = self._front_overflow != 0
            grew = False
            if np.any(
                self._front_sizes[flagged] + FRONT_RESERVE
                > self._front_capacity
            ):
                self._grow_front()
                grew = True
            tiles_h, tiles_w = self._tile_idx.shape[1:]
            tile_demand = np.minimum(
                self._tile_counts[flagged] + TILE_RESERVE, tiles_h * tiles_w
            )
            if np.any(tile_demand > self._tile_capacity):
                self._grow_tiles()
                grew = True
            if not grew:
                raise RuntimeError(
                    "Propagation kernel suspended without a growable "
                    "resource; this is a bug."
                )
            self._front_overflow[:] = 0

        if (
            self.out_of_bounds_mode == "raise"
            and int(np.sum(out_of_bounds)) > 0
        ):
            raise PropagatorOutOfBoundsError(
                "Simulation reached the edge of the grid. The state is "
                "intact: checkpoint() and resume on a larger grid with "
                "Propagator.from_checkpoint(), or set out_of_bounds_mode "
                "to 'ignore' to let the fire stop at the boundary."
            )

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

    def get_output(self) -> PropagatorOutput:
        """Assemble the current outputs and summary stats into a dataclass.

        Returns:
            PropagatorOutput: Snapshot of fire probability,
                RoS, intensity, stats.
        """
        # single fold pass shared by every derived map: get_output is called
        # frequently in production, so avoid re-folding per quantity
        fold = self._fold_state()
        fire_probability = self._probability(fold.count)
        spotting_generation_probability = self._probability(
            fold.spot_gen_count
        )
        spotting_receiving_probability = self._probability(
            fold.spot_recv_count
        )
        min_arrival_time = self._arrival_time_min(fold)
        mean_arrival_time = self._masked_mean(fold.arrival_sum, fold.count)
        ros_max = fold.ros_max
        ros_mean = self._masked_mean(fold.ros_sum, fold.count)
        fireline_intensity_max = fold.fli_max
        fireline_intensity_mean = self._masked_mean(fold.fli_sum, fold.count)
        stats = self.compute_stats(fire_probability)

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
            stats=stats,
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

    # --- Checkpointing and domain growth ---------------------------------

    def world_bounds(self) -> tuple[int, int, int, int]:
        """Inclusive world-cell bounds (row0, col0, row1, col1) of the grid."""
        rows, cols = self.veg.shape
        return (
            self.origin[0],
            self.origin[1],
            self.origin[0] + rows - 1,
            self.origin[1] + cols - 1,
        )

    def checkpoint(self) -> PropagatorCheckpoint:
        """Snapshot the full dynamic state as an immutable checkpoint.

        The checkpoint is independent of this instance (deep copies) and
        can be used for in-place rollback (`restore`), persisted to disk
        (`PropagatorCheckpoint.save`/`load`), or resumed on a larger grid
        (`Propagator.from_checkpoint`).

        Note: the stochastic RNG state is not captured, so runs resumed
        from a checkpoint are statistically equivalent but not bitwise
        reproductions of the original continuation.
        """

        def _optional(name: str) -> npt.NDArray | None:
            value = getattr(self, name, None)
            return None if value is None else np.array(value, copy=True)

        front_size = int(self._front_sizes.max())
        tile_count = int(self._tile_counts.max())
        return PropagatorCheckpoint(
            version=CHECKPOINT_VERSION,
            time=int(self.time),
            origin=(int(self.origin[0]), int(self.origin[1])),
            cellsize=float(self.cellsize),
            realizations=int(self.realizations),
            do_spotting=bool(self.do_spotting),
            veg=self.veg.copy(),
            dem=self.dem.copy(),
            moisture=_optional("moisture"),
            wind_dir=_optional("wind_dir"),
            wind_speed=_optional("wind_speed"),
            actions_moisture=_optional("actions_moisture"),
            front_times=self._front_times[:, :front_size].copy(),
            front_rows=self._front_rows[:, :front_size].copy(),
            front_cols=self._front_cols[:, :front_size].copy(),
            front_ros=self._front_ros[:, :front_size].copy(),
            front_fli=self._front_fli[:, :front_size].copy(),
            front_sizes=self._front_sizes.copy(),
            tile_idx=self._tile_idx.copy(),
            tile_counts=self._tile_counts.copy(),
            tile_flags=self._tile_flags[:, :tile_count].copy(),
            tile_arrival=self._tile_arrival[:, :tile_count].copy(),
            tile_ros=self._tile_ros[:, :tile_count].copy(),
            tile_fli=self._tile_fli[:, :tile_count].copy(),
            scheduler_events=self.scheduler.snapshot(),
        )

    def restore(self, checkpoint: PropagatorCheckpoint) -> None:
        """Roll this propagator back to a checkpoint taken on the same grid.

        Restores simulation time, front heaps, tiled state, weather
        fields, vegetation (including past `vegetation_changes`) and the
        pending boundary-condition queue. The checkpoint stays valid and
        can be restored again.
        """
        if checkpoint.shape != self.veg.shape:
            raise ValueError(
                f"Checkpoint grid {checkpoint.shape} does not match this "
                f"propagator's grid {self.veg.shape}; use "
                "Propagator.from_checkpoint() to change the domain."
            )
        if tuple(checkpoint.origin) != tuple(self.origin):
            raise ValueError(
                f"Checkpoint origin {checkpoint.origin} does not match "
                f"this propagator's origin {tuple(self.origin)}."
            )
        self.veg = checkpoint.veg.copy()
        self._fuel_idx = build_fuel_index_grid(self.fuels, self.veg)
        self._load_state(checkpoint, 0, 0)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: PropagatorCheckpoint,
        *,
        veg: npt.NDArray[np.integer] | None = None,
        dem: npt.NDArray[np.floating] | None = None,
        origin: tuple[int, int] | None = None,
        fuels: FuelSystem | None = None,
        p_time_fn: Any = None,
        p_moist_fn: Any = None,
        out_of_bounds_mode: Literal["ignore", "raise"] = "raise",
    ) -> "Propagator":
        """Build a propagator resuming from a checkpoint.

        Without `veg`/`dem` the checkpointed domain is reconstructed
        as-is. Passing `veg`, `dem` and a world `origin` resumes on a new
        (typically larger) grid: it must fully contain the checkpointed
        grid, and any north/west growth (`checkpoint.origin - origin`)
        must be a multiple of TILE_SIZE cells so the block-sparse tiles
        transfer without being rewritten. Vegetation in the overlap is
        taken from the checkpoint, preserving past `vegetation_changes`.

        `fuels`, `p_time_fn` and `p_moist_fn` are not serialized; pass
        the same objects used originally (defaults apply when omitted).
        """
        if veg is None:
            if dem is not None or origin is not None:
                raise ValueError(
                    "veg, dem and origin must be provided together"
                )
            veg = checkpoint.veg.copy()
            dem = checkpoint.dem.copy()
            origin = checkpoint.origin
            row_shift = 0
            col_shift = 0
        else:
            if dem is None:
                raise ValueError("dem is required when providing veg")
            if veg.shape != dem.shape:
                raise ValueError("veg and dem must have the same shape")
            if origin is None:
                origin = checkpoint.origin
            row_shift = checkpoint.origin[0] - origin[0]
            col_shift = checkpoint.origin[1] - origin[1]
            old_rows, old_cols = checkpoint.shape
            if row_shift < 0 or col_shift < 0:
                raise ValueError(
                    "New origin must not be south/east of the checkpoint "
                    "origin: the new grid must contain the old one."
                )
            if row_shift % TILE_SIZE or col_shift % TILE_SIZE:
                raise ValueError(
                    "North/west growth must be a multiple of "
                    f"TILE_SIZE ({TILE_SIZE}) cells; got shift "
                    f"({row_shift}, {col_shift})."
                )
            if (
                row_shift + old_rows > veg.shape[0]
                or col_shift + old_cols > veg.shape[1]
            ):
                raise ValueError(
                    "New grid does not fully contain the checkpoint grid."
                )
            veg = np.array(veg, copy=True)
            veg[
                row_shift : row_shift + old_rows,
                col_shift : col_shift + old_cols,
            ] = checkpoint.veg

        extra: dict[str, Any] = {}
        if fuels is not None:
            extra["fuels"] = fuels
        if p_time_fn is not None:
            extra["p_time_fn"] = p_time_fn
        if p_moist_fn is not None:
            extra["p_moist_fn"] = p_moist_fn

        propagator = cls(
            veg=veg,
            dem=dem,
            cellsize=checkpoint.cellsize,
            do_spotting=checkpoint.do_spotting,
            realizations=checkpoint.realizations,
            origin=(int(origin[0]), int(origin[1])),
            out_of_bounds_mode=out_of_bounds_mode,
            **extra,
        )
        propagator._load_state(checkpoint, row_shift, col_shift)
        return propagator

    def _load_state(
        self,
        checkpoint: PropagatorCheckpoint,
        row_shift: int,
        col_shift: int,
    ) -> None:
        """Replace all dynamic state with the checkpoint's, re-anchored by
        (row_shift, col_shift) into this propagator's grid."""
        if checkpoint.version > CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {checkpoint.version} is newer than "
                f"supported version {CHECKPOINT_VERSION}"
            )
        if checkpoint.realizations != self.realizations:
            raise ValueError("Checkpoint realizations mismatch")
        if checkpoint.do_spotting != self.do_spotting:
            raise ValueError("Checkpoint do_spotting mismatch")
        if checkpoint.cellsize != self.cellsize:
            raise ValueError("Checkpoint cellsize mismatch")

        old_shape = checkpoint.shape
        new_shape = self.veg.shape
        grew = old_shape != new_shape or row_shift != 0 or col_shift != 0
        self.time = int(checkpoint.time)

        # front event heaps (stored trimmed to the largest heap)
        stored_events = checkpoint.front_times.shape[1]
        capacity = FRONT_INITIAL_CAPACITY
        while capacity < stored_events + FRONT_RESERVE + 1:
            capacity *= 2
        self._front_capacity = capacity
        for name, source, shift in (
            ("_front_times", checkpoint.front_times, 0),
            ("_front_rows", checkpoint.front_rows, row_shift),
            ("_front_cols", checkpoint.front_cols, col_shift),
            ("_front_ros", checkpoint.front_ros, 0),
            ("_front_fli", checkpoint.front_fli, 0),
        ):
            array = np.zeros((self.realizations, capacity), source.dtype)
            array[:, :stored_events] = source + shift if shift else source
            setattr(self, name, array)
        self._front_sizes = checkpoint.front_sizes.astype(np.int32).copy()
        self._front_overflow = np.zeros((self.realizations,), dtype=np.int8)

        # block-sparse tile state: the tile-index grid is re-anchored by
        # whole tiles; pool contents transfer untouched
        tiles_h = -(-new_shape[0] // TILE_SIZE)
        tiles_w = -(-new_shape[1] // TILE_SIZE)
        tile_row0 = row_shift >> TILE_SHIFT
        tile_col0 = col_shift >> TILE_SHIFT
        old_tiles_h, old_tiles_w = checkpoint.tile_idx.shape[1:]
        self._tile_idx = np.full(
            (self.realizations, tiles_h, tiles_w), -1, dtype=np.int32
        )
        self._tile_idx[
            :,
            tile_row0 : tile_row0 + old_tiles_h,
            tile_col0 : tile_col0 + old_tiles_w,
        ] = checkpoint.tile_idx
        self._tile_counts = checkpoint.tile_counts.astype(np.int32).copy()

        stored_tiles = checkpoint.tile_flags.shape[1]
        tile_capacity = max(TILE_INITIAL_CAPACITY, stored_tiles + TILE_RESERVE)
        tile_capacity = max(
            min(tile_capacity, tiles_h * tiles_w), stored_tiles
        )
        self._tile_capacity = tile_capacity
        for name, source in (
            ("_tile_flags", checkpoint.tile_flags),
            ("_tile_arrival", checkpoint.tile_arrival),
            ("_tile_ros", checkpoint.tile_ros),
            ("_tile_fli", checkpoint.tile_fli),
        ):
            array = np.zeros(
                (self.realizations, tile_capacity, TILE_SIZE, TILE_SIZE),
                source.dtype,
            )
            array[:, :stored_tiles] = source
            setattr(self, name, array)

        # environmental fields: weather grows by edge replication, action
        # moisture with zeros (no suppression outside the old domain)
        pad = (
            (row_shift, new_shape[0] - old_shape[0] - row_shift),
            (col_shift, new_shape[1] - old_shape[1] - col_shift),
        )
        for name in ("moisture", "wind_dir", "wind_speed"):
            value = getattr(checkpoint, name)
            if value is None:
                if hasattr(self, name):
                    delattr(self, name)
                continue
            setattr(
                self,
                name,
                np.pad(value, pad, mode="edge") if grew else value.copy(),
            )
        if checkpoint.actions_moisture is None:
            self.actions_moisture = None
        elif grew:
            self.actions_moisture = np.pad(
                checkpoint.actions_moisture,
                pad,
                mode="constant",
                constant_values=0,
            )
        else:
            self.actions_moisture = checkpoint.actions_moisture.copy()

        # pending boundary-condition queue
        self.scheduler = Scheduler(realizations=self.realizations)
        for event_time, event in checkpoint.scheduler_events:
            event = copy.deepcopy(event)
            reanchor_event(event, row_shift, col_shift, old_shape, new_shape)
            self.scheduler.add_event(int(event_time), event)
