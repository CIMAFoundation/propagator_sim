"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from propagator.core.constants import CELLSIZE, REALIZATIONS
from propagator.core.models import (
    BoundaryConditions,
    PropagatorOutput,
    PropagatorStats,
    UpdateBatch,
    UpdateBatchWithTime,
)
from propagator.core.numba import (
    FUEL_SYSTEM_LEGACY,
    FuelSystem,
    get_p_moisture_fn,
    get_p_time_fn,
    next_updates_fn,
)
from propagator.core.scheduler import Scheduler, SchedulerEvent


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
    """

    # domain parameters for the simulation

    # input
    veg: npt.NDArray[np.integer]
    dem: npt.NDArray[np.floating]

    # set fuels
    fuels: FuelSystem = field(default_factory=lambda: FUEL_SYSTEM_LEGACY)

    # simulation settings
    cellsize: float = field(default=CELLSIZE)
    do_spotting: bool = False
    realizations: int = REALIZATIONS

    # selected simulation functions
    p_time_fn: Any = field(default=get_p_time_fn("wang"))
    p_moist_fn: Any = field(default=get_p_moisture_fn("trucchia"))

    # scheduler object
    scheduler: Scheduler = field(init=False)

    # simulation state
    time: int = field(init=False, default=0)
    fire: npt.NDArray[np.int8] = field(init=False)
    ros: npt.NDArray[np.float32] = field(init=False)
    fireline_int: npt.NDArray[np.float32] = field(init=False)
    moisture: npt.NDArray[np.floating] = field(init=False)
    wind_dir: npt.NDArray[np.floating] = field(init=False)
    wind_speed: npt.NDArray[np.floating] = field(init=False)
    actions_moisture: npt.NDArray[np.floating] | None = field(
        default=None, init=False
    )  # additional moisture due to fighting actions
    # (ideally it should decay over time)

    def __post_init__(self):
        """Allocate internal state arrays based
        on the vegetation grid shape."""
        shape = self.veg.shape
        self.scheduler = Scheduler(realizations=self.realizations)
        self.fire = np.zeros(shape + (self.realizations,), dtype=np.int8)
        self.ros = np.zeros(shape + (self.realizations,), dtype=np.float32)
        self.fireline_int = np.zeros(
            shape + (self.realizations,), dtype=np.float32
        )
        if not self.do_spotting:
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

    def compute_ros_max(self) -> npt.NDArray[np.floating]:
        """Return per-cell maximum Rate of Spread across realizations.

        Returns
        -------
        numpy.ndarray
            2D array with max RoS per cell.
        """
        RoS_max = np.nanmax(self.ros, axis=2).astype(np.float32)
        return RoS_max

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
        fl_I_max = np.nanmax(self.fireline_int, axis=2).astype(np.float32)
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
            2D array with mean values where fire has spread; NaN otherwise.
        """

        mask = self.fire > 0

        # accumulate in float64 to reduce precision loss
        s = np.nansum(np.where(mask, the_var, 0.0), axis=2, dtype=np.float64)
        c = np.sum(mask, axis=2)

        # mean where count>0; NaN otherwise
        out = np.full(self.veg.shape, np.nan, dtype=np.float32)
        np.divide(s, c, out=out, where=c > 0)
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
        n_active = len(self.scheduler.active().tolist())
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
        """Externally apply boundary conditions at desired time.

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
            event.moisture = boundary_condition.moisture / 100.0
        if boundary_condition.wind_dir is not None:
            # wind direction is given in degrees clockwise, north is 0
            # we need to transform it to radians, counter-clockwise east is 0
            wind_dir_radians = np.radians(90 - boundary_condition.wind_dir)
            wind_dir_norm = (wind_dir_radians + np.pi) % (2 * np.pi) - np.pi
            event.wind_dir = wind_dir_norm
        if boundary_condition.wind_speed is not None:
            # wind speed is given in km/h
            event.wind_speed = boundary_condition.wind_speed
        if boundary_condition.additional_moisture is not None:
            # additional moisture is given as % > transform in fraction
            event.additional_moisture = (
                boundary_condition.additional_moisture / 100.0
            )
        if boundary_condition.vegetation_changes is not None:
            event.vegetation_changes = boundary_condition.vegetation_changes

        if boundary_condition.ignition_mask is not None:
            ign_arr = boundary_condition.ignition_mask
            points = np.argwhere(ign_arr > 0)

            points_repeated = np.repeat(points, self.realizations, axis=0)
            realizations = np.tile(np.arange(self.realizations), len(points))

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

    def apply_updates(
        self,
        updates: UpdateBatch,
    ) -> None:
        """Apply a batch of burning updates and schedule new ones.
        Parameters
        ----------
        updates : list[numpy.ndarray] | numpy.ndarray
            Coordinates to activate as burning, each as
            [row, col, realization].
        Returns
        -------
        list[tuple[float, numpy.ndarray]]
            Pairs of (time, array[n, 3]) for future updates to schedule.
        """
        moisture = self.get_moisture()

        must_be_updated = (
            self.fire[updates.rows, updates.cols, updates.realizations] == 0
        )
        rows = updates.rows[must_be_updated]
        cols = updates.cols[must_be_updated]
        realizations = updates.realizations[must_be_updated]
        ros = updates.rates_of_spread[must_be_updated]
        fireline_intensity = updates.fireline_intensities[must_be_updated]

        self.fire[rows, cols, realizations] = 1
        self.ros[rows, cols, realizations] = ros
        self.fireline_int[rows, cols, realizations] = fireline_intensity

        new_updates_tuple = next_updates_fn(
            rows,
            cols,
            realizations,
            self.cellsize,
            self.time,
            self.veg,
            self.dem,
            self.fire,
            moisture,
            self.wind_dir,
            self.wind_speed,
            self.fuels,
            self.p_time_fn,
            self.p_moist_fn,
        )

        next_updates = UpdateBatchWithTime.from_tuple(new_updates_tuple)
        self.scheduler.push_updates(next_updates)

    def decay_actions_moisture(
        self, time_delta: int, decay_factor: float = 0.01
    ) -> None:
        """
        Decay the actions moisture over time.

        Args:
            time_delta (int): Elapsed simulation time since last step.
            decay_factor (float): Per-unit-time fractional decay in [0, 1].
        """
        if self.actions_moisture is None:
            return
        k = np.clip(decay_factor, 0, 1)
        self.actions_moisture *= (1 - k) ** max(time_delta, 0)

    def get_moisture(self) -> npt.NDArray[np.floating]:
        """
        Get the fuel moisture at the current time step.

        Returns:
            np.ndarray: Base moisture plus action-derived increments,
            clipped to [0, 1].
        """
        if self.actions_moisture is None:
            return self.moisture

        moisture = self.moisture + self.actions_moisture
        moisture = np.clip(moisture, 0.0, 1.0)

        return moisture

    def step(
        self,
    ) -> None:
        """Advance the simulation to the next scheduled
        time and update state."""
        time, scheduler_event = self.scheduler.pop()

        time_delta = time - self.time
        self.time = time
        self.decay_actions_moisture(time_delta)

        if scheduler_event.moisture is not None:
            self.moisture = scheduler_event.moisture

        if scheduler_event.additional_moisture is not None:
            if self.actions_moisture is None:
                self.actions_moisture = np.zeros_like(self.moisture)
            self.actions_moisture += scheduler_event.additional_moisture
            self.actions_moisture = np.clip(self.actions_moisture, 0.0, 1.0)

        if scheduler_event.wind_dir is not None:
            self.wind_dir = scheduler_event.wind_dir

        if scheduler_event.wind_speed is not None:
            self.wind_speed = scheduler_event.wind_speed

        if scheduler_event.vegetation_changes is not None:
            # mutate vegetation where needed
            mask = ~np.isnan(scheduler_event.vegetation_changes)
            self.veg[mask] = scheduler_event.vegetation_changes[mask]

        if scheduler_event.updates is not None:
            self.apply_updates(scheduler_event.updates)

    def get_output(self) -> PropagatorOutput:
        """Assemble the current outputs and summary stats into a dataclass.

        Returns:
            PropagatorOutput: Snapshot of fire probability,
                RoS, intensity, stats.
        """
        fire_probability = self.compute_fire_probability()
        ros_max = self.compute_ros_max()
        ros_mean = self.compute_ros_mean()
        fireline_intensity_max = self.compute_fireline_int_max()
        fireline_intensity_mean = self.compute_fireline_int_mean()
        stats = self.compute_stats(fire_probability)

        return PropagatorOutput(
            time=self.time,
            fire_probability=fire_probability,
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
        if len(self.scheduler) == 0:
            return None

        return self.scheduler.next_time()
