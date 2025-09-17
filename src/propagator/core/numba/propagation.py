"""Propagation functions for fire spread simulation.

This module contains functions for simulating fire spread in a grid environment,
including ember spotting and cell updates.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from numba import jit
from numpy.random import normal, poisson, random, uniform

from propagator.core.constants import NO_FUEL
from propagator.core.models import UpdateBatchTuple
from propagator.core.numba.functions import FIRE_SPOTTING_DISTANCE_COEFFICIENT

from .functions import (
    fireline_intensity,
    get_probability_to_neighbour,
    lhv_fuel,
)
from .models import Fuel, FuelSystem

# P_c = P_c0 (1 + P_cd), where P_c0 constant spread_probability of
# ignition by spotting and P_cd is a correction factor that
# depends on vegetation type and density...
P_C0 = 0.6

# The following constants are used in the Fire-Spotting model.
# Alexandridis et al. (2009,2011)
LAMBDA_SPOTTING = 2.0
SPOTTING_RN_MEAN = 100
SPOTTING_RN_STD = 25

NEIGHBOURS = np.array(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
)
NEIGHBOURS_DISTANCE = np.array([1.414, 1, 1.414, 1, 1, 1.414, 1, 1.414])
NEIGHBOURS_ANGLE = np.array(
    [
        np.pi * 3 / 4,
        np.pi / 2,
        np.pi / 4,
        np.pi,
        0,
        -np.pi * 3 / 4,
        -np.pi / 2,
        -np.pi / 4,
    ]
)


@jit(cache=True)
def fire_spotting(
    angle: float,
    w_dir: float,
    w_speed: float,
) -> tuple[float, float]:
    """Evaluate spotting distance using Alexandridis' formulation.

    Parameters
    ----------
    angle : float
        The angle of the ember's trajectory (radians between [-π, π], 0 is east->west)
    w_dir : float
        The wind direction (radians between [-π, π], 0 is east->west)
    w_speed : float
        The wind speed (km/h)

    Returns
    -------
    tuple[float, float]
        The spotting distance (meters) and the landing time (minutes)
    """
    r_n = normal(
        SPOTTING_RN_MEAN, SPOTTING_RN_STD
    )  # main thrust of the ember: sampled from a
    # Gaussian Distribution (Alexandridis et al, 2008 and 2011)
    w_speed_ms = w_speed / 3.6  # wind speed [m/s]
    # Alexandridis' formulation for spotting distance
    ember_distance = r_n * np.exp(
        w_speed_ms
        * FIRE_SPOTTING_DISTANCE_COEFFICIENT
        * (np.cos(w_dir - angle) - 1)
    )
    ember_landing_time_min = int(ember_distance / w_speed_ms / 60)
    return ember_distance, ember_landing_time_min


@jit(cache=True, nopython=True, fastmath=True)
def compute_spotting(
    row: int,
    col: int,
    cellsize: float,
    veg: npt.NDArray[np.integer],
    fire: npt.NDArray[np.int8],
    wind_dir: float,
    wind_speed: float,
    fuels: FuelSystem,
) -> list[tuple[int, int, int, float, float]]:
    """
    Compute ember spotting updates for a given cell.

    Parameters
    ----------
    row : int
        The row index of the current cell
    col : int
        The column index of the current cell
    cellsize : float
        The size of each cell (m)
    veg : npt.NDArray[np.integer]
        The vegetation type array
    fire : npt.NDArray[np.int8]
        The fire state array
    wind_dir : float
        The wind direction (radians between [-π, π], 0 is east->west)
    wind_speed : float
        The wind speed (km/h)
    fuels : FuelSystem
        The fuel system object

    Returns
    -------
    list[tuple[int, int, int, float, float]]
        A list of spotting updates, each represented as a tuple
        (transition_times, rows, cols, rates_of_spread, fireline_intensities)
    """

    # calculate number of embers per emitter > Poisson distribution
    spotting_updates = []

    num_embers = poisson(LAMBDA_SPOTTING)

    if num_embers == 0:
        return spotting_updates

    for _ in range(num_embers):
        # calculate angle > uniform distribution
        ember_angle = uniform(0, 2.0 * np.pi)
        # calculate distance > depends on wind speed and direction
        # NOTE: it is computed considering wind speed and direction
        # of the cell of origin of the ember
        ember_distance, ember_landing_time = fire_spotting(
            ember_angle,
            wind_dir,
            wind_speed,
        )

        # filter out short embers
        if ember_distance < 2 * cellsize:
            continue

        # calculate landing locations
        # vertical delta [meters]
        delta_r = ember_distance * np.cos(ember_angle)
        # horizontal delta [meters]
        delta_c = ember_distance * np.sin(ember_angle)

        # location of the cell to be ignited by the ember
        row_to = row + int(delta_r / cellsize)
        col_to = col + int(delta_c / cellsize)

        # check if the landing location is within the grid, otherwise discard
        if col_to < 0 or col_to > fire.shape[1] - 1:
            continue

        if row_to < 0 or row_to > fire.shape[0] - 1:
            continue

        # prevent ignition of already burning cells
        if fire[row_to, col_to] != 0:
            continue
        veg_to = veg[row_to, col_to]
        if veg_to == NO_FUEL:
            continue

        # we want to put another probabilistic filter in order
        # to assess the success of ember ignition.
        # Formula (10) of Alexandridis et al IJWLF 2011
        # P_c = P_c0 (1 + P_cd), where P_c0 constant probability of ignition
        # by spotting and P_cd is a correction factor that
        # depends on vegetation type and density > set on the fuels system
        fuel_to = fuels.get_fuel(veg_to)  # type: ignore

        P_c = P_C0 * (1 + fuel_to.prob_ign_by_embers)
        if uniform() > P_c:
            continue

        ember_landing_time = max(ember_landing_time, 1)

        spotting_update = (ember_landing_time, row_to, col_to, np.nan, np.nan)
        spotting_updates.append(spotting_update)

    return spotting_updates


@jit(cache=True, nopython=True, fastmath=True)
def calculate_fire_behavior(
    fuel_from: Fuel,
    fuel_to: Fuel,
    dh: float,
    dist: float,
    angle: float,
    moisture: float,
    w_dir: float,
    w_speed: float,
    p_time_fn: Any,
) -> tuple[int, float, float]:
    """Calculate fire behaviour during propagation between cells

    Parameters
    ----------
    fuel_from : Fuel
        The fuel object for the source cell.
    fuel_to : Fuel
        The fuel object for the target cell.
    dh : float
        The elevation difference between the source and target cells (m).
    dist : float
        The distance to the target cell (m).
    angle : float
        The angle to the target cell (radians between [-π, π], 0 is east->west).
    moisture : float
        The moisture content of the fuel (fraction).
    w_dir : float
        The wind direction (radians between [-π, π], 0 is east->west).
    w_speed : float
        The wind speed (km/h).
    p_time_fn: Any
        The function to compute the spread time (must be jit-compiled). Units are compliant with other functions.
            signature: (v0: float, dh: float, angle_to: float, dist: float, moist: float, w_dir: float, w_speed: float) -> tuple[float, float]


    Returns
    -------
    tuple[int, float, float]
        A tuple containing the transition time, rate of spread, and fireline intensity.
    """

    transition_time, ros_value = p_time_fn(
        fuel_from.v0,
        dh,
        angle,
        dist,
        moisture,
        w_dir,
        w_speed,
    )

    transition_time = int(transition_time)
    if transition_time < 1:
        transition_time = 1

    # evaluate LHV of dead fuel
    lhv_dead_fuel_value = lhv_fuel(fuel_to.hhv, moisture)
    # evaluate LHV of the canopy
    lhv_canopy_value = lhv_fuel(fuel_to.hhv, fuel_to.humidity)
    # evaluate fireline intensity
    fireline_intensity_value = fireline_intensity(
        fuel_to.d0,
        fuel_to.d1,
        ros_value,
        lhv_dead_fuel_value,
        lhv_canopy_value,
    )
    return transition_time, ros_value, fireline_intensity_value


@jit(cache=True, parallel=False, nopython=True, fastmath=True)
def single_cell_updates(
    row: int,
    col: int,
    cellsize: float,
    veg: npt.NDArray[np.integer],
    dem: npt.NDArray[np.floating],
    fire: npt.NDArray[np.int8],
    moisture: npt.NDArray[np.floating],
    wind_dir: npt.NDArray[np.floating],
    wind_speed: npt.NDArray[np.floating],
    fuels: FuelSystem,
    p_time_fn: Any,
    p_moist_fn: Any,
) -> list[tuple[int, int, int, float, float]]:
    """
    Apply fire spread to a single cell and get the next spread updates.

    Parameters
    ----------
    row : int
        The row index of the current cell
    col : int
        The column index of the current cell
    cellsize: float
        The size of each cell (in meters)
    veg : npt.NDArray[np.integer]
        The 2D vegetation array
    dem : npt.NDArray[np.floating]
        The 2D digital elevation model array
    fire: npt.NDArray[np.int8]
        The 2D current fire state
    moisture: npt.NDArray[np.floating]
        The 2D moisture array (units: fraction [0, 1])
    wind_dir: npt.NDArray[np.floating]
        The 2D wind direction array (units: radians between [-π, π], 0 is east->west)
    wind_speed: npt.NDArray[np.floating]
        The 2D wind speed array (units: km/h)
    fuels: FuelSystem
        The fuel system
    p_time_fn: Any
        The function to compute the spread time (must be jit-compiled). Units are compliant with other functions.
            signature: (v0: float, dh: float, angle_to: float, dist: float, moist: float, w_dir: float, w_speed: float) -> tuple[float, float]
    p_moist_fn: Any
        The function to compute the moisture probability (must be jit-compiled). Units are compliant with other functions.
            signature: (moist: float) -> float

    Returns
    -------
    list[tuple[int, int, int, float, float]]
        A list of fire spread updates (transition_times, rows, cols, rates_of_spread, fireline_intensities)
    """
    fire_spread_updates = []

    dem_from = dem[row, col]
    veg_from = veg[row, col]

    if veg_from == NO_FUEL:
        return fire_spread_updates

    w_dir_r = wind_dir[row, col]
    w_speed_r = wind_speed[row, col]

    fuel_from = fuels.get_fuel(veg_from)  # type: ignore

    for neighbour, dist_to_lattice, angle_to in zip(
        NEIGHBOURS, NEIGHBOURS_DISTANCE, NEIGHBOURS_ANGLE
    ):
        row_to = row + neighbour[0]
        col_to = col + neighbour[1]
        veg_to = veg[row_to, col_to]
        dist_to = dist_to_lattice * cellsize

        # keep only pixels where fire can spread
        if fire[row_to, col_to] != 0 or veg_to == NO_FUEL:
            continue

        dh = dem[row_to, col_to] - dem_from
        moisture_r = moisture[row_to, col_to]
        transition_probability = fuels.get_transition_probability(
            veg_from,
            veg_to,  # type: ignore
        )

        p_prob = get_probability_to_neighbour(
            angle_to,
            dist_to,
            w_dir_r,
            w_speed_r,
            moisture_r,  # type: ignore
            dh,
            transition_probability,
            p_moist_fn,
        )

        do_propagate = p_prob > random()
        if not do_propagate:
            continue

        fuel_to = fuels.get_fuel(veg_to)  # type: ignore

        transition_time, ros, fireline_intensity = calculate_fire_behavior(
            fuel_from,
            fuel_to,
            dh,
            dist_to,
            angle_to,
            moisture_r,  # type: ignore
            w_dir_r,
            w_speed_r,
            p_time_fn,
        )
        fire_spread_updates.append(
            (transition_time, row_to, col_to, ros, fireline_intensity)
        )

    if fuel_from.spotting:
        spotting_updates = compute_spotting(
            row,
            col,
            cellsize,
            veg,
            fire,
            wind_dir[row, col],
            wind_speed[row, col],
            fuels,
        )
        fire_spread_updates.extend(spotting_updates)

    return fire_spread_updates


@jit(cache=True, parallel=False, nopython=True, fastmath=True)
def next_updates_fn(
    rows: npt.NDArray[np.integer],
    cols: npt.NDArray[np.integer],
    realizations: npt.NDArray[np.integer],
    cellsize: float,
    time: int,
    veg: npt.NDArray[np.integer],
    dem: npt.NDArray[np.floating],
    fire: npt.NDArray[np.int8],
    moisture: npt.NDArray[np.floating],
    wind_dir: npt.NDArray[np.floating],
    wind_speed: npt.NDArray[np.floating],
    fuels: FuelSystem,
    p_time_fn: Any,
    p_moist_fn: Any,
) -> UpdateBatchTuple:
    """
    Compute the next updates for the fire spread simulation.

    Parameters
    ----------
    rows: npt.NDArray[np.integer]
        The row indices of the fire spread updates.
    cols: npt.NDArray[np.integer]
        The column indices of the fire spread updates.
    realizations: npt.NDArray[np.integer]
        The realization indices of the fire spread updates.
    cellsize: float
        The size of each cell (in meters)
    time: int
        The current time step.
    veg : npt.NDArray[np.integer]
        The 2D vegetation array
    dem : npt.NDArray[np.floating]
        The 2D digital elevation model array
    fire: npt.NDArray[np.int8]
        The 3D current fire state
    moisture: npt.NDArray[np.floating]
        The 2D moisture array (units: fraction [0, 1])
    wind_dir: npt.NDArray[np.floating]
        The 2D wind direction array (units: radians between [-π, π], 0 is east->west)
    wind_speed: npt.NDArray[np.floating]
        The 2D wind speed array (units: km/h)
    fuels: FuelSystem
        The fuel system
    p_time_fn: Any
        The function to compute the spread time (must be jit-compiled). Units are compliant with other functions.
            signature: (v0: float, dh: float, angle_to: float, dist: float, moist: float, w_dir: float, w_speed: float) -> tuple[float, float]
    p_moist_fn: Any
        The function to compute the moisture probability (must be jit-compiled). Units are compliant with other functions.
            signature: (moist: float) -> float

    Returns
    -------
    UpdateBatchTuple
        A tuple containing the arrays for the next updates.
        (next_times, next_rows, next_cols, next_realizations, next_ros, next_fireline_intensities)
    """
    next_rows = []
    next_cols = []
    next_realizations = []
    next_times = []
    next_ros = []
    next_fireline_intensities = []

    for index in range(len(rows)):
        row: int = rows[index]
        col: int = cols[index]
        realization: int = realizations[index]

        fire_spread_update = single_cell_updates(
            row,
            col,
            cellsize,
            veg,
            dem,
            fire[:, :, realization],
            moisture,
            wind_dir,
            wind_speed,  # type: ignore
            fuels,
            p_time_fn,
            p_moist_fn,
        )

        for fire_spread in fire_spread_update:
            (transition_time, row_to, col_to, ros, fireline_intensity) = (
                fire_spread
            )
            next_times.append(time + transition_time)
            next_rows.append(row_to)
            next_cols.append(col_to)
            next_realizations.append(realization)
            next_ros.append(ros)
            next_fireline_intensities.append(fireline_intensity)

    return (
        np.array(next_times),
        np.array(next_rows),
        np.array(next_cols),
        np.array(next_realizations),
        np.array(next_ros),
        np.array(next_fireline_intensities),
    )
