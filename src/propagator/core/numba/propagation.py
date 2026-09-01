"""Propagation functions for fire spread simulation.

This module contains functions for simulating fire spread in a grid environment,
including ember spotting and cell updates.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from numba import jit  # type: ignore
from numpy.random import lognormal, poisson, random, uniform

from propagator.core.constants import NO_FUEL

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

# Poisson mean number of embers emitted by a burning spotting-prone cell.
# Alexandridis et al. (2009, 2011)
LAMBDA_SPOTTING = 2.0

# Wind- and intensity-scaled landing-distance model (see docs/spotting.md).
# The median landing distance is
#     d_median = SPOTTING_DISTANCE_REF
#              * (U / SPOTTING_WIND_REF)
#              * (I / SPOTTING_FLI_REF) ** SPOTTING_FLI_EXPONENT
# so distance -> 0 as wind -> 0 (spotting is a wind-driven phenomenon) and
# grows with the source cell's fireline intensity through plume lofting.
SPOTTING_DISTANCE_REF = (
    100.0  # m,    median landing distance at the reference state
)
SPOTTING_WIND_REF = 20.0  # km/h, reference wind speed
SPOTTING_FLI_REF = 10000.0  # kW/m, reference fireline intensity
SPOTTING_FLI_EXPONENT = (
    1.0 / 3.0
)  # loft chaining: H ~ I^(2/3), d ~ U*sqrt(H) ~ U*I^(1/3)
# Downwind concentration: a wind-INDEPENDENT shape factor that shortens embers
# thrown across/against the wind (= 1 downwind). Keeping it decoupled from wind
# speed is the key fix for spurious isotropic spotting at very low wind.
SPOTTING_ANISOTROPY = 5.0
# Lognormal spread of the landing distance about its median (Sardoy et al. 2008).
SPOTTING_DISTANCE_LOG_SIGMA = 0.5
# Floor on the along-trajectory wind fraction cos(w_dir - angle) used for the
# ember travel time, so near-crosswind embers do not get an unbounded time.
SPOTTING_MIN_ALIGNMENT = 0.2

# Delay between ember landing and the development of a fire capable of
# propagation, sampled from a lognormal distribution.
SPOTTING_TIME_TO_PROPAGATION_MEDIAN = 600.0
SPOTTING_TIME_TO_PROPAGATION_LOG_SIGMA = 0.4


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
# calculate the distance to the neighbours in a lattice from NEIGHBOURS
NEIGHBOURS_DISTANCE = np.sqrt(NEIGHBOURS[:, 0] ** 2 + NEIGHBOURS[:, 1] ** 2)
# calculate the angle to the neighbours in a lattice from NEIGHBOURS using meteorological convention
# 0 is north->south, pi/2 is east->west
NEIGHBOURS_ANGLE = (
    np.arctan2(NEIGHBOURS[:, 1], -NEIGHBOURS[:, 0]) + np.pi
) % (2 * np.pi)


@jit(cache=False)
def fire_spotting(
    angle: float,
    w_dir: float,
    w_speed: float,
    fireline_intensity_value: float,
) -> tuple[float, float]:
    """Sample the landing distance and travel time of a single ember.

    The landing distance is drawn from a lognormal distribution (Sardoy et
    al. 2008) whose median scales linearly with wind speed and sub-linearly
    with the source cell's fireline intensity, and whose central value is
    concentrated downwind. See ``docs/spotting.md`` for the full rationale.

    Parameters
    ----------
    angle : float
        The angle of the ember's trajectory (clockwise radians, 0 is north -> south)
    w_dir : float
        The wind direction (clockwise radians, 0 is north -> south)
    w_speed : float
        The wind speed (km/h)
    fireline_intensity_value : float
        The fireline intensity of the source (emitting) cell (kW/m)

    Returns
    -------
    tuple[float, float]
        The spotting distance (meters) and the landing time (seconds)
    """
    w_speed_ms = w_speed / 3.6  # wind speed [m/s]
    # Embers are wind-carried and lofted by the fire's own plume: with no
    # wind or no fire intensity there is no transport, hence no spotting.
    if w_speed_ms <= 0.0 or fireline_intensity_value <= 0.0:
        return 0.0, 1.0

    # Median landing distance: linear in wind speed (so it collapses to zero
    # as the wind dies) and ~I^(1/3) in fireline intensity through lofting.
    d_median = (
        SPOTTING_DISTANCE_REF
        * (w_speed / SPOTTING_WIND_REF)
        * (fireline_intensity_value / SPOTTING_FLI_REF)
        ** SPOTTING_FLI_EXPONENT
    )

    # Downwind concentration: shortens embers thrown across/against the wind.
    # This is a pure shape factor, decoupled from wind magnitude.
    alignment = np.cos(w_dir - angle)
    directional = np.exp(SPOTTING_ANISOTROPY * (alignment - 1.0))

    # Lognormal landing distance about the (directional) median.
    median = d_median * directional
    ember_distance = lognormal(np.log(median), SPOTTING_DISTANCE_LOG_SIGMA)

    # Travel time uses the wind component along the ember trajectory
    # (U * cos(w_dir - angle)); the alignment is floored to avoid a
    # singularity for near-crosswind embers.
    transport_speed = w_speed_ms * max(alignment, SPOTTING_MIN_ALIGNMENT)
    ember_landing_time_sec = ember_distance / transport_speed
    return ember_distance, ember_landing_time_sec


@jit(cache=False, nopython=True, fastmath=True)
def compute_spotting(
    row: int,
    col: int,
    cellsize: float,
    veg: npt.NDArray[np.integer],
    fire: npt.NDArray[np.int8],
    wind_dir: float,
    wind_speed: float,
    fireline_intensity_value: float,
    fuels: FuelSystem,
) -> list[tuple[int, int, int, float, float, bool]]:
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
        The wind direction (clockwise radians, 0 is north -> south)
    wind_speed : float
        The wind speed (km/h)
    fireline_intensity_value : float
        The fireline intensity of the source (emitting) cell (kW/m)
    fuels : FuelSystem
        The fuel system object

    Returns
    -------
    list[tuple[int, int, int, float, float, bool]]
        A list of spotting updates, each represented as a tuple
        (transition_times, rows, cols, rates_of_spread, fireline_intensities,
        is_spotting)
    """

    # calculate number of embers per emitter > Poisson distribution
    # let numba assign the type
    spotting_updates = []  # type: ignore

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
            fireline_intensity_value,
        )

        # filter out short embers
        if ember_distance < 2 * cellsize:
            continue

        # calculate landing locations
        # vertical delta [meters]
        delta_r = ember_distance * np.cos(ember_angle)
        # horizontal delta [meters]
        # Keep the same angle convention used by spread kernels:
        # 0 -> south, pi/2 -> west, pi -> north, 3pi/2 -> east.
        delta_c = -ember_distance * np.sin(ember_angle)

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
        fuel_to = fuels.get_fuel(veg_to)  # type: ignore
        if not fuel_to.burn:
            continue

        # we want to put another probabilistic filter in order
        # to assess the success of ember ignition.
        # Formula (10) of Alexandridis et al IJWLF 2011
        # P_c = P_c0 (1 + P_cd), where P_c0 constant probability of ignition
        # by spotting and P_cd is a correction factor that
        # depends on vegetation type and density > set on the fuels system

        P_c = P_C0 * (1 + fuel_to.prob_ign_by_embers)
        if uniform() > P_c:
            continue

        # Sample the delay between ember landing and the development of
        # a fire capable of propagation using a lognormal distribution:
        time_to_propagation = lognormal(
            np.log(SPOTTING_TIME_TO_PROPAGATION_MEDIAN),
            SPOTTING_TIME_TO_PROPAGATION_LOG_SIGMA,
        )

        ember_landing_time = max(
            int(np.ceil(ember_landing_time)),
            1,
        )

        time_to_propagation = max(
            int(np.ceil(time_to_propagation)),
            1,
        )

        propagation_time = ember_landing_time + time_to_propagation

        spotting_update = (propagation_time, row_to, col_to, 0.0, 0.0, True)
        spotting_updates.append(spotting_update)

    return spotting_updates


@jit(cache=False, nopython=True, fastmath=True)
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
        The angle to the target cell (clockwise radians, 0 is north -> south).
    moisture : float
        The moisture content of the fuel (fraction).
    w_dir : float
        The wind direction (clockwise radians, 0 is north -> south).
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


@jit(cache=False, parallel=False, nopython=True, fastmath=True)
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
    fireline_intensity_value: float,
) -> list[tuple[int, int, int, float, float, bool]]:
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
        The 2D wind direction array (units: radians, meteorological
        convention matching NEIGHBOURS_ANGLE: 0 is north->south,
        increasing clockwise)
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
    fireline_intensity_value: float
        The fireline intensity of this (source/emitting) cell (kW/m), used to
        scale the ember spotting distance.

    Returns
    -------
    list[tuple[int, int, int, float, float, bool]]
        A list of fire spread updates (transition_times, rows, cols,
        rates_of_spread, fireline_intensities, is_spotting)
    """

    # let numba assign the type
    fire_spread_updates = []  # type: ignore

    dem_from = dem[row, col]
    veg_from = veg[row, col]

    if veg_from == NO_FUEL:
        return fire_spread_updates

    w_dir_r = wind_dir[row, col]
    w_speed_r = wind_speed[row, col]

    fuel_from = fuels.get_fuel(veg_from)  # type: ignore
    if not fuel_from.burn:
        # A non-burnable fuel (e.g. a firebreak neutralized by a
        # firefighting action) can end up marked as burning via forced
        # ignition, but it must never itself propagate fire outward.
        return fire_spread_updates

    for neighbour, dist_to_lattice, angle_to in zip(
        NEIGHBOURS, NEIGHBOURS_DISTANCE, NEIGHBOURS_ANGLE
    ):
        row_to = row + neighbour[0]
        col_to = col + neighbour[1]

        # check if the neighbour is within the grid, otherwise discard
        if row_to < 0 or row_to >= fire.shape[0]:
            continue
        if col_to < 0 or col_to >= fire.shape[1]:
            continue

        veg_to = veg[row_to, col_to]
        dist_to = dist_to_lattice * cellsize

        # keep only pixels where fire can spread
        if fire[row_to, col_to] or veg_to == NO_FUEL:
            continue
        fuel_to = fuels.get_fuel(veg_to)  # type: ignore
        if not fuel_to.burn:
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
            (transition_time, row_to, col_to, ros, fireline_intensity, False)
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
            fireline_intensity_value,
            fuels,
        )
        fire_spread_updates.extend(spotting_updates)

    return fire_spread_updates
