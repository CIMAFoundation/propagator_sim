"""Propagation functions for the NumPy fire spread simulation backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from propagator.core.constants import NO_FUEL
from propagator.core.models import UpdateBatchTuple

from .functions import (
    FIRE_SPOTTING_DISTANCE_COEFFICIENT,
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
# calculate the distance to the neighbours in a lattice from NEIGHBOURS
NEIGHBOURS_DISTANCE = np.sqrt(NEIGHBOURS[:, 0] ** 2 + NEIGHBOURS[:, 1] ** 2)
# calculate the angle to the neighbours in a lattice from NEIGHBOURS using meteorological convention
# 0 is north->south, pi/2 is east->west
NEIGHBOURS_ANGLE = (
    np.arctan2(NEIGHBOURS[:, 1], -NEIGHBOURS[:, 0]) + np.pi
) % (2 * np.pi)


def fire_spotting(
    angle: float,
    w_dir: float,
    w_speed: float,
) -> tuple[float, float]:
    """Evaluate spotting distance using Alexandridis' formulation."""

    r_n = np.random.normal(SPOTTING_RN_MEAN, SPOTTING_RN_STD)
    w_speed_ms = w_speed / 3.6  # wind speed [m/s]
    if w_speed_ms <= 0:
        return 0.0, 1.0
    ember_distance = r_n * np.exp(
        w_speed_ms
        * FIRE_SPOTTING_DISTANCE_COEFFICIENT
        * (np.cos(w_dir - angle) - 1)
    )
    ember_landing_time_sec = ember_distance / w_speed_ms
    return ember_distance, ember_landing_time_sec


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
    """Compute ember spotting updates for a given cell."""

    spotting_updates: list[tuple[int, int, int, float, float]] = []

    num_embers = np.random.poisson(LAMBDA_SPOTTING)

    if num_embers == 0:
        return spotting_updates

    for _ in range(num_embers):
        ember_angle = np.random.uniform(0, 2.0 * np.pi)
        ember_distance, ember_landing_time = fire_spotting(
            ember_angle,
            wind_dir,
            wind_speed,
        )

        if ember_distance < 2 * cellsize:
            continue

        delta_r = ember_distance * np.cos(ember_angle)
        delta_c = ember_distance * np.sin(ember_angle)

        row_to = row + int(delta_r / cellsize)
        col_to = col + int(delta_c / cellsize)

        if (
            col_to < 0
            or col_to > fire.shape[1] - 1
            or row_to < 0
            or row_to > fire.shape[0] - 1
        ):
            continue

        if fire[row_to, col_to] != 0:
            continue
        veg_to = veg[row_to, col_to]
        if veg_to == NO_FUEL:
            continue

        fuel_to = fuels.get_fuel(veg_to)

        P_c = P_C0 * (1 + fuel_to.prob_ign_by_embers)
        if np.random.uniform() > P_c:
            continue

        ember_landing_time = max(int(ember_landing_time), 1)

        spotting_update = (ember_landing_time, row_to, col_to, np.nan, np.nan)
        spotting_updates.append(spotting_update)

    return spotting_updates


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
    """Calculate fire behaviour during propagation between cells."""

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

    lhv_dead_fuel_value = lhv_fuel(fuel_to.hhv, moisture)
    lhv_canopy_value = lhv_fuel(fuel_to.hhv, fuel_to.humidity)
    fireline_intensity_value = fireline_intensity(
        fuel_to.d0,
        fuel_to.d1,
        ros_value,
        lhv_dead_fuel_value,
        lhv_canopy_value,
    )
    return transition_time, ros_value, fireline_intensity_value


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
    """Apply fire spread to a single cell and get the next spread updates."""

    fire_spread_updates: list[tuple[int, int, int, float, float]] = []

    dem_from = dem[row, col]
    veg_from = veg[row, col]

    if veg_from == NO_FUEL:
        return fire_spread_updates

    w_dir_r = wind_dir[row, col]
    w_speed_r = wind_speed[row, col]

    fuel_from = fuels.get_fuel(veg_from)

    for neighbour, dist_to_lattice, angle_to in zip(
        NEIGHBOURS, NEIGHBOURS_DISTANCE, NEIGHBOURS_ANGLE
    ):
        row_to = row + neighbour[0]
        col_to = col + neighbour[1]

        if row_to < 0 or row_to >= fire.shape[0]:
            continue
        if col_to < 0 or col_to >= fire.shape[1]:
            continue

        veg_to = veg[row_to, col_to]
        dist_to = dist_to_lattice * cellsize

        if fire[row_to, col_to] != 0 or veg_to == NO_FUEL:
            continue

        dh = dem[row_to, col_to] - dem_from
        moisture_r = moisture[row_to, col_to]
        transition_probability = fuels.get_transition_probability(
            veg_from,
            veg_to,
        )

        p_prob = get_probability_to_neighbour(
            angle_to,
            dist_to,
            w_dir_r,
            w_speed_r,
            moisture_r,
            dh,
            transition_probability,
            p_moist_fn,
        )

        if p_prob <= np.random.random():
            continue

        fuel_to = fuels.get_fuel(veg_to)

        transition_time, ros, fireline_intensity_value = (
            calculate_fire_behavior(
                fuel_from,
                fuel_to,
                dh,
                dist_to,
                angle_to,
                moisture_r,
                w_dir_r,
                w_speed_r,
                p_time_fn,
            )
        )
        fire_spread_updates.append(
            (transition_time, row_to, col_to, ros, fireline_intensity_value)
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
    """Compute the next updates for the fire spread simulation."""

    next_rows: list[int] = []
    next_cols: list[int] = []
    next_realizations: list[int] = []
    next_times: list[int] = []
    next_ros: list[float] = []
    next_fireline_intensities: list[float] = []

    for index in range(len(rows)):
        row = int(rows[index])
        col = int(cols[index])
        realization = int(realizations[index])

        fire_spread_update = single_cell_updates(
            row,
            col,
            cellsize,
            veg,
            dem,
            fire[:, :, realization],
            moisture,
            wind_dir,
            wind_speed,
            fuels,
            p_time_fn,
            p_moist_fn,
        )

        for (
            transition_time,
            row_to,
            col_to,
            ros,
            fireline_intensity_value,
        ) in fire_spread_update:
            next_times.append(time + transition_time)
            next_rows.append(row_to)
            next_cols.append(col_to)
            next_realizations.append(realization)
            next_ros.append(ros)
            next_fireline_intensities.append(fireline_intensity_value)

    return (
        np.array(next_times),
        np.array(next_rows),
        np.array(next_cols),
        np.array(next_realizations),
        np.array(next_ros, dtype=np.float32),
        np.array(next_fireline_intensities, dtype=np.float32),
    )
