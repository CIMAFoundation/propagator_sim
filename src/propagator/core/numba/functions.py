"""Spread, probability, and intensity model functions.

This module contains pluggable formulations for rate-of-spread, probability
modulators for wind/slope/moisture, fire spotting distance, and fireline
intensity utilities used by the core propagator.
"""

from typing import Any, Literal

import numpy as np
from numba import jit  # type: ignore

# constants for wind/slope effect
D1 = 0.5
D2 = 1.4
D3 = 8.2
D4 = 2.0
D5 = 50.0
A = 1 - ((D1 * (D2 * np.tanh((0 / D3) - D4))) + (0 / D5))

# Fire-spotting distance coefficient
FIRE_SPOTTING_DISTANCE_COEFFICIENT = 0.191

# Rothermel parameters
ROTHERMEL_ALPHA1 = 0.0693
ROTHERMEL_ALPHA2 = 0.0576

# Wang parameters
WANG_BETA1 = 0.1783
WANG_BETA2 = 3.533
WANG_BETA3 = 1.2

# Moisture constants
# probabilità
M1 = -3.5995
M2 = 5.2389
M3 = -2.6355
M4 = 1.019

# RoS
C_MOIST = -0.014


# variable for fireline intensity
Q = 2442.0

# Moisture of extinction
MOISTURE_OF_EXTINCTION = 0.3

RateOfSpreadModel = Literal["wang", "rothermel"]
MoistureModel = Literal["trucchia", "baghino"]


@jit(cache=True)
def clip(x: float, min: float, max: float) -> float:
    """Clip x to the range [min, max].

    Parameters
    ----------
    x : float
        Value to clip.
    min : float
        Minimum value.
    max : float
        Maximum value.

    Returns
    -------
    float
        Clipped value.
    """
    if x < min:
        return min
    if x > max:
        return max
    return x


def get_p_time_fn(ros_model_code: RateOfSpreadModel) -> Any:
    """Select a rate-of-spread model by code.

    Parameters
    ----------
    ros_model_code : RateOfSpreadModel
        The code of the rate-of-spread model to select.

    Returns
    --------
        function with signature
        `(v0, dem_from, dem_to, angle_to, dist, moist, w_dir, w_speed) -> (time_seconds, ros)`.
    """
    match ros_model_code:
        case "standard":
            return p_time_standard
        case "wang":
            return p_time_wang
        case "rothermel":
            return p_time_rothermel

    raise ValueError(f"Unknown ros_model_code: {ros_model_code!r}")


def get_p_moisture_fn(moist_model_code: MoistureModel) -> Any:
    """Select a moisture probability correction by code.

    Parameters
    ----------
    moist_model_code : MoistureModel
        The code of the moisture model to select.

    Returns
    -------
        function with signature
        `(moist: float) -> float`.
    """
    match moist_model_code:
        case "trucchia":
            return p_moisture_trucchia
        case "baghino":
            return p_moisture_baghino

    raise ValueError(f"Unknown moist_model_code: {moist_model_code!r}")


@jit(cache=True)
def p_time_rothermel(
    v0: float,
    dh: float,
    angle: float,
    dist: float,
    moist: float,
    w_dir: float,
    w_speed: float,
) -> tuple[float, float]:
    """Propagation time and ROS according to Rothermel-like scaling.

    Parameters
    ----------
    v0 : float
        Base Rate of Spread for the cell vegetation (m/min)
    dh : float
        Elevation difference between source and neighbor cells. (m)
    angle : float
        Direction to neighbor (clockwise radians, 0 is north -> south)
    dist : float
        Distance between cells (m).
    moist : float
        Moisture values (fraction).
    w_dir : float
        Wind direction (clockwise radians, 0 is north -> south).
    w_speed : float
        Wind speed (km/h).

    Returns
    -------
    tuple[float, float]
        (transition time [s], ROS [m/min]).
    """

    real_dist = np.sqrt(dist**2 + dh**2)

    # wind component in propagation direction
    w_proj = np.cos(w_dir - angle)
    # wind speed in the direction of propagation
    w_spd = (w_speed * w_proj) / 3.6

    teta_s_rad = np.arctan(dh / dist)  # slope angle [rad]
    teta_s = np.degrees(teta_s_rad)  # slope angle [°]

    # flame angle measured from the vertical
    # in the direction of fire spread [rad]
    teta_f_rad = np.arctan(0.4226 * w_spd)
    teta_f = np.degrees(teta_f_rad)  # flame angle [°]

    sf = np.exp(ROTHERMEL_ALPHA1 * teta_s)  # slope factor
    sf_clip = clip(sf, 0.01, 10)  # slope factor clipped at 10
    wf = np.exp(ROTHERMEL_ALPHA2 * teta_f)  # wind factor
    wf_rescaled = wf / 13  # wind factor rescaled to have 10 as max value
    wf_clip = clip(wf_rescaled, 1, 20)  # max value is 20, min is 1

    v_wh_pre = (
        v0 * sf_clip * wf_clip
    )  # Rate of Spread evaluate with Rothermel's model
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    # v_wh = clip(v_wh_pre, 0.01, 100) #adoptable RoS
    v_wh = clip(v_wh_pre * moist_eff, 0.01, 100)  # adoptable RoS [m/min]

    t = real_dist / v_wh

    time_seconds = t * 60.0

    return time_seconds, v_wh


@jit(cache=True)
def p_time_wang(
    v0: float,
    dh: float,
    angle: float,
    dist: float,
    moist: float,
    w_dir: float,
    w_speed: float,
) -> tuple[float, float]:
    """Propagation time and ROS according to Wang et al.

    Parameters
    ----------
    v0 : float
        Base ROS vector per vegetation type.
    dh : float
        Elevation at source and neighbor cells.
    angle : float
        Direction to neighbor (clockwise radians, 0 is north -> south).
    dist : float
        Distance to neighbour cell (m).
    moist : float
        Moisture values (fractional).
    w_dir : float
        Wind direction (clockwise radians, 0 is north -> south).
    w_speed : float
        Wind speed (km/h).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (transition time [s], ROS [m/min]).
    """
    # velocità di base modulata con la densità(tempo di attraversamento)

    real_dist = np.sqrt(dist**2 + dh**2)

    # wind component in propagation direction
    w_proj = np.cos(w_dir - angle)
    # wind speed in the direction of propagation
    w_spd = (w_speed * w_proj) / 3.6

    teta_s_rad = np.arctan(dh / dist)  # slope angle [rad]
    teta_s_pos = np.absolute(teta_s_rad)  # absolute values of slope angle
    # +1 if fire spreads upslope, -1 if fire spreads downslope
    p_reverse = np.sign(dh)

    wf = np.exp(WANG_BETA1 * w_spd)  # wind factor
    wf_clip = clip(wf, 0.01, 10)  # clipped at 10
    sf = np.exp(
        p_reverse * WANG_BETA2 * np.tan(teta_s_pos) ** WANG_BETA3
    )  # slope factor
    sf_clip = clip(sf, 0.01, 10)

    # Rate of Spread evaluate with Wang Zhengfei's model
    v_wh_pre = v0 * wf_clip * sf_clip
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    # v_wh = clip(v_wh_pre, 0.01, 100) #adoptable RoS
    v_wh = clip(v_wh_pre * moist_eff, 0.01, 100)  # adoptable RoS [m/min]

    t = real_dist / v_wh

    time_seconds = t * 60.0

    return time_seconds, v_wh


@jit(cache=True)
def p_time_standard(
    v0: float,
    dh: float,
    angle: float,
    dist: float,
    moist: float,
    w_dir: float,
    w_speed: float,
) -> tuple[float, float]:
    """Baseline propagation time and ROS with combined wind-slope factor.

    Parameters
    ----------
    v0 : float
        Base ROS vector per vegetation type.
    dh : float
        Elevation difference between source and neighbor cells.
    angle : float
        Direction to neighbor (clockwise radians, 0 is north -> south).
    dist : float
        Distance to neighbor (m).
    moist : float
        Moisture values (%).
    w_dir : float
        Wind direction (clockwise radians, 0 is north -> south).
    w_speed : float
        Wind speed (km/h).

    Returns
    -------
    tuple[float, float]
        (transition time [s], ROS [m/min]).
    """
    wh = w_h_effect(angle, w_speed, w_dir, dh, dist)
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    v_wh = clip(v0 * wh * moist_eff, 0.01, 100)

    real_dist = np.sqrt(dist**2 + dh**2)
    t = real_dist / v_wh
    time_seconds = t * 60.0
    return time_seconds, v_wh


@jit(cache=True)
def w_h_effect(
    angle: float,
    w_speed: float,
    w_dir: float,
    dh: float,
    dist: float,
) -> float:
    """
    Scale factor taking into account wind, slope and aspect effects on propagation rate of spread.

    Parameters
    ----------
    angle : float
        The angle to the neighboring pixel (clockwise radians, 0 is north -> south).
    w_speed : float
        The wind speed (km/h).
    w_dir : float
        The wind direction (clockwise radians, 0 is north -> south).
    dh : float
        The elevation difference between source and neighbor cells (meters).
    dist : float
        The distance to the neighbor (meters).

    Returns
    -------
    float
        Scale factor for wind, slope, and aspect effects on propagation rate of spread.
    """
    w_effect_module = (
        A + (D1 * (D2 * np.tanh((w_speed / D3) - D4))) + (w_speed / D5)
    )
    a = (w_effect_module - 1) / 4
    w_effect_on_direction = (
        (a + 1) * (1 - a**2) / (1 - a * np.cos(w_dir - angle))
    )
    slope = dh / dist
    h_effect = 2 ** (np.tanh((slope * 3) ** 2.0 * np.sign(slope)))

    w_h = h_effect * w_effect_on_direction
    return w_h


@jit(cache=True)
def w_h_effect_on_probability(
    angle: float,
    w_speed: float,
    w_dir: float,
    dh: float,
    dist: float,
) -> float:
    """
    Scale factor taking into account wind, slope and aspect effects on probability.
    This is derived from `w_h_effect` by scaling the output non linearly.

    Parameters
    ----------
    angle : float
        The angle to the neighboring pixel (clockwise radians, 0 is north -> south).
    w_speed : float
        The wind speed (km/h).
    w_dir : float
        The wind direction (clockwise radians, 0 is north -> south).
    dh : float
        The elevation difference between source and neighbor cells (meters).
    dist : float
        The distance to the neighbor (meters).

    Returns
    -------
    float
        Scale factor for wind, slope, and aspect effects on probability.
    """
    w_speed_norm = clip(w_speed, 0, 60)
    wh_orig = w_h_effect(angle, w_speed_norm, w_dir, dh, dist)
    wh = wh_orig - 1.0
    if wh > 0:
        wh = wh / 2.13
    elif wh < 0:
        wh = wh / 1.12
    wh += 1.0
    return wh


@jit(cache=True)
def p_moisture_trucchia(
    moist: float,
) -> float:
    """
    Moisture correction to the transition probability p_{i,j}.

    Uses a 5th-degree polynomial in x = moist/Mx, with Mx = 0.3
    (Trucchia et al., Fire 2020).

    Parameters
    ----------
    moist : float
        Moisture content (fractional).

    Returns
    -------
    float
        Moisture correction factor (p_{i,j}).
    """
    x = moist / MOISTURE_OF_EXTINCTION
    p_moist = (
        (-11.507 * x**5)
        + (22.963 * x**4)
        + (-17.331 * x**3)
        + (6.598 * x**2)
        + (-1.7211 * x)
        + 1.0003
    )
    p_moist = clip(p_moist, 0.0, 1.0)
    return p_moist


@jit(cache=True)
def p_moisture_baghino(
    moist: float,
) -> float:
    """
    Moisture correction to p_{i,j}.
    Older formulation, Baghino; Trucchia et al., 2020).
    Parameters come from constants.

    Parameters
    ----------
    moist : float
        Moisture content (fractional).

    Returns
    -------
    float
        Moisture correction factor (p_{i,j}).
    """
    p_moist = M1 * moist**3 + M2 * moist**2 + M3 * moist + M4
    return p_moist


@jit(cache=True)
def lhv_fuel(
    hhv: float,
    moisture: float,
) -> float:
    """
    Lower heating value of fuels given higher
    heating value and FFMC.

    Parameters
    ----------
    hhv : float
        Higher heating value of dead fuel (kJ/kg).
    moisture : float
        fuel moisture content (fractional).
    """
    lhv = hhv * (1.0 - moisture) - Q * moisture
    return lhv


@jit(cache=True)
def fireline_intensity(
    d0: float, d1: float, ros: float, lhv_dead_fuel: float, lhv_canopy: float
) -> float:
    """
    Estimate fireline intensity (kW/m) from fuel loads and Rate of spread.

    Parameters
    ----------
    d0 : float
        Dead fuel density (kg/m^2).
    d1 : float
        Canopy fuel density (kg/m^2).
    ros : float
        Rate of spread (m/min).
    lhv_dead_fuel : float
        Lower heating value of dead fuel (kJ/kg).
    lhv_canopy : float
        Lower heating value of canopy fuel (kJ/kg).

    Returns
    -------
    float
        Fireline intensity (kW/m).
    """
    intensity = (ros / 60) * (lhv_dead_fuel * d0 + lhv_canopy * d1)
    return intensity


@jit(cache=True, nopython=True, fastmath=True)
def get_probability_to_neighbour(
    angle: float,
    dist: float,
    w_dir: float,
    w_speed: float,
    moisture: float,
    dh: float,
    transition_probability: float,
    p_moist_fn: Any,
) -> float:
    """
    Get the probability of fire spread to a neighboring pixel.

    Parameters
    ----------
    angle: float
        The angle to the neighboring pixel (clockwise radians, 0 is north -> south).
    dist: float
        The distance to the neighboring pixel (meters).
    w_dir: float
        The wind direction (clockwise radians, 0 is north -> south).
    w_speed: float
        The wind speed (km/h).
    moisture: float
        The moisture content (fraction).
    dh: float
        The difference in height (meters).
    transition_probability: float
        The base transition probability.
    p_moist_fn: Any
        The function to compute the moisture probability (must be jit-compiled). Units are compliant with other functions.
            signature: (moist: float) -> float

    Returns
    -------
    float
        The probability of fire spread to the neighboring pixel.
    """

    moisture_effect = p_moist_fn(moisture)
    alpha_wh = w_h_effect_on_probability(angle, w_speed, w_dir, dh, dist)

    alpha_wh = np.maximum(alpha_wh, 0)  # prevent alpha < 0
    p_prob = 1 - (1 - transition_probability) ** alpha_wh
    p_prob = clip(p_prob * moisture_effect, 0, 1.0)
    # try the propagation
    return p_prob
