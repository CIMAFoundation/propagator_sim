import os
import numpy as np

from propagator.constants import (
    D1,
    D2,
    D3,
    D4,
    D5,
    M1,
    M2,
    M3,
    M4,
    A,
    Q,
    ROTHERMEL_ALPHA1,
    ROTHERMEL_ALPHA2,
    WANG_BETA1,
    WANG_BETA2,
    WANG_BETA3,
    FIRE_SPOTTING_DISTANCE_COEFFICIENT,
    C_MOIST,
    CELLSIZE,
    SPOTTING_RN_MEAN,
    SPOTTING_RN_STD,
)
from propagator.utils import (
    normalize,
)


def load_parameters(probability_file=None, v0_file=None, p_vegetation=None):
    """
    Override the default values for vegetation speed and probabilities by loading them from file
    :param probability_file:
    :param time_file:
    :return:
    """
    global v0, prob_table, p_veg
    if v0_file:
        v0 = np.loadtxt(v0_file)
    if probability_file:
        prob_table = np.loadtxt(probability_file)
    if p_vegetation:
        p_veg = np.loadtxt(p_vegetation)


def get_p_time_fn(ros_model_code):
    ros_models = {
        DEFAULT_TAG: p_time_standard,
        WANG_TAG: p_time_wang,
        ROTHERMEL_TAG: p_time_rothermel,
    }
    p_time_function = ros_models.get(ros_model_code, p_time_wang)
    return p_time_function


def get_p_moist_fn(moist_model_code):
    moist_models = {
        DEFAULT_TAG: moist_proba_correction_1,
        NEW_FORMULATION_TAG: moist_proba_correction_1,
        ROTHERMEL_TAG: moist_proba_correction_2,
    }
    p_moist_function = moist_models.get(moist_model_code, moist_proba_correction_1)
    return p_moist_function


def p_time_rothermel(
    dem_from, dem_to, veg_from, veg_to, angle_to, dist, moist, w_dir, w_speed
):
    # velocità di base modulata con la densità(tempo di attraversamento)
    dh = dem_to - dem_from

    v = v0[veg_from - 1] / 60  # tempo in minuti di attraversamento di una cella

    real_dist = np.sqrt((CELLSIZE * dist) ** 2 + dh**2)

    # wind component in propagation direction
    w_proj = np.cos(w_dir - angle_to)
    # wind speed in the direction of propagation
    w_spd = (w_speed * w_proj) / 3.6

    teta_s_rad = np.arctan(dh / CELLSIZE * dist)  # slope angle [rad]
    teta_s = np.degrees(teta_s_rad)  # slope angle [°]

    # flame angle measured from the vertical in the direction of fire spread [rad]
    teta_f_rad = np.arctan(0.4226 * w_spd)
    teta_f = np.degrees(teta_f_rad)  # flame angle [°]

    sf = np.exp(ROTHERMEL_ALPHA1 * teta_s)  # slope factor
    sf_clip = np.clip(sf, 0.01, 10)  # slope factor clipped at 10
    wf = np.exp(ROTHERMEL_ALPHA2 * teta_f)  # wind factor
    wf_rescaled = wf / 13  # wind factor rescaled to have 10 as max value
    wf_clip = np.clip(wf_rescaled, 1, 20)  # max value is 20, min is 1

    v_wh_pre = v * sf_clip * wf_clip  # Rate of Spread evaluate with Rothermel's model
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    # v_wh = np.clip(v_wh_pre, 0.01, 100) #adoptable RoS
    v_wh = np.clip(v_wh_pre * moist_eff, 0.01, 100)  # adoptable RoS [m/min]

    t = real_dist / v_wh
    t[t >= 1] = np.around(t[t >= 1])
    t = np.clip(t, 0.1, np.inf)
    return t, v_wh
    # return t


def p_time_wang(
    v0, dem_from, dem_to, veg_from, veg_to, angle_to, dist, moist, w_dir, w_speed
):
    # velocità di base modulata con la densità(tempo di attraversamento)
    dh = dem_to - dem_from

    v = v0[veg_from - 1] / 60  # tempo in minuti di attraversamento di una cella

    real_dist = np.sqrt((CELLSIZE * dist) ** 2 + dh**2)

    # wind component in propagation direction
    w_proj = np.cos(w_dir - angle_to)
    # wind speed in the direction of propagation
    w_spd = (w_speed * w_proj) / 3.6

    teta_s_rad = np.arctan(dh / CELLSIZE * dist)  # slope angle [rad]
    teta_s_pos = np.absolute(teta_s_rad)  # absolute values of slope angle
    # +1 if fire spreads upslope, -1 if fire spreads downslope
    p_reverse = np.sign(dh)

    wf = np.exp(WANG_BETA1 * w_spd)  # wind factor
    wf_clip = np.clip(wf, 0.01, 10)  # clipped at 10
    sf = np.exp(
        p_reverse * WANG_BETA2 * np.tan(teta_s_pos) ** WANG_BETA3
    )  # slope factor
    sf_clip = np.clip(sf, 0.01, 10)

    # Rate of Spread evaluate with Wang Zhengfei's model
    v_wh_pre = v * wf_clip * sf_clip
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    # v_wh = np.clip(v_wh_pre, 0.01, 100) #adoptable RoS
    v_wh = np.clip(v_wh_pre * moist_eff, 0.01, 100)  # adoptable RoS [m/min]

    t = real_dist / v_wh

    t[t >= 1] = np.around(t[t >= 1])
    t = np.clip(t, 0.1, np.inf)
    return t, v_wh


def p_time_standard(
    v0, dem_from, dem_to, veg_from, veg_to, angle_to, dist, moist, w_dir, w_speed
):
    dh = dem_to - dem_from
    v = v0[veg_from - 1] / 60
    wh = w_h_effect(angle_to, w_speed, w_dir, dh, dist)
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    v_wh = np.clip(v * wh * moist_eff, 0.01, 100)

    real_dist = np.sqrt((CELLSIZE * dist) ** 2 + dh**2)
    t = real_dist / v_wh
    t[t >= 1] = np.around(t[t >= 1])
    t = np.clip(t, 0.1, np.inf)
    return t, v_wh


def w_h_effect(angle_to, w_speed, w_dir, dh, dist):
    w_effect_module = A + (D1 * (D2 * np.tanh((w_speed / D3) - D4))) + (w_speed / D5)
    a = (w_effect_module - 1) / 4
    w_effect_on_direction = (
        (a + 1) * (1 - a**2) / (1 - a * np.cos(normalize(w_dir - angle_to)))
    )
    slope = dh / (CELLSIZE * dist)
    h_effect = 2 ** (np.tanh((slope * 3) ** 2.0 * np.sign(slope)))

    w_h = h_effect * w_effect_on_direction
    return w_h


def w_h_effect_on_probability(angle_to, w_speed, w_dir, dh, dist_to):
    """
    scales the wh factor for using it on the probability modulation
    """
    w_speed_norm = np.clip(w_speed, 0, 60)
    wh_orig = w_h_effect(angle_to, w_speed_norm, w_dir, dh, dist_to)
    wh = wh_orig - 1.0
    wh[wh > 0] = wh[wh > 0] / 2.13
    wh[wh < 0] = wh[wh < 0] / 1.12
    wh += 1.0
    return wh


def moist_proba_correction_1(moist):
    """
    e_m is the moinsture correction to the transition probability p_{i,j}.
    e_m = f(m), with m the Fine Fuel Moisture Content
    e_m = -11,507x5 + 22,963x4 - 17,331x3 + 6,598x2 - 1,7211x + 1,0003, where x is moisture / moisture of extintion (Mx).
    Mx = 0.3
    (reference: Trucchia et al, Fire 2020 )
    """
    Mx = 0.3
    x = np.clip(moist, 0.0, 1.0) / Mx
    p_moist = (
        (-11.507 * x**5)
        + (22.963 * x**4)
        + (-17.331 * x**3)
        + (6.598 * x**2)
        + (-1.7211 * x)
        + 1.0003
    )
    p_moist = np.clip(p_moist, 0.0, 1.0)
    return p_moist


def moist_proba_correction_2(moist):
    """
    e_m is the moinsture correction to the transition probability p_{i,j}. e_m = f(m), with m the Fine Fuel Moisture Content
    Old formulation by Baghino, adopted in Trucchia et al, Fire 2020.
    Here, the parameters come straight from constants.py.
    """
    p_moist = M1 * moist**3 + M2 * moist**2 + M3 * moist + M4
    return p_moist


def fire_spotting(angle_to, w_dir, w_speed):
    """this function evaluates the distance that an ember can reach, by the use of the Alexandridis' formulation"""
    r_n = np.random.normal(
        SPOTTING_RN_MEAN, SPOTTING_RN_STD, size=angle_to.shape
    )  # main thrust of the ember: sampled from a Gaussian Distribution (Alexandridis et al, 2008 and 2011)
    w_speed_ms = w_speed / 3.6  # wind speed [m/s]
    # Alexandridis' formulation for spotting distance
    d_p = r_n * np.exp(
        w_speed_ms * FIRE_SPOTTING_DISTANCE_COEFFICIENT * (np.cos(w_dir - angle_to) - 1)
    )
    return d_p


# functions useful for evaluating the fire line intensity


def lhv_dead_fuel(hhv, dffm):
    lhv = hhv * (1.0 - (dffm / 100.0)) - Q * (dffm / 100.0)
    return lhv


def lhv_canopy(hhv, hum):
    lhv = hhv * (1.0 - (hum / 100.0)) - Q * (hum / 100.0)
    lhv[np.isnan(lhv)] = 0
    return lhv


def fireline_intensity(d0, d1, ros, lhv_dead_fuel, lhv_canopy, rg=None):
    intensity = np.full(ros.shape[0], np.nan, dtype="float32")
    if rg is not None:
        rg_idx = ~np.isnan(rg)
        d1_idx = (not np.isclose(d1, 0.0, rtol=1e-09, atol=1e-09)) & rg_idx
        d0_idx = (np.isclose(d1, 0.0, rtol=1e-09, atol=1e-09)) & rg_idx
        intensity[d0_idx] = (
            ros[d0_idx]
            * ((lhv_dead_fuel[d0_idx] * d0[d0_idx] * (1.0 - rg[d0_idx])) / 2)
            / 60.0
        )
        intensity[d1_idx] = (
            ros[d1_idx]
            * (
                (
                    lhv_dead_fuel[d1_idx] * d0[d1_idx]
                    + lhv_canopy[d1_idx] * (d1[d1_idx] * (1 - rg[d1_idx]))
                )
                / 2
            )
            / 60.0
        )
        intensity[~rg_idx] = (
            ros[~rg_idx]
            * (
                (
                    lhv_dead_fuel[~rg_idx] * d0[~rg_idx]
                    + lhv_canopy[~rg_idx] * d1[~rg_idx]
                )
                / 2
            )
            / 60.0
        )
    else:
        # divided by 60 instead of 3600 because RoS is required in m/s and it is given in m/min (so it has to be divided by 60)
        intensity = ros * (lhv_dead_fuel * d0 + lhv_canopy * d1) / 60.0
    return intensity
