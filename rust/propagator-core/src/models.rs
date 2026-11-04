//! Spread physics: rate-of-spread / travel-time models, moisture
//! probability corrections, wind+slope factors, fireline intensity.
//!
//! Formula-for-formula port of `core/numba/functions.py` (see the spec
//! §5); all functions are pure and kernel-safe.

// Empirical wind/slope effect constants of the `Standard` model (`w_h_effect`).
const D1: f64 = 0.5;
const D2: f64 = 1.4;
const D3: f64 = 8.2;
const D4: f64 = 2.0;
const D5: f64 = 50.0;

// Rothermel ROS slope/wind exponents.
const ROTHERMEL_ALPHA1: f64 = 0.0693;
const ROTHERMEL_ALPHA2: f64 = 0.0576;

// Wang ROS wind/slope factor coefficients.
const WANG_BETA1: f64 = 0.1783;
const WANG_BETA2: f64 = 3.533;
const WANG_BETA3: f64 = 1.2;

// Cubic moisture-probability polynomial coefficients (Baghino model).
const M1: f64 = -3.5995;
const M2: f64 = 5.2389;
const M3: f64 = -2.6355;
const M4: f64 = 1.019;

// Exponential decay rate of ROS with fuel moisture (`exp(C_MOIST * moist)`).
const C_MOIST: f64 = -0.014;

// Latent heat of vaporization term (kJ/kg) subtracted per unit fuel moisture
// when computing the lower heating value.
const Q: f64 = 2442.0;

/// Fuel moisture (fraction) at which spread ceases; also normalizes the
/// Trucchia moisture polynomial.
pub const MOISTURE_OF_EXTINCTION: f64 = 0.3;

/// Rate-of-spread model selecting how wind and slope scale the base ROS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RosModel {
    /// Wang (default): exponential wind and slope factors.
    #[default]
    Wang,
    /// Rothermel: exponential slope and wind factors in degrees.
    Rothermel,
    /// Standard: the combined `w_h_effect` wind+slope scaling.
    Standard,
}

/// Model relating fuel moisture to the spread-probability correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MoistureModel {
    /// Trucchia (default): quintic polynomial normalized by extinction.
    #[default]
    Trucchia,
    /// Baghino: cubic polynomial in the moisture fraction.
    Baghino,
}

#[inline]
fn clip(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// numpy-style sign: 0.0 for 0.0 (Rust `signum` returns ±1 for ±0).
#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `A` evaluated at zero wind speed (see functions.py).
#[inline]
fn a_const() -> f64 {
    1.0 - D1 * D2 * (-D4).tanh()
}

/// Combined wind+slope scale factor on the rate of spread.
///
/// `angle`/`w_dir` in radians (0 = north->south), `w_speed` km/h,
/// `dh`/`dist` meters.
pub fn w_h_effect(angle: f64, w_speed: f64, w_dir: f64, dh: f64, dist: f64) -> f64 {
    let module = a_const() + D1 * (D2 * (w_speed / D3 - D4).tanh()) + w_speed / D5;
    let a = (module - 1.0) / 4.0;
    let w_effect_on_direction =
        (a + 1.0) * (1.0 - a * a) / (1.0 - a * (w_dir - angle).cos());
    let slope = dh / dist;
    let h_effect = 2f64.powf(((slope * 3.0).powi(2) * sign(slope)).tanh());
    h_effect * w_effect_on_direction
}

/// Wind+slope scale factor on the *probability*, derived from
/// `w_h_effect` by non-linear rescaling.
pub fn w_h_effect_on_probability(
    angle: f64,
    w_speed: f64,
    w_dir: f64,
    dh: f64,
    dist: f64,
) -> f64 {
    let w_speed_norm = clip(w_speed, 0.0, 60.0);
    let mut wh = w_h_effect(angle, w_speed_norm, w_dir, dh, dist) - 1.0;
    if wh > 0.0 {
        wh /= 2.13;
    } else if wh < 0.0 {
        wh /= 1.12;
    }
    wh + 1.0
}

/// (transition time [s], ROS [m/min]) for one cell-to-cell transition.
///
/// `v0` m/min, `dh` m, `angle` radians, `dist` m, `moist` fraction,
/// `w_dir` radians, `w_speed` km/h.
pub fn p_time(
    model: RosModel,
    v0: f64,
    dh: f64,
    angle: f64,
    dist: f64,
    moist: f64,
    w_dir: f64,
    w_speed: f64,
) -> (f64, f64) {
    let real_dist = (dist * dist + dh * dh).sqrt();
    let moist_eff = (C_MOIST * moist).exp();

    let ros = match model {
        RosModel::Wang => {
            let w_proj = (w_dir - angle).cos();
            let w_spd = w_speed * w_proj / 3.6;
            let teta_s = (dh / dist).atan();
            let teta_s_pos = teta_s.abs();
            let p_reverse = sign(dh);
            let wf = clip((WANG_BETA1 * w_spd).exp(), 0.01, 10.0);
            let sf = clip(
                (p_reverse * WANG_BETA2 * teta_s_pos.tan().powf(WANG_BETA3)).exp(),
                0.01,
                10.0,
            );
            clip(v0 * wf * sf * moist_eff, 0.01, 100.0)
        }
        RosModel::Rothermel => {
            let w_proj = (w_dir - angle).cos();
            let w_spd = w_speed * w_proj / 3.6;
            let teta_s = (dh / dist).atan().to_degrees();
            let teta_f = (0.4226 * w_spd).atan().to_degrees();
            let sf = clip((ROTHERMEL_ALPHA1 * teta_s).exp(), 0.01, 10.0);
            let wf = clip((ROTHERMEL_ALPHA2 * teta_f).exp() / 13.0, 1.0, 20.0);
            clip(v0 * sf * wf * moist_eff, 0.01, 100.0)
        }
        RosModel::Standard => {
            let wh = w_h_effect(angle, w_speed, w_dir, dh, dist);
            clip(v0 * wh * moist_eff, 0.01, 100.0)
        }
    };

    let time_seconds = real_dist / ros * 60.0;
    (time_seconds, ros)
}

/// Moisture correction factor on the transition probability.
/// `moist` is a fraction.
pub fn p_moisture(model: MoistureModel, moist: f64) -> f64 {
    match model {
        MoistureModel::Trucchia => {
            let x = moist / MOISTURE_OF_EXTINCTION;
            let p = -11.507 * x.powi(5)
                + 22.963 * x.powi(4)
                - 17.331 * x.powi(3)
                + 6.598 * x.powi(2)
                - 1.7211 * x
                + 1.0003;
            clip(p, 0.0, 1.0)
        }
        MoistureModel::Baghino => {
            M1 * moist.powi(3) + M2 * moist.powi(2) + M3 * moist + M4
        }
    }
}

/// Lower heating value from HHV and moisture (both fuel-specific).
#[inline]
pub fn lhv_fuel(hhv: f64, moisture: f64) -> f64 {
    hhv * (1.0 - moisture) - Q * moisture
}

/// Fireline intensity (kW/m) from fuel loads and rate of spread.
#[inline]
pub fn fireline_intensity(
    d0: f64,
    d1: f64,
    ros: f64,
    lhv_dead_fuel: f64,
    lhv_canopy: f64,
) -> f64 {
    (ros / 60.0) * (lhv_dead_fuel * d0 + lhv_canopy * d1)
}

/// Probability of fire spread to a neighbouring cell.
pub fn probability_to_neighbour(
    moisture_model: MoistureModel,
    angle: f64,
    dist: f64,
    w_dir: f64,
    w_speed: f64,
    moisture: f64,
    dh: f64,
    transition_probability: f64,
) -> f64 {
    let moisture_effect = p_moisture(moisture_model, moisture);
    let alpha_wh = w_h_effect_on_probability(angle, w_speed, w_dir, dh, dist).max(0.0);
    let p = 1.0 - (1.0 - transition_probability).powf(alpha_wh);
    clip(p * moisture_effect, 0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moisture_trucchia_bounds() {
        // dry fuel -> ~1 (clipped)
        assert!((p_moisture(MoistureModel::Trucchia, 0.0) - 1.0).abs() < 1e-3);
        // at extinction moisture -> ~0
        assert!(p_moisture(MoistureModel::Trucchia, 0.3) < 0.05);
        // clipped to [0, 1]
        for i in 0..=30 {
            let p = p_moisture(MoistureModel::Trucchia, i as f64 / 100.0);
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn ros_clipped_and_time_positive() {
        for model in [RosModel::Wang, RosModel::Rothermel, RosModel::Standard] {
            let (t, ros) = p_time(model, 140.0 / 60.0, 5.0, 0.0, 20.0, 0.1, 0.5, 30.0);
            assert!(ros >= 0.01 && ros <= 100.0, "{model:?} ros {ros}");
            assert!(t > 0.0, "{model:?} time {t}");
        }
    }

    #[test]
    fn wind_alignment_speeds_up_spread() {
        // propagation aligned with the wind must be faster than against it
        let angle = 0.0;
        let (t_with, _) =
            p_time(RosModel::Wang, 2.0, 0.0, angle, 20.0, 0.1, angle, 30.0);
        let (t_against, _) = p_time(
            RosModel::Wang,
            2.0,
            0.0,
            angle,
            20.0,
            0.1,
            angle + std::f64::consts::PI,
            30.0,
        );
        assert!(t_with < t_against);
    }

    #[test]
    fn upslope_faster_than_downslope() {
        let (t_up, _) = p_time(RosModel::Wang, 2.0, 10.0, 0.0, 20.0, 0.1, 0.0, 0.0);
        let (t_down, _) = p_time(RosModel::Wang, 2.0, -10.0, 0.0, 20.0, 0.1, 0.0, 0.0);
        assert!(t_up < t_down);
    }

    #[test]
    fn probability_monotone_in_base_probability() {
        let p_low = probability_to_neighbour(
            MoistureModel::Trucchia,
            0.0,
            20.0,
            0.0,
            10.0,
            0.1,
            0.0,
            0.1,
        );
        let p_high = probability_to_neighbour(
            MoistureModel::Trucchia,
            0.0,
            20.0,
            0.0,
            10.0,
            0.1,
            0.0,
            0.5,
        );
        assert!(p_high > p_low);
        assert!((0.0..=1.0).contains(&p_low));
        assert!((0.0..=1.0).contains(&p_high));
    }
}
