//! Per-realization random number generation.
//!
//! Unlike the Python/numba implementation (per-thread RNGs, reproducible
//! only for a fixed thread count), each realization owns its own
//! xoshiro256++ stream seeded from `(base_seed, realization)`. Runs with
//! the same seed are bitwise reproducible regardless of thread count.

#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
    /// cached second output of Box-Muller
    spare_normal: Option<f64>,
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Deterministic stream for one realization of a seeded run.
    pub fn for_realization(seed: u64, realization: u64) -> Self {
        let mut sm = seed ^ realization.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut state = [0u64; 4];
        for slot in &mut state {
            *slot = splitmix64(&mut sm);
        }
        // avoid the all-zero state (xoshiro fixed point)
        if state == [0, 0, 0, 0] {
            state[0] = 1;
        }
        Rng {
            state,
            spare_normal: None,
        }
    }

    /// Non-deterministic stream (unseeded runs).
    pub fn from_entropy(realization: u64) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let addr = &realization as *const u64 as u64;
        Rng::for_realization(nanos ^ addr.rotate_left(17), realization)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // xoshiro256++
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [low, high).
    #[inline]
    pub fn uniform_range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// Standard normal via Box-Muller (with spare caching).
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return mean + std * z;
        }
        // u1 in (0, 1] to keep ln() finite
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare_normal = Some(radius * theta.sin());
        mean + std * radius * theta.cos()
    }

    /// Lognormal sample: `exp(normal(log_mean, log_std))`. Parameters are
    /// the mean and std of the underlying normal (i.e. in log-space).
    pub fn lognormal(&mut self, log_mean: f64, log_std: f64) -> f64 {
        self.normal(log_mean, log_std).exp()
    }

    /// Poisson sample (Knuth's algorithm; fine for the small lambdas the
    /// spotting model uses).
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        let limit = (-lambda).exp();
        let mut k = 0u32;
        let mut product = 1.0;
        loop {
            product *= self.uniform();
            if product <= limit {
                return k;
            }
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_per_realization() {
        let mut a = Rng::for_realization(42, 3);
        let mut b = Rng::for_realization(42, 3);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = Rng::for_realization(42, 4);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn poisson_mean_close_to_lambda() {
        let mut rng = Rng::for_realization(7, 0);
        let n = 20_000;
        let total: u64 = (0..n).map(|_| rng.poisson(2.0) as u64).sum();
        let mean = total as f64 / n as f64;
        assert!((mean - 2.0).abs() < 0.05, "poisson mean {mean}");
    }

    #[test]
    fn normal_moments() {
        let mut rng = Rng::for_realization(9, 0);
        let n = 20_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.normal(100.0, 25.0)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((mean - 100.0).abs() < 1.0, "normal mean {mean}");
        assert!((var.sqrt() - 25.0).abs() < 1.0, "normal std {}", var.sqrt());
    }
}
