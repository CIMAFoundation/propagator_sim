//! The GPU backend must reproduce the CPU core's ensemble statistics on the
//! homogeneous-grassland scenario: same burned area within a few percent
//! and a near-identical burn-probability map. Skipped when no GPU adapter
//! is available (CI without a GPU).

use propagator_core::{
    legacy_fuel_system, BoundaryConditions, FieldInput, Grid2, Ignitions, OobMode, Propagator,
    PropagatorConfig,
};
use propagator_gpu::{GpuConfig, GpuError, GpuPropagator};

const N: usize = 256;
const REALS: usize = 50;
const HOURS: i64 = 6;

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let m = a.len() as f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    (sab - sa * sb / m) / ((saa - sa * sa / m).sqrt() * (sbb - sb * sb / m).sqrt())
}

#[test]
fn gpu_matches_cpu_ensemble_statistics() {
    let veg = Grid2::filled(N, N, 4);
    let dem = Grid2::filled(N, N, 0.0);
    let c = N / 2;

    let mut gcfg = GpuConfig::new(veg.clone(), dem.clone(), legacy_fuel_system());
    gcfg.realizations = REALS;
    gcfg.seed = 7;
    let mut gpu = match GpuPropagator::new(gcfg) {
        Ok(gpu) => gpu,
        Err(GpuError::NoAdapter(msg)) => {
            eprintln!("skipping: no GPU adapter ({msg})");
            return;
        }
        Err(e) => panic!("gpu init: {e}"),
    };
    gpu.set_boundary_conditions(0, 90.0, 30.0, 0.0, &[(c, c)])
        .unwrap();
    for h in 1..=HOURS {
        gpu.advance_to(h * 3600).unwrap();
    }
    let gpu_out = gpu.get_output().unwrap();

    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = REALS;
    config.seed = Some(7);
    config.oob_mode = OobMode::Ignore;
    let mut cpu = Propagator::new(config).unwrap();
    cpu.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(0.0)),
        wind_dir: Some(FieldInput::Scalar(90.0)),
        wind_speed: Some(FieldInput::Scalar(30.0)),
        ignitions: Some(Ignitions::Points(vec![(c, c)])),
        ..Default::default()
    })
    .unwrap();
    cpu.step_window(HOURS * 3600).unwrap();
    let cpu_out = cpu.get_output().unwrap();

    let ratio = gpu_out.area_mean / cpu_out.stats.area_mean;
    assert!(
        (0.95..1.05).contains(&ratio),
        "area ratio gpu/cpu = {ratio}"
    );
    let corr = correlation(
        cpu_out.fire_probability.as_slice(),
        gpu_out.fire_probability.as_slice(),
    );
    assert!(corr > 0.99, "fire-probability map correlation = {corr}");
    // the ignition burns at t = 0 in every realization
    assert_eq!(gpu_out.fire_probability[(c, c)], 1.0);
    assert_eq!(gpu_out.min_arrival_time[(c, c)], 0.0);
    // the fire drifts downwind (wind from the east pushes it west)
    let a = gpu_out.fire_probability.as_slice();
    let (mut sc, mut sw) = (0.0f64, 0.0f64);
    for (i, &v) in a.iter().enumerate() {
        sc += (i % N) as f64 * v as f64;
        sw += v as f64;
    }
    assert!(sc / sw < c as f64 - 10.0, "centroid column {}", sc / sw);
}

#[test]
fn bucket_width_does_not_change_the_result() {
    let veg = Grid2::filled(128, 128, 4);
    let dem = Grid2::filled(128, 128, 0.0);
    let mut maps = Vec::new();
    for (bucket, sweeps) in [(60u32, 2u32), (900, 6), (3600, 12)] {
        let mut gcfg = GpuConfig::new(veg.clone(), dem.clone(), legacy_fuel_system());
        gcfg.realizations = 8;
        gcfg.seed = 3;
        gcfg.bucket_seconds = bucket;
        gcfg.sweeps_per_bucket = sweeps;
        let mut gpu = match GpuPropagator::new(gcfg) {
            Ok(gpu) => gpu,
            Err(GpuError::NoAdapter(_)) => return,
            Err(e) => panic!("gpu init: {e}"),
        };
        gpu.set_boundary_conditions(0, 45.0, 20.0, 5.0, &[(64, 64)])
            .unwrap();
        gpu.advance_to(3 * 3600).unwrap();
        maps.push(gpu.get_output().unwrap().fire_probability);
    }
    for m in &maps[1..] {
        assert_eq!(m.as_slice(), maps[0].as_slice());
    }
}
