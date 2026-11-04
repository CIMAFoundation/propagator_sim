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
    gpu.set_scalar_weather(0, 90.0, 30.0, 0.0, &[(c, c)])
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
        gpu.set_scalar_weather(0, 45.0, 20.0, 5.0, &[(64, 64)])
            .unwrap();
        gpu.advance_to(3 * 3600).unwrap();
        maps.push(gpu.get_output().unwrap().fire_probability);
    }
    for m in &maps[1..] {
        assert_eq!(m.as_slice(), maps[0].as_slice());
    }
}

// ---- feature parity with the CPU core -------------------------------------

use propagator_core::{legacy_fuel_defs, FuelSystem};
use propagator_gpu::{Field, GpuBoundaryConditions, GpuCheckpoint};

struct Scenario {
    n: usize,
    reals: usize,
    hours: i64,
    veg: Grid2<i32>,
    dem: Grid2<f64>,
    fuels: FuelSystem,
    do_spotting: bool,
    wind_dir: FieldInput,
    wind_speed: FieldInput,
    moisture: FieldInput,
    additional_moisture: Option<Grid2<f32>>,
    vegetation_changes: Option<Grid2<f64>>,
}

fn to_field(f: &FieldInput) -> Field {
    match f {
        FieldInput::Scalar(v) => Field::Scalar(*v),
        FieldInput::Grid(g) => Field::Grid(g.clone()),
    }
}

/// CPU core run, hourly windows, outputs at the end.
fn run_cpu(s: &Scenario, seed: u64) -> propagator_core::PropagatorOutput {
    let c = s.n / 2;
    let mut config = PropagatorConfig::new(s.veg.clone(), s.dem.clone());
    config.realizations = s.reals;
    config.seed = Some(seed);
    config.oob_mode = OobMode::Ignore;
    config.fuels = Some(s.fuels.clone());
    config.do_spotting = s.do_spotting;
    let mut cpu = Propagator::new(config).unwrap();
    cpu.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(s.moisture.clone()),
        wind_dir: Some(s.wind_dir.clone()),
        wind_speed: Some(s.wind_speed.clone()),
        ignitions: Some(Ignitions::Points(vec![(c, c)])),
        additional_moisture: s.additional_moisture.clone(),
        vegetation_changes: s.vegetation_changes.clone(),
    })
    .unwrap();
    for _ in 0..s.hours {
        cpu.step_window(3600).unwrap();
    }
    cpu.get_output().unwrap()
}

fn gpu_config(s: &Scenario, seed: u32) -> GpuConfig {
    let mut g = GpuConfig::new(s.veg.clone(), s.dem.clone(), s.fuels.clone());
    g.realizations = s.reals;
    g.seed = seed;
    g.do_spotting = s.do_spotting;
    g
}

fn gpu_bc(s: &Scenario) -> GpuBoundaryConditions {
    let c = s.n / 2;
    GpuBoundaryConditions {
        time: 0,
        moisture: Some(to_field(&s.moisture)),
        wind_dir: Some(to_field(&s.wind_dir)),
        wind_speed: Some(to_field(&s.wind_speed)),
        ignitions: vec![(c, c)],
        additional_moisture: s.additional_moisture.clone(),
        vegetation_changes: s.vegetation_changes.clone(),
    }
}

/// GPU run, hourly windows; `None` without a GPU adapter.
fn run_gpu(s: &Scenario, seed: u32) -> Option<propagator_gpu::GpuOutput> {
    let mut gpu = match GpuPropagator::new(gpu_config(s, seed)) {
        Ok(gpu) => gpu,
        Err(GpuError::NoAdapter(_)) => return None,
        Err(e) => panic!("gpu init: {e}"),
    };
    gpu.set_boundary_conditions(gpu_bc(s)).unwrap();
    for h in 1..=s.hours {
        gpu.advance_to(h * 3600).unwrap();
    }
    Some(gpu.get_output().unwrap())
}

fn grassland(n: usize) -> Scenario {
    Scenario {
        n,
        reals: 100,
        hours: 6,
        veg: Grid2::filled(n, n, 4),
        dem: Grid2::filled(n, n, 0.0),
        fuels: legacy_fuel_system(),
        do_spotting: false,
        wind_dir: FieldInput::Scalar(90.0),
        wind_speed: FieldInput::Scalar(30.0),
        moisture: FieldInput::Scalar(0.0),
        additional_moisture: None,
        vegetation_changes: None,
    }
}

fn check(
    name: &str,
    cpu: &propagator_core::PropagatorOutput,
    gpu: &propagator_gpu::GpuOutput,
    area_tol: f64,
    min_corr: f64,
) {
    let ratio = gpu.area_mean / cpu.stats.area_mean;
    let corr = correlation(
        cpu.fire_probability.as_slice(),
        gpu.fire_probability.as_slice(),
    );
    eprintln!(
        "{name}: cpu {:.1} ha, gpu {:.1} ha, ratio {ratio:.4}, corr {corr:.4}",
        cpu.stats.area_mean / 1e4,
        gpu.area_mean / 1e4
    );
    assert!((ratio - 1.0).abs() < area_tol, "{name}: area ratio {ratio}");
    assert!(corr > min_corr, "{name}: correlation {corr}");
}

#[test]
fn spotting_matches_cpu_statistics() {
    let spotter = legacy_fuel_defs()
        .iter()
        .find(|f| f.spotting)
        .expect("a spotting fuel")
        .id;
    let mut s = grassland(256);
    s.veg = Grid2::filled(256, 256, spotter);
    s.do_spotting = true;
    let Some(gpu) = run_gpu(&s, 11) else { return };
    let cpu = run_cpu(&s, 11);
    check("spotting", &cpu, &gpu, 0.10, 0.97);
    let spots: f32 = gpu.spotting_receiving_probability.as_slice().iter().sum();
    assert!(spots > 0.0, "no cell was ignited by embers");
    // the same scenario without spotting burns less
    s.do_spotting = false;
    let plain = run_gpu(&s, 11).unwrap();
    assert!(gpu.area_mean > plain.area_mean, "spotting did not add area");
}

#[test]
fn per_cell_weather_matches_cpu() {
    let n = 256;
    let mut s = grassland(n);
    let mut wd = Grid2::filled(n, n, 0.0f32);
    let mut ws = Grid2::filled(n, n, 0.0f32);
    let mut mo = Grid2::filled(n, n, 0.0f32);
    for row in 0..n {
        for col in 0..n {
            wd[(row, col)] = 60.0 + 60.0 * col as f32 / n as f32;
            ws[(row, col)] = 20.0 + 20.0 * row as f32 / n as f32;
            mo[(row, col)] = 8.0 * col as f32 / n as f32;
        }
    }
    s.wind_dir = FieldInput::Grid(wd);
    s.wind_speed = FieldInput::Grid(ws);
    s.moisture = FieldInput::Grid(mo);
    let Some(gpu) = run_gpu(&s, 5) else { return };
    let cpu = run_cpu(&s, 5);
    check("fields", &cpu, &gpu, 0.05, 0.99);
}

#[test]
fn decaying_action_moisture_matches_cpu() {
    let n = 256;
    let mut s = grassland(n);
    // a wet strip downwind (wind from the east pushes the fire west)
    let mut extra = Grid2::filled(n, n, 0.0f32);
    for row in 96..160 {
        for col in 80..96 {
            extra[(row, col)] = 0.25;
        }
    }
    s.additional_moisture = Some(extra);
    let Some(gpu) = run_gpu(&s, 3) else { return };
    let cpu = run_cpu(&s, 3);
    check("actions", &cpu, &gpu, 0.05, 0.98);
    let plain = run_gpu(&grassland(n), 3).unwrap();
    assert!(
        gpu.area_mean < plain.area_mean,
        "the action did not reduce the area"
    );
}

#[test]
fn vegetation_change_matches_cpu() {
    let n = 256;
    let mut s = grassland(n);
    // a firebreak: three columns of no fuel downwind of the ignition
    let mut changes = Grid2::filled(n, n, f64::NAN);
    for row in 0..n {
        for col in 100..103 {
            changes[(row, col)] = 0.0;
        }
    }
    s.vegetation_changes = Some(changes);
    let Some(gpu) = run_gpu(&s, 9) else { return };
    let cpu = run_cpu(&s, 9);
    check("firebreak", &cpu, &gpu, 0.05, 0.99);
    // nothing west of the break burns
    let west: f32 = (0..n).map(|row| gpu.fire_probability[(row, 50)]).sum();
    assert_eq!(west, 0.0);
}

#[test]
fn checkpoint_restore_is_exact() {
    let s = grassland(128);
    let Some(reference) = run_gpu(&s, 21) else {
        return;
    };
    let mut gpu = GpuPropagator::new(gpu_config(&s, 21)).unwrap();
    gpu.set_boundary_conditions(gpu_bc(&s)).unwrap();
    for h in 1..=3 {
        gpu.advance_to(h * 3600).unwrap();
    }
    let cp: GpuCheckpoint = gpu.checkpoint().unwrap();
    assert_eq!(cp.time, 3 * 3600);
    let mut resumed = GpuPropagator::from_checkpoint(gpu_config(&s, 21), &cp).unwrap();
    assert_eq!(resumed.time(), 3 * 3600);
    for h in 4..=6 {
        resumed.advance_to(h * 3600).unwrap();
    }
    let out = resumed.get_output().unwrap();
    assert_eq!(
        out.fire_probability.as_slice(),
        reference.fire_probability.as_slice()
    );
}

#[test]
fn growing_the_domain_keeps_the_draws() {
    // small fire on 128^2, grown onto 192^2 with the old grid placed at
    // (32, 32); the same scenario run on the large grid from the start must
    // give the same map because draws are hashed by world position
    let mut small = grassland(128);
    small.wind_speed = FieldInput::Scalar(15.0);
    small.hours = 3;
    let mut gpu = match GpuPropagator::new(gpu_config(&small, 8)) {
        Ok(gpu) => gpu,
        Err(GpuError::NoAdapter(_)) => return,
        Err(e) => panic!("gpu init: {e}"),
    };
    gpu.set_boundary_conditions(gpu_bc(&small)).unwrap();
    gpu.advance_to(3 * 3600).unwrap();
    let cp = gpu.checkpoint().unwrap();

    let mut large = grassland(192);
    large.wind_speed = FieldInput::Scalar(15.0);
    let mut cfg = gpu_config(&large, 8);
    cfg.origin = (-32, -32);
    let mut grown = GpuPropagator::from_checkpoint(cfg, &cp).unwrap();
    for h in 4..=6 {
        grown.advance_to(h * 3600).unwrap();
    }
    let grown_out = grown.get_output().unwrap();

    let mut cfg = gpu_config(&large, 8);
    cfg.origin = (-32, -32);
    let mut whole = GpuPropagator::new(cfg).unwrap();
    whole
        .set_boundary_conditions(GpuBoundaryConditions {
            ignitions: vec![(96, 96)],
            ..gpu_bc(&large)
        })
        .unwrap();
    for h in 1..=6 {
        whole.advance_to(h * 3600).unwrap();
    }
    let whole_out = whole.get_output().unwrap();
    assert_eq!(
        grown_out.fire_probability.as_slice(),
        whole_out.fire_probability.as_slice()
    );
    assert!(grown_out.area_mean > 0.0);
}
