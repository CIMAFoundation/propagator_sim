//! Standalone benchmark / end-to-end driver for the Rust core.
//!
//! Runs a homogeneous-grassland point-ignition scenario matching the
//! Python benchmark harness so timings and burned-area aggregates can be
//! compared directly. Usage:
//!
//!   bench <grid> <realizations> <hours> [seed] [threads] [dump.f32]
//!
//! Prints a human summary to stderr and one tab-separated machine line to
//! stdout: `RUST\t<grid>\t<reals>\t<sim_s>\t<wall_s>\t<area_mean_ha>\t<area_50_ha>`.
//! When a dump path is given, writes the fire-probability grid as raw
//! row-major little-endian f32 for cross-core comparison.

use std::time::Instant;

use propagator_core::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: &str| args.get(i).map(String::as_str).unwrap_or(d).to_string();

    let n: usize = arg(1, "1000").parse().expect("grid size");
    let reals: usize = arg(2, "100").parse().expect("realizations");
    let hours: i64 = arg(3, "12").parse().expect("hours");
    let seed: u64 = arg(4, "12345").parse().expect("seed");
    let threads: Option<usize> = args.get(5).and_then(|s| s.parse().ok()).filter(|&t| t > 0);
    let dump = args.get(6).cloned();

    let veg = Grid2::filled(n, n, 4); // grassland (fuel code 4)
    let dem = Grid2::filled(n, n, 0.0);
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = reals;
    config.seed = Some(seed);
    config.oob_mode = OobMode::Ignore;
    config.n_threads = threads;

    let mut sim = Propagator::new(config).expect("construct propagator");
    let c = n / 2;
    sim.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(0.0)),
        wind_dir: Some(FieldInput::Scalar(90.0)),
        wind_speed: Some(FieldInput::Scalar(30.0)),
        ignitions: Some(Ignitions::Points(vec![(c, c)])),
        ..Default::default()
    })
    .expect("set boundary conditions");

    let target = hours * 3600;
    let t0 = Instant::now();
    let mut steps = 0u32;
    while sim.time() < target {
        if sim.next_time().is_none() {
            break;
        }
        sim.step_window(3600).expect("step");
        steps += 1;
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let output = sim.get_output().expect("output");
    let s = &output.stats;
    let area_mean_ha = s.area_mean / 1e4;
    let area_50_ha = s.area_50 / 1e4;

    eprintln!(
        "RUST  grid={n}x{n} reals={reals} threads={} steps={steps} sim_time={}s",
        threads
            .map(|t| t.to_string())
            .unwrap_or_else(|| "auto".into()),
        sim.time()
    );
    eprintln!(
        "      wall={elapsed:.3}s  area_mean={area_mean_ha:.1}ha  area_50={area_50_ha:.1}ha  n_active={}",
        s.n_active
    );
    println!(
        "RUST\t{n}\t{reals}\t{}\t{elapsed:.4}\t{area_mean_ha:.2}\t{area_50_ha:.2}",
        sim.time()
    );

    if let Some(path) = dump {
        let fp = output.fire_probability.as_slice();
        let mut bytes = Vec::with_capacity(fp.len() * 4);
        for v in fp {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, bytes).expect("write dump");
        eprintln!("      dumped fire_probability [{n}x{n} f32] -> {path}");
    }
}
