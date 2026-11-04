//! Standalone benchmark / end-to-end driver for the Rust core.
//!
//! Runs a homogeneous-grassland point-ignition scenario matching the
//! Python benchmark harness so timings and burned-area aggregates can be
//! compared directly. Usage:
//!
//!   bench <grid> <realizations> <hours> [seed] [threads] [dump.f32] [flags]
//!
//! Flags (after the positional arguments):
//!   --step <s>       step window in seconds (default 3600)
//!   --field-weather  per-cell wind direction / speed grids instead of
//!                    scalars (a smooth gradient across the domain)
//!   --actions        a decaying action-moisture patch next to the
//!                    ignition, so the effective moisture (and the edge
//!                    cache) changes every step window
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
    let dump = args.get(6).cloned().filter(|s| !s.starts_with("--") && !s.is_empty());

    let mut step_seconds: i64 = 3600;
    let mut field_weather = false;
    let mut actions = false;
    let mut i = 6;
    while i < args.len() {
        match args[i].as_str() {
            "--step" => {
                i += 1;
                step_seconds = args.get(i).and_then(|s| s.parse().ok()).expect("--step <s>");
            }
            "--field-weather" => field_weather = true,
            "--actions" => actions = true,
            other if other.starts_with("--") => panic!("unknown flag {other}"),
            _ => {}
        }
        i += 1;
    }

    let veg = Grid2::filled(n, n, 4); // grassland (fuel code 4)
    let dem = Grid2::filled(n, n, 0.0);
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = reals;
    config.seed = Some(seed);
    config.oob_mode = OobMode::Ignore;
    config.n_threads = threads;

    let mut sim = Propagator::new(config).expect("construct propagator");
    let c = n / 2;
    let (wind_dir, wind_speed) = if field_weather {
        // direction swings 60..120 deg west->east, speed 20..40 km/h
        // north->south: every cell gets its own edge quantities
        let mut dir = Grid2::filled(n, n, 0.0f32);
        let mut speed = Grid2::filled(n, n, 0.0f32);
        for row in 0..n {
            for col in 0..n {
                dir[(row, col)] = 60.0 + 60.0 * col as f32 / n as f32;
                speed[(row, col)] = 20.0 + 20.0 * row as f32 / n as f32;
            }
        }
        (FieldInput::Grid(dir), FieldInput::Grid(speed))
    } else {
        (FieldInput::Scalar(90.0), FieldInput::Scalar(30.0))
    };
    let additional_moisture = actions.then(|| {
        // a wet strip downwind of the ignition (+30 % moisture), decaying;
        // wind from 90 deg (east) drives the fire west, to lower columns
        let mut grid = Grid2::filled(n, n, 0.0f32);
        let half = (n / 20).max(2);
        for row in c.saturating_sub(half)..(c + half).min(n) {
            for col in c.saturating_sub(3 * half)..c.saturating_sub(2 * half) {
                grid[(row, col)] = 30.0;
            }
        }
        grid
    });
    sim.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(0.0)),
        wind_dir: Some(wind_dir),
        wind_speed: Some(wind_speed),
        ignitions: Some(Ignitions::Points(vec![(c, c)])),
        additional_moisture,
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
        sim.step_window(step_seconds).expect("step");
        steps += 1;
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let t_out = Instant::now();
    let output = sim.get_output().expect("output");
    let output_ms = t_out.elapsed().as_secs_f64() * 1e3;
    let s = &output.stats;
    let area_mean_ha = s.area_mean / 1e4;
    let area_50_ha = s.area_50 / 1e4;

    eprintln!(
        "RUST  grid={n}x{n} reals={reals} threads={} steps={steps} sim_time={}s step={step_seconds}s field_weather={field_weather} actions={actions}",
        threads
            .map(|t| t.to_string())
            .unwrap_or_else(|| "auto".into()),
        sim.time()
    );
    eprintln!(
        "      wall={elapsed:.3}s  get_output={output_ms:.1}ms  area_mean={area_mean_ha:.1}ha  area_50={area_50_ha:.1}ha  n_active={}  edge_cache={:.1}MiB/{} tiles",
        s.n_active,
        sim.edge_cache_bytes() as f64 / (1024.0 * 1024.0),
        sim.edge_cache_tiles()
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
