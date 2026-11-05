//! Benchmark of tile freezing and ignition-triggered thawing.
//!
//! Runs a grassland point-ignition scenario with scattered non-fuel
//! patches, calling `freeze_inactive_tiles` after every step window, then
//! schedules a batch of new ignitions (inside the burnt scar and in
//! unburnt territory) to time the thaw path. Usage:
//!
//!   freeze_bench <grid> <realizations> <hours> [threads] [--spotting]
//!                [--fuel <code>] [--fuzzy <threshold>] [--margin <tiles>]
//!                [--no-freeze]
//!
//! `--fuel` picks the vegetation code (default 4, grassland; 5 spots),
//! `--fuzzy` enables fuzzy freezing (margin default 1), `--no-freeze`
//! skips freezing entirely (kernel-only timing). Prints per-window step
//! and freeze timings, live tile counts and freeze stats, then the thaw
//! timing and the total wall time.

use std::time::Instant;

use propagator_core::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    const VALUE_FLAGS: [&str; 3] = ["--fuel", "--fuzzy", "--margin"];
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(i, a)| !a.starts_with("--") && !VALUE_FLAGS.contains(&args[i - 1].as_str()))
        .map(|(_, a)| a)
        .collect();
    let arg = |i: usize, d: &str| {
        positional
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or(d)
            .to_string()
    };
    let n: usize = arg(0, "2048").parse().expect("grid size");
    let reals: usize = arg(1, "100").parse().expect("realizations");
    let hours: i64 = arg(2, "8").parse().expect("hours");
    let threads: Option<usize> = arg(3, "0").parse().ok().filter(|&t| t > 0);
    let spotting = args.iter().any(|a| a == "--spotting");
    let no_freeze = args.iter().any(|a| a == "--no-freeze");
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let fuel_code: i32 = flag("--fuel").map(|v| v.parse().expect("--fuel")).unwrap_or(4);
    let fuzzy = flag("--fuzzy").map(|v| FuzzyFreeze {
        min_inactive_fraction: v.parse().expect("--fuzzy"),
        event_margin_tiles: flag("--margin")
            .map(|m| m.parse().expect("--margin"))
            .unwrap_or(1),
    });
    let wall0 = Instant::now();

    // grassland with deterministic scattered non-fuel blocks
    let mut veg = Grid2::filled(n, n, fuel_code);
    for row in 0..n {
        for col in 0..n {
            let h = (row / 7)
                .wrapping_mul(2_654_435_761)
                .wrapping_add((col / 7).wrapping_mul(40_503))
                % 97;
            if h < 6 {
                veg[(row, col)] = NO_FUEL;
            }
        }
    }
    let dem = Grid2::filled(n, n, 0.0);
    let freeze_dir = std::env::temp_dir()
        .join(format!("prop-freeze-bench-{}", std::process::id()));
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = reals;
    config.seed = Some(4242);
    config.oob_mode = OobMode::Ignore;
    config.n_threads = threads;
    config.do_spotting = spotting;
    if !no_freeze {
        config.freeze_dir = Some(freeze_dir.clone());
        config.fuzzy_freeze = fuzzy;
    }
    let mut sim = Propagator::new(config).expect("construct propagator");

    let c = n / 2;
    sim.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(2.0)),
        wind_dir: Some(FieldInput::Scalar(90.0)),
        wind_speed: Some(FieldInput::Scalar(40.0)),
        ignitions: Some(Ignitions::Points(vec![(c, c)])),
        ..Default::default()
    })
    .expect("set boundary conditions");

    let mut freeze_wall = 0.0f64;
    let mut step_wall = 0.0f64;
    let mut frozen_total = 0usize;
    let mut peak_live = 0usize;
    for hour in 0..hours {
        let t0 = Instant::now();
        sim.step_window(3600).expect("step");
        let step_s = t0.elapsed().as_secs_f64();
        step_wall += step_s;
        if no_freeze {
            let live = sim.freeze_diagnostic().allocated;
            peak_live = peak_live.max(live);
            eprintln!("hour {:>2}: step {step_s:.3}s  live {live}", hour + 1);
            continue;
        }
        let t1 = Instant::now();
        let frozen = sim.freeze_inactive_tiles().expect("freeze");
        let freeze_s = t1.elapsed().as_secs_f64();
        freeze_wall += freeze_s;
        frozen_total += frozen;
        let live = sim.freeze_diagnostic().allocated;
        peak_live = peak_live.max(live);
        let stats = sim.freeze_stats();
        eprintln!(
            "hour {:>2}: step {step_s:.3}s  freeze {freeze_s:.4}s  frozen +{frozen} (total {frozen_total})  live {live}  dead {}  cold {}  fault-ins {}",
            hour + 1,
            stats.dead_tiles,
            stats.cold_tiles,
            stats.fault_ins
        );
    }
    eprintln!("kernel (step) total {step_wall:.3}s; peak live tiles {peak_live}");
    if no_freeze {
        eprintln!("total wall {:.3}s", wall0.elapsed().as_secs_f64());
        return;
    }

    // a batch of ignitions: inside the burnt scar (frozen) and far away
    // in unburnt territory, every realization
    let points = vec![(c, c - n / 16), (c + 2, c - n / 16 + 3), (n / 8, n / 8), (7 * n / 8, 7 * n / 8)];
    let time = sim.time();
    sim.set_boundary_conditions(BoundaryConditions {
        time,
        ignitions: Some(Ignitions::Points(points)),
        ..Default::default()
    })
    .expect("ignitions");
    eprintln!("before ignitions: {:?}", sim.freeze_diagnostic());
    let frozen_before = sim.frozen_tiles();
    let t2 = Instant::now();
    sim.step_window(0).expect("thaw step");
    let thaw_s = t2.elapsed().as_secs_f64();
    eprintln!(
        "after ignitions: {:?} (thawed {} of {frozen_before} frozen tiles)",
        sim.freeze_diagnostic(),
        frozen_before - sim.frozen_tiles()
    );
    let t3 = Instant::now();
    let refrozen = sim.freeze_inactive_tiles().expect("freeze");
    let refreeze_s = t3.elapsed().as_secs_f64();
    let output = sim.get_output().expect("output");

    eprintln!(
        "grid={n} reals={reals} hours={hours} spotting={spotting} threads={}",
        threads.map(|t| t.to_string()).unwrap_or_else(|| "auto".into())
    );
    eprintln!(
        "freeze total {freeze_wall:.4}s ({:.4}s/call), frozen {frozen_total}",
        freeze_wall / hours.max(1) as f64
    );
    eprintln!(
        "ignition thaw step {thaw_s:.4}s; refreeze {refreeze_s:.4}s (+{refrozen})"
    );
    eprintln!("area_mean {:.1} ha", output.stats.area_mean / 1e4);
    let stats = sim.freeze_stats();
    eprintln!(
        "fuzzy={fuzzy:?} final: dead {} cold {} fault-ins {} ({:.1} MiB read)",
        stats.dead_tiles,
        stats.cold_tiles,
        stats.fault_ins,
        stats.fault_in_bytes as f64 / (1024.0 * 1024.0)
    );
    eprintln!("total wall {:.3}s", wall0.elapsed().as_secs_f64());
    std::fs::remove_dir_all(&freeze_dir).ok();
}
