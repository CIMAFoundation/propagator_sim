//! GPU vs CPU core on the homogeneous-grassland point-ignition scenario of
//! `propagator-core`'s bench. Usage:
//!
//!   gpu_bench <grid> <realizations> <hours> [seed] [bucket_s] [sweeps]
//!
//! Prints wall times and compares burned areas and fire-probability maps.

use std::time::Instant;

use propagator_core::{
    legacy_fuel_system, BoundaryConditions, FieldInput, FuelDef, FuelSystem, Grid2, Ignitions,
    OobMode, Propagator, PropagatorConfig,
};

/// A landscape exported from the demo asset (`SCENE=<dir>`): `header.txt`
/// (rows cols cellsize ign_row ign_col), `veg.i32`, `dem.f32`, `fuels.txt`
/// (one fuel per line: id v0 d0 d1 hhv humidity|nan spotting prob_ign burn
/// n_pairs id:p ...).
struct Scene {
    veg: Grid2<i32>,
    dem: Grid2<f64>,
    cellsize: f64,
    ignition: (usize, usize),
    fuels: FuelSystem,
}

fn load_scene(dir: &str) -> Scene {
    let dir = std::path::Path::new(dir);
    let header = std::fs::read_to_string(dir.join("header.txt")).expect("header.txt");
    let h: Vec<f64> = header
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let (rows, cols) = (h[0] as usize, h[1] as usize);
    let veg_bytes = std::fs::read(dir.join("veg.i32")).expect("veg.i32");
    let dem_bytes = std::fs::read(dir.join("dem.f32")).expect("dem.f32");
    let veg: Vec<i32> = veg_bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let dem: Vec<f64> = dem_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
        .collect();
    let mut defs = Vec::new();
    for line in std::fs::read_to_string(dir.join("fuels.txt"))
        .expect("fuels.txt")
        .lines()
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() {
            continue;
        }
        let n_pairs: usize = f[9].parse().unwrap();
        let spread_probability = f[10..10 + n_pairs]
            .iter()
            .map(|kv| {
                let (k, v) = kv.split_once(':').unwrap();
                (k.parse::<i32>().unwrap(), v.parse::<f64>().unwrap())
            })
            .collect();
        defs.push(FuelDef {
            id: f[0].parse().unwrap(),
            name: format!("fuel {}", f[0]),
            v0: f[1].parse().unwrap(),
            d0: f[2].parse().unwrap(),
            d1: f[3].parse().unwrap(),
            hhv: f[4].parse().unwrap(),
            humidity: f[5].parse::<f64>().ok().filter(|v| !v.is_nan()),
            spotting: f[6] == "1",
            prob_ign_by_embers: f[7].parse().unwrap(),
            burn: f[8] == "1",
            spread_probability,
        });
    }
    Scene {
        veg: Grid2::from_vec(rows, cols, veg),
        dem: Grid2::from_vec(rows, cols, dem),
        cellsize: h[2],
        ignition: (h[3] as usize, h[4] as usize),
        fuels: FuelSystem::from_defs(&defs).expect("fuel system"),
    }
}
use propagator_gpu::{GpuConfig, GpuPropagator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: &str| args.get(i).map(String::as_str).unwrap_or(d).to_string();
    let n: usize = arg(1, "1000").parse().expect("grid");
    let reals: usize = arg(2, "100").parse().expect("realizations");
    let hours: i64 = arg(3, "12").parse().expect("hours");
    let seed: u64 = arg(4, "12345").parse().expect("seed");
    let bucket: u32 = arg(5, "900").parse().expect("bucket seconds");
    let sweeps: u32 = arg(6, "6").parse().expect("sweeps");

    // scenario knobs (env): MOIST (percent), WIND (km/h), WDIR (deg),
    // SLOPE (m per cell along columns), MIXED=1 (random legacy fuels 1..7),
    // SPOT=1 (ember spotting), SCENE=<dir> (exported demo landscape)
    let envf = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let moisture = envf("MOIST", 0.0);
    let wind_speed = envf("WIND", 30.0);
    let wind_dir = envf("WDIR", 90.0);
    let slope = envf("SLOPE", 0.0);
    let mixed = std::env::var("MIXED").is_ok();
    let spotting = std::env::var("SPOT").is_ok();
    let scene = std::env::var("SCENE").ok().map(|d| load_scene(&d));
    let n = scene.as_ref().map(|s| s.veg.shape().0).unwrap_or(n);
    let c = n / 2;
    let mut veg = Grid2::filled(n, n, 4);
    let mut dem = Grid2::filled(n, n, 0.0);
    let mut cellsize = 20.0;
    let mut ignition = (c, c);
    let mut fuels = legacy_fuel_system();
    if let Some(s) = &scene {
        veg = s.veg.clone();
        dem = s.dem.clone();
        cellsize = s.cellsize;
        ignition = s.ignition;
        fuels = s.fuels.clone();
    }
    if mixed {
        // deterministic pseudo-random legacy fuel codes 1..7 in 8x8 patches
        let mut x = 0x9E37_79B9u32;
        for row in 0..n {
            for col in 0..n {
                if row % 8 == 0 && col % 8 == 0 {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                }
                let patch = ((row / 8) * 7919 + (col / 8) * 104729) as u32 ^ x;
                veg[(row, col)] = 1 + (patch % 7) as i32;
            }
        }
    }
    if slope != 0.0 {
        for row in 0..n {
            for col in 0..n {
                dem[(row, col)] =
                    slope * col as f64 + 0.3 * slope * ((row as f64) * 0.05).sin() * 10.0;
            }
        }
    }

    // --- CPU ---
    let mut config = PropagatorConfig::new(veg.clone(), dem.clone());
    config.realizations = reals;
    config.seed = Some(seed);
    config.oob_mode = OobMode::Ignore;
    config.cellsize = cellsize;
    config.fuels = Some(fuels.clone());
    config.do_spotting = spotting;
    let mut cpu = Propagator::new(config).expect("cpu propagator");
    cpu.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(moisture)),
        wind_dir: Some(FieldInput::Scalar(wind_dir)),
        wind_speed: Some(FieldInput::Scalar(wind_speed)),
        ignitions: Some(Ignitions::Points(vec![ignition])),
        ..Default::default()
    })
    .expect("bc");
    let t0 = Instant::now();
    for _ in 0..hours {
        cpu.step_window(3600).expect("step");
    }
    let cpu_wall = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let cpu_out = cpu.get_output().expect("cpu output");
    let cpu_out_wall = t0.elapsed().as_secs_f64();

    // --- GPU ---
    let t0 = Instant::now();
    let mut gcfg = GpuConfig::new(veg, dem, fuels);
    gcfg.realizations = reals;
    gcfg.seed = seed as u32;
    gcfg.cellsize = cellsize;
    gcfg.do_spotting = spotting;
    gcfg.bucket_seconds = bucket;
    gcfg.sweeps_per_bucket = sweeps;
    let mut gpu = GpuPropagator::new(gcfg).expect("gpu propagator");
    gpu.set_scalar_weather(0, wind_dir, wind_speed, moisture, &[ignition])
        .expect("gpu bc");
    let gpu_setup = t0.elapsed().as_secs_f64();
    if std::env::var("GPU_DEBUG").is_ok() {
        let edges = gpu
            .debug_edges(ignition.0, ignition.1)
            .expect("debug edges");
        for (d, e) in edges.iter().enumerate() {
            eprintln!(
                "dir {d}: p={:.4} tt={:.0} ros={:.3} fli={:.1} angle={:.3} u={:.3} wdir={:.3} wspd={:.1}",
                e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]
            );
        }
        // CPU reference for the same geometry
        let fuels = legacy_fuel_system();
        let idx = fuels
            .build_fuel_index_grid(&Grid2::filled(1, 1, 4))
            .unwrap()[(0, 0)] as usize;
        let p0 = fuels.transition_probability(idx, idx);
        for (d, (dr, dc)) in [
            (-1i64, -1i64),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        .iter()
        .enumerate()
        {
            let dist = ((dr * dr + dc * dc) as f64).sqrt() * 20.0;
            let angle = ((*dc as f64).atan2(-(*dr as f64)) + std::f64::consts::PI)
                .rem_euclid(2.0 * std::f64::consts::PI);
            let w_dir = wind_dir.to_radians();
            let p = propagator_core::probability_to_neighbour(
                propagator_core::MoistureModel::Trucchia,
                angle,
                dist,
                w_dir,
                wind_speed,
                moisture / 100.0,
                0.0,
                p0,
            );
            let (t, ros) = propagator_core::p_time(
                propagator_core::RosModel::Wang,
                fuels.v0()[idx],
                0.0,
                angle,
                dist,
                moisture / 100.0,
                w_dir,
                wind_speed,
            );
            eprintln!(
                "cpu {d}: p={p:.4} tt={:.0} ros={ros:.3} angle={angle:.3}",
                t.floor().max(1.0)
            );
        }
    }
    let t0 = Instant::now();
    if std::env::var("GPU_ONE_SHOT").is_ok() {
        gpu.advance_to(hours * 3600).expect("advance");
    } else {
        for h in 1..=hours {
            gpu.advance_to(h * 3600).expect("advance");
        }
    }
    let gpu_wall = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let gpu_out = gpu.get_output().expect("gpu output");
    let gpu_out_wall = t0.elapsed().as_secs_f64();

    // --- compare ---
    let a = cpu_out.fire_probability.as_slice();
    let b = gpu_out.fire_probability.as_slice();
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut l1 = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
        l1 += (x - y).abs();
    }
    let m = a.len() as f64;
    let corr = (sab - sa * sb / m) / ((saa - sa * sa / m).sqrt() * (sbb - sb * sb / m).sqrt());
    let cpu_area = cpu_out.stats.area_mean / 1e4;
    let gpu_area = gpu_out.area_mean / 1e4;
    // probability-weighted centroid offset from the ignition (rows, cols)
    let centroid = |p: &[f32]| -> (f64, f64) {
        let (mut sr, mut sc, mut sw) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &v) in p.iter().enumerate() {
            let v = v as f64;
            sr += (i / n) as f64 * v;
            sc += (i % n) as f64 * v;
            sw += v;
        }
        (sr / sw - ignition.0 as f64, sc / sw - ignition.1 as f64)
    };
    let (cpu_cr, cpu_cc) = centroid(a);
    let (gpu_cr, gpu_cc) = centroid(b);
    eprintln!(
        "     centroid offset (drow, dcol): cpu=({cpu_cr:.1}, {cpu_cc:.1}) gpu=({gpu_cr:.1}, {gpu_cc:.1})"
    );

    eprintln!(
        "CPU  wall={cpu_wall:.3}s output={:.1}ms area_mean={cpu_area:.1}ha area_50={:.1}ha",
        cpu_out_wall * 1e3,
        cpu_out.stats.area_50 / 1e4
    );
    eprintln!(
        "GPU  wall={gpu_wall:.3}s output={:.1}ms setup={:.3}s area_mean={gpu_area:.1}ha area_50={:.1}ha  (bucket={bucket}s sweeps={sweeps})",
        gpu_out_wall * 1e3,
        gpu_setup,
        gpu_out.area_50 / 1e4
    );
    eprintln!(
        "     area ratio gpu/cpu={:.4}  prob-map corr={corr:.4}  mean|dp|={:.5}  speedup={:.2}x",
        gpu_area / cpu_area,
        l1 / m,
        cpu_wall / gpu_wall
    );
    println!(
        "GPU\t{n}\t{reals}\t{hours}\t{cpu_wall:.4}\t{gpu_wall:.4}\t{cpu_area:.2}\t{gpu_area:.2}\t{corr:.4}"
    );
}
