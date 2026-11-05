//! Fuzzy freezing: cold tiles are a pure performance knob. Aggressive
//! settings (every tile without a pending event goes cold, freezing every
//! few simulated minutes) must leave seeded runs bitwise identical to
//! runs without any freezing.

use std::path::PathBuf;

use propagator_core::*;

fn temp_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("prop-core-fuzzy-{tag}-{}", std::process::id()))
}

const AGGRESSIVE: FuzzyFreeze = FuzzyFreeze {
    min_inactive_fraction: 0.0,
    event_margin_tiles: 0,
};

/// Everything a run produces: per-realization state and output maps (f32
/// maps as bit patterns so NaN compares equal).
#[derive(Debug, PartialEq)]
struct Snapshot {
    fire: Vec<Vec<u8>>,
    arrival: Vec<Vec<i32>>,
    ros: Vec<Vec<u32>>,
    fli: Vec<Vec<u32>>,
    spot_gen: Vec<Vec<u8>>,
    spot_recv: Vec<Vec<u8>>,
    maps: Vec<Vec<u32>>,
    stats: (usize, u64, u64, u64, u64),
}

fn bits(grid: &Grid2<f32>) -> Vec<u32> {
    grid.as_slice().iter().map(|v| v.to_bits()).collect()
}

fn snapshot(sim: &mut Propagator) -> Snapshot {
    let flat_u8 = |g: Vec<Grid2<u8>>| g.iter().map(|g| g.as_slice().to_vec()).collect();
    let flat_f32 = |g: Vec<Grid2<f32>>| g.iter().map(bits).collect();
    let out = sim.get_output().unwrap();
    Snapshot {
        fire: flat_u8(sim.get_fire().unwrap()),
        arrival: sim
            .get_arrival_time()
            .unwrap()
            .iter()
            .map(|g| g.as_slice().to_vec())
            .collect(),
        ros: flat_f32(sim.get_ros().unwrap()),
        fli: flat_f32(sim.get_fireline_int().unwrap()),
        spot_gen: flat_u8(sim.get_spotting_generation().unwrap()),
        spot_recv: flat_u8(sim.get_spotting_receiving().unwrap()),
        maps: vec![
            bits(&out.fire_probability),
            bits(&out.spotting_generation_probability),
            bits(&out.spotting_receiving_probability),
            bits(&out.min_arrival_time),
            bits(&out.mean_arrival_time),
            bits(&out.ros_mean),
            bits(&out.ros_max),
            bits(&out.fli_mean),
            bits(&out.fli_max),
            bits(&out.flame_height_mean),
            bits(&out.flame_height_max),
            bits(&out.radiant_reach_mean),
            bits(&out.radiant_reach_max),
        ],
        stats: (
            out.stats.n_active,
            out.stats.area_mean.to_bits(),
            out.stats.area_50.to_bits(),
            out.stats.area_75.to_bits(),
            out.stats.area_90.to_bits(),
        ),
    }
}

/// A scripted run: wind change, an ignition batch, an expand and a
/// checkpoint/restore mid-run. `freeze`: (store dir, fuzzy setting),
/// freezing after every 5-minute window. Returns the snapshot and the
/// freeze stats.
fn scripted_run(
    spotting: bool,
    threads: usize,
    freeze: Option<(PathBuf, Option<FuzzyFreeze>)>,
) -> (Snapshot, FreezeStats) {
    // 100 is not a multiple of the tile size: partial edge tiles
    let n = 100;
    let mut veg = Grid2::filled(n, n, 5);
    for row in 0..n {
        // a non-fuel strip with a gap: islands and detours
        if !(40..48).contains(&row) {
            veg[(row, 70)] = NO_FUEL;
        }
    }
    let dem = Grid2::filled(n, n, 0.0);
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = 6;
    config.seed = Some(2024);
    config.do_spotting = spotting;
    config.oob_mode = OobMode::Ignore;
    config.n_threads = Some(threads);
    if let Some((dir, fuzzy)) = &freeze {
        config.freeze_dir = Some(dir.clone());
        config.fuzzy_freeze = *fuzzy;
    }
    let mut sim = Propagator::new(config).unwrap();
    sim.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(4.0)),
        wind_dir: Some(FieldInput::Scalar(270.0)),
        wind_speed: Some(FieldInput::Scalar(30.0)),
        ignitions: Some(Ignitions::Points(vec![(50, 20)])),
        ..Default::default()
    })
    .unwrap();
    // wind change
    sim.set_boundary_conditions(BoundaryConditions {
        time: 1800,
        wind_dir: Some(FieldInput::Scalar(160.0)),
        wind_speed: Some(FieldInput::Scalar(40.0)),
        ..Default::default()
    })
    .unwrap();
    // an ignition batch, some cells inside the burnt area
    sim.set_boundary_conditions(BoundaryConditions {
        time: 2700,
        ignitions: Some(Ignitions::PointsWithRealization(vec![
            (50, 20, 0),
            (85, 85, 1),
            (85, 85, 3),
            (10, 90, 3),
            (30, 30, 5),
        ])),
        ..Default::default()
    })
    .unwrap();

    let freezing = freeze.is_some();
    let window = |sim: &mut Propagator| {
        sim.step_window(600).unwrap();
        if freezing {
            sim.freeze_inactive_tiles().unwrap();
        }
    };
    for _ in 0..12 {
        window(&mut sim);
    }
    // grow by one tile on every side (north/west tile-aligned)
    let grown = n + 64;
    sim.expand(
        Grid2::filled(grown, grown, 5),
        Grid2::filled(grown, grown, 0.0),
        (-32, -32),
    )
    .unwrap();
    for _ in 0..24 {
        window(&mut sim);
    }
    let checkpoint = sim.checkpoint();
    for _ in 0..6 {
        window(&mut sim);
    }
    sim.restore(&checkpoint).unwrap();
    sim.reseed(99);
    for _ in 0..48 {
        window(&mut sim);
    }
    let stats = sim.freeze_stats();
    (snapshot(&mut sim), stats)
}

#[test]
fn aggressive_fuzzy_freezing_is_bitwise_neutral() {
    for spotting in [false, true] {
        let (plain, _) = scripted_run(spotting, 1, None);
        assert!(plain.fire.iter().all(|f| f.iter().any(|&c| c != 0)));
        for threads in [1, 4] {
            let tag = format!("stress-{spotting}-{threads}");
            let dir = temp_dir(&tag);
            let (fuzzy, stats) = scripted_run(spotting, threads, Some((dir.clone(), Some(AGGRESSIVE))));
            std::fs::remove_dir_all(&dir).ok();
            eprintln!("{tag}: {stats:?}");
            assert!(stats.fault_ins > 0, "{tag}: no fault-in happened");
            assert_eq!(stats.fault_in_bytes, stats.fault_ins * RECORD_SIZE as u64);
            assert!(
                plain == fuzzy,
                "{tag}: fuzzy-frozen run differs from the plain run"
            );
        }
        // default threshold and the dead criterion alone, too
        for (tag, fuzzy) in [("default", Some(FuzzyFreeze::default())), ("dead", None)] {
            let tag = format!("{tag}-{spotting}");
            let dir = temp_dir(&tag);
            let (frozen, _) = scripted_run(spotting, 2, Some((dir.clone(), fuzzy)));
            std::fs::remove_dir_all(&dir).ok();
            assert!(plain == frozen, "{tag}: frozen run differs from the plain run");
        }
    }
}

/// Region B (cols 64..) is cut off from the fire by a non-fuel moat; its
/// tiles are allocated by forced ignitions of non-fuel cells, then go cold.
/// Only embers can reach it, so B burning means embers landed in cold
/// tiles and faulted them in.
fn moat_run(freeze: Option<PathBuf>) -> (Snapshot, FreezeStats, f32) {
    let n = 128;
    let mut veg = Grid2::filled(n, n, 5);
    for row in 0..n {
        for col in 58..64 {
            veg[(row, col)] = NO_FUEL;
        }
    }
    let anchors: Vec<(usize, usize)> = (0..4)
        .flat_map(|k| [(16 + 32 * k, 80), (16 + 32 * k, 112)])
        .collect();
    for &(row, col) in &anchors {
        veg[(row, col)] = NO_FUEL;
    }
    let mut config = PropagatorConfig::new(veg, Grid2::filled(n, n, 0.0));
    config.realizations = 12;
    config.seed = Some(77);
    config.do_spotting = true;
    config.oob_mode = OobMode::Ignore;
    config.n_threads = Some(3);
    if let Some(dir) = &freeze {
        config.freeze_dir = Some(dir.clone());
        config.fuzzy_freeze = Some(AGGRESSIVE);
    }
    let mut sim = Propagator::new(config).unwrap();
    let mut ignitions = vec![(64, 30)];
    ignitions.extend(anchors.iter().copied());
    sim.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(2.0)),
        // wind from the west drives fire and embers east, toward B
        wind_dir: Some(FieldInput::Scalar(270.0)),
        wind_speed: Some(FieldInput::Scalar(50.0)),
        ignitions: Some(Ignitions::Points(ignitions)),
        ..Default::default()
    })
    .unwrap();
    let mut cold_seen = 0;
    for _ in 0..96 {
        sim.step_window(300).unwrap();
        if freeze.is_some() {
            sim.freeze_inactive_tiles().unwrap();
            cold_seen = cold_seen.max(sim.freeze_stats().cold_tiles);
        }
    }
    if freeze.is_some() {
        // B's tiles really went cold
        assert!(cold_seen >= 12 * 4, "only {cold_seen} cold tiles");
    }
    let snap = snapshot(&mut sim);
    let output = sim.get_output().unwrap();
    let mut burnt_b = 0.0f32;
    for row in 0..n {
        for col in 64..n {
            burnt_b += output.fire_probability[(row, col)];
        }
    }
    (snap, sim.freeze_stats(), burnt_b)
}

#[test]
fn embers_landing_in_cold_tiles_fault_them_in_and_burn() {
    let (plain, _, plain_b) = moat_run(None);
    let dir = temp_dir("moat");
    let (fuzzy, stats, fuzzy_b) = moat_run(Some(dir.clone()));
    std::fs::remove_dir_all(&dir).ok();
    // well beyond the 8 forced (non-propagating) anchor cells
    assert!(plain_b > 8.0 + 20.0, "embers never crossed the moat ({plain_b})");
    assert!(stats.fault_ins > 0);
    assert_eq!(plain_b, fuzzy_b);
    assert!(plain == fuzzy, "fuzzy-frozen run differs from the plain run");
}

#[test]
fn fuzzy_freeze_requires_a_store_and_a_valid_fraction() {
    let mut config = PropagatorConfig::new(Grid2::filled(64, 64, 5), Grid2::filled(64, 64, 0.0));
    config.fuzzy_freeze = Some(FuzzyFreeze::default());
    assert!(matches!(
        Propagator::new(config),
        Err(PropagatorError::FreezeDirNotSet)
    ));
    let mut config = PropagatorConfig::new(Grid2::filled(64, 64, 5), Grid2::filled(64, 64, 0.0));
    config.freeze_dir = Some(temp_dir("fraction"));
    config.fuzzy_freeze = Some(FuzzyFreeze {
        min_inactive_fraction: 1.5,
        event_margin_tiles: 1,
    });
    assert!(Propagator::new(config).is_err());
    std::fs::remove_dir_all(temp_dir("fraction")).ok();
}
