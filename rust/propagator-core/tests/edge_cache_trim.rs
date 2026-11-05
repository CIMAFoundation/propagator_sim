//! Trimming the shared edge cache is a pure memory knob: released tiles
//! are recomputed on demand with identical values, so seeded runs stay
//! bitwise identical to untrimmed runs, even when trimming every few
//! simulated minutes with no margin.

use propagator_core::*;

/// Per-realization fire / arrival / ROS / FLI layers plus the output maps
/// (f32 as bit patterns so NaN compares equal).
#[derive(Debug, PartialEq)]
struct Snapshot {
    fire: Vec<Vec<u8>>,
    arrival: Vec<Vec<i32>>,
    ros: Vec<Vec<u32>>,
    fli: Vec<Vec<u32>>,
    maps: Vec<Vec<u32>>,
}

fn bits(grid: &Grid2<f32>) -> Vec<u32> {
    grid.as_slice().iter().map(|v| v.to_bits()).collect()
}

fn snapshot(sim: &mut Propagator) -> Snapshot {
    let out = sim.get_output().unwrap();
    Snapshot {
        fire: sim
            .get_fire()
            .unwrap()
            .iter()
            .map(|g| g.as_slice().to_vec())
            .collect(),
        arrival: sim
            .get_arrival_time()
            .unwrap()
            .iter()
            .map(|g| g.as_slice().to_vec())
            .collect(),
        ros: sim.get_ros().unwrap().iter().map(bits).collect(),
        fli: sim.get_fireline_int().unwrap().iter().map(bits).collect(),
        maps: vec![
            bits(&out.fire_probability),
            bits(&out.min_arrival_time),
            bits(&out.mean_arrival_time),
            bits(&out.ros_mean),
            bits(&out.ros_max),
            bits(&out.fli_mean),
            bits(&out.fli_max),
            bits(&out.spotting_receiving_probability),
        ],
    }
}

/// A run with a wind change, an ignition batch and an expand mid-run.
/// `trim`: margin passed to `trim_edge_cache` after every window. Returns
/// the snapshot and the total number of edge tiles released.
fn run(spotting: bool, threads: usize, trim: Option<usize>) -> (Snapshot, usize) {
    let n = 100;
    let mut veg = Grid2::filled(n, n, 5);
    for row in 0..n {
        if !(40..48).contains(&row) {
            veg[(row, 70)] = NO_FUEL;
        }
    }
    let mut config = PropagatorConfig::new(veg, Grid2::filled(n, n, 0.0));
    config.realizations = 6;
    config.seed = Some(31);
    config.do_spotting = spotting;
    config.oob_mode = OobMode::Ignore;
    config.n_threads = Some(threads);
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
    sim.set_boundary_conditions(BoundaryConditions {
        time: 1800,
        wind_dir: Some(FieldInput::Scalar(160.0)),
        wind_speed: Some(FieldInput::Scalar(40.0)),
        ..Default::default()
    })
    .unwrap();
    // re-ignite inside already burnt (and trimmed) areas
    sim.set_boundary_conditions(BoundaryConditions {
        time: 5400,
        ignitions: Some(Ignitions::PointsWithRealization(vec![
            (50, 20, 0),
            (85, 85, 1),
            (30, 30, 4),
        ])),
        ..Default::default()
    })
    .unwrap();
    let mut released = 0;
    let mut window = |sim: &mut Propagator| {
        sim.step_window(1200).unwrap();
        if let Some(margin) = trim {
            let before = sim.edge_cache_bytes();
            let freed = sim.trim_edge_cache(margin);
            assert!(sim.edge_cache_bytes() <= before);
            released += freed;
        }
    };
    for _ in 0..12 {
        window(&mut sim);
    }
    let grown = n + 64;
    sim.expand(
        Grid2::filled(grown, grown, 5),
        Grid2::filled(grown, grown, 0.0),
        (-32, -32),
    )
    .unwrap();
    for _ in 0..60 {
        window(&mut sim);
    }
    (snapshot(&mut sim), released)
}

#[test]
fn trimming_the_edge_cache_is_bitwise_neutral() {
    for spotting in [false, true] {
        let (plain, _) = run(spotting, 1, None);
        for threads in [1, 4] {
            for margin in [0, 1] {
                let (trimmed, released) = run(spotting, threads, Some(margin));
                let tag = format!("spotting={spotting} threads={threads} margin={margin}");
                assert!(released > 0, "{tag}: nothing was released");
                assert!(plain == trimmed, "{tag}: trimmed run differs");
            }
        }
    }
}

#[test]
fn trimming_keeps_tiles_near_pending_events() {
    let mut config =
        PropagatorConfig::new(Grid2::filled(256, 256, 5), Grid2::filled(256, 256, 0.0));
    config.realizations = 4;
    config.seed = Some(5);
    config.oob_mode = OobMode::Ignore;
    let mut sim = Propagator::new(config).unwrap();
    sim.set_boundary_conditions(BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(4.0)),
        wind_dir: Some(FieldInput::Scalar(0.0)),
        wind_speed: Some(FieldInput::Scalar(10.0)),
        ignitions: Some(Ignitions::Points(vec![(128, 128)])),
        ..Default::default()
    })
    .unwrap();
    sim.step_window(1800).unwrap();
    assert!(sim.next_time().is_some(), "fire must still be active");
    // a huge margin covers the whole grid: nothing may be released
    assert_eq!(sim.trim_edge_cache(1_000), 0);
    assert!(sim.edge_cache_bytes() > 0);
}
