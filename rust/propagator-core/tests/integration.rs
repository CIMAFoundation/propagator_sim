//! End-to-end tests of the spec's behavioural invariants (spec §12).

use propagator_core::*;

const ROWS: usize = 128;
const COLS: usize = 128;

fn flat_grass() -> (Grid2<i32>, Grid2<f64>) {
    (
        Grid2::filled(ROWS, COLS, 4), // grassland
        Grid2::filled(ROWS, COLS, 0.0),
    )
}

fn basic_conditions(ignition: (usize, usize)) -> BoundaryConditions {
    BoundaryConditions {
        time: 0,
        moisture: Some(FieldInput::Scalar(5.0)),
        wind_dir: Some(FieldInput::Scalar(0.0)),
        wind_speed: Some(FieldInput::Scalar(10.0)),
        ignitions: Some(Ignitions::Points(vec![ignition])),
        ..Default::default()
    }
}

fn new_sim(seed: u64) -> Propagator {
    let (veg, dem) = flat_grass();
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = 20;
    config.seed = Some(seed);
    config.oob_mode = OobMode::Ignore;
    Propagator::new(config).unwrap()
}

#[test]
fn fire_spreads_from_ignition() {
    let mut sim = new_sim(1);
    sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
    sim.step_window(1800).unwrap();
    let output = sim.get_output().unwrap();

    // the ignition cell burns in every realization
    assert_eq!(output.fire_probability[(64, 64)], 1.0);
    // fire spread beyond the ignition
    assert!(output.stats.area_mean > 20.0 * 20.0 * 10.0);
    // arrival times: ignition at 0, neighbours strictly later
    assert_eq!(output.min_arrival_time[(64, 64)], 0.0);
    let arrivals = sim.get_arrival_time().unwrap();
    let fires = sim.get_fire().unwrap();
    for r in 0..sim.realizations() {
        for row in 0..ROWS {
            for col in 0..COLS {
                if fires[r][(row, col)] != 0 && (row, col) != (64, 64) {
                    assert!(arrivals[r][(row, col)] >= 1);
                }
            }
        }
    }
}

#[test]
fn seeded_runs_are_reproducible() {
    let run = || {
        let mut sim = new_sim(1234);
        sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
        sim.step_window(1800).unwrap();
        sim.get_output().unwrap()
    };
    let first = run();
    let second = run();
    assert_eq!(
        first.fire_probability.as_slice(),
        second.fire_probability.as_slice()
    );
    assert_eq!(first.ros_max.as_slice(), second.ros_max.as_slice());
    assert_eq!(first.stats, second.stats);
}

#[test]
fn thread_count_does_not_change_results() {
    let run = |threads: usize| {
        let (veg, dem) = flat_grass();
        let mut config = PropagatorConfig::new(veg, dem);
        config.realizations = 20;
        config.seed = Some(99);
        config.oob_mode = OobMode::Ignore;
        config.n_threads = Some(threads);
        let mut sim = Propagator::new(config).unwrap();
        sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
        sim.step_window(1800).unwrap();
        sim.get_output().unwrap()
    };
    let serial = run(1);
    let parallel = run(8);
    assert_eq!(
        serial.fire_probability.as_slice(),
        parallel.fire_probability.as_slice()
    );
}

#[test]
fn no_fuel_blocks_spread() {
    let (mut veg, dem) = flat_grass();
    // vertical NO_FUEL barrier splitting the domain
    for row in 0..ROWS {
        for col in 80..84 {
            veg[(row, col)] = NO_FUEL;
        }
    }
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = 10;
    config.seed = Some(7);
    config.oob_mode = OobMode::Ignore;
    let mut sim = Propagator::new(config).unwrap();
    sim.set_boundary_conditions(basic_conditions((64, 40))).unwrap();
    sim.step_window(24 * 3600).unwrap();
    let output = sim.get_output().unwrap();
    for row in 0..ROWS {
        for col in 84..COLS {
            assert_eq!(
                output.fire_probability[(row, col)],
                0.0,
                "fire crossed the barrier at ({row}, {col})"
            );
        }
    }
}

#[test]
fn cells_burn_at_most_once_per_realization() {
    let mut sim = new_sim(3);
    sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
    sim.step_window(3600).unwrap();
    // fire probability is count/R; a double burn would push it over 1
    let output = sim.get_output().unwrap();
    for &p in output.fire_probability.as_slice() {
        assert!((0.0..=1.0).contains(&p));
    }
}

#[test]
fn boundary_halt_is_resumable_by_expansion() {
    let (veg, dem) = flat_grass();
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = 10;
    config.seed = Some(5);
    config.oob_mode = OobMode::Raise;
    let mut sim = Propagator::new(config).unwrap();
    // ignite near the west edge so the fire hits the boundary quickly
    sim.set_boundary_conditions(basic_conditions((64, 3))).unwrap();
    sim.step_window(1).unwrap();
    assert_eq!(
        sim.boundary_proximity(4),
        (false, false, true, false),
        "westward boundary proximity was not detected before propagation",
    );

    let mut hit_boundary = false;
    for _ in 0..200 {
        match sim.step_window(600) {
            Ok(()) => {}
            Err(PropagatorError::OutOfBounds) => {
                hit_boundary = true;
                break;
            }
            Err(err) => panic!("unexpected error: {err}"),
        }
    }
    assert!(hit_boundary, "fire never reached the boundary");

    // expand west by 2 tiles (64 cells) and resume
    let new_cols = COLS + 64;
    let veg = Grid2::filled(ROWS, new_cols, 4);
    let dem = Grid2::filled(ROWS, new_cols, 0.0);
    sim.expand(veg, dem, (0, -64)).unwrap();
    assert_eq!(sim.world_bounds(), (0, -64, ROWS as i64 - 1, COLS as i64 - 1));

    // grassland spreads at ~1 cell / 10 min here: give it an hour
    sim.step_window(3600).unwrap();
    let output = sim.get_output().unwrap();
    // the burnt state survived the expansion, shifted by 64 columns
    assert_eq!(output.fire_probability[(64, 3 + 64)], 1.0);
    // and the fire kept spreading west of the old boundary
    let west_burnt: f32 = (0..ROWS)
        .map(|row| output.fire_probability[(row, 63)])
        .sum();
    assert!(west_burnt > 0.0, "fire did not resume past the old edge");
}

#[test]
fn giving_up_on_growth_clips_instead_of_freezing() {
    let (veg, dem) = flat_grass();
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = 10;
    config.seed = Some(5);
    config.oob_mode = OobMode::Raise;
    let mut sim = Propagator::new(config).unwrap();
    // ignite near the west edge so the fire halts on the boundary quickly
    sim.set_boundary_conditions(basic_conditions((64, 3))).unwrap();

    let mut hit_boundary = false;
    for _ in 0..200 {
        match sim.step_window(600) {
            Ok(()) => {}
            Err(PropagatorError::OutOfBounds) => {
                hit_boundary = true;
                break;
            }
            Err(err) => panic!("unexpected error: {err}"),
        }
    }
    assert!(hit_boundary, "fire never reached the boundary");
    let burnt_at_halt: f32 = sim
        .get_output()
        .unwrap()
        .fire_probability
        .as_slice()
        .iter()
        .sum();

    // No larger grid is coming: clip at the edge rather than suspending the
    // front forever. Stepping must now succeed and keep burning inwards.
    sim.set_oob_mode(OobMode::Ignore);
    assert_eq!(sim.oob_mode(), OobMode::Ignore);
    for _ in 0..6 {
        sim.step_window(600).expect("a clipping run must not halt");
    }
    let burnt_after: f32 = sim
        .get_output()
        .unwrap()
        .fire_probability
        .as_slice()
        .iter()
        .sum();
    assert!(
        burnt_after > burnt_at_halt,
        "front stayed frozen after growth was given up ({burnt_after} <= {burnt_at_halt})"
    );
}

#[test]
fn checkpoint_restore_roundtrip() {
    let mut sim = new_sim(11);
    sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
    sim.step_window(900).unwrap();

    let checkpoint = sim.checkpoint();
    let at_checkpoint = sim.get_output().unwrap();

    // deterministic continuation A
    sim.reseed(555);
    sim.step_window(900).unwrap();
    let continued_a = sim.get_output().unwrap();

    // roll back and repeat with the same seed
    sim.restore(&checkpoint).unwrap();
    let restored = sim.get_output().unwrap();
    assert_eq!(
        restored.fire_probability.as_slice(),
        at_checkpoint.fire_probability.as_slice()
    );
    assert_eq!(restored.time, at_checkpoint.time);

    sim.reseed(555);
    sim.step_window(900).unwrap();
    let continued_b = sim.get_output().unwrap();
    assert_eq!(
        continued_a.fire_probability.as_slice(),
        continued_b.fire_probability.as_slice()
    );
    assert_eq!(
        continued_a.ros_max.as_slice(),
        continued_b.ros_max.as_slice()
    );
}

#[test]
fn from_checkpoint_grows_the_domain() {
    let mut sim = new_sim(13);
    sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
    sim.step_window(900).unwrap();
    let checkpoint = sim.checkpoint();
    let before = sim.get_output().unwrap();

    // resume on a grid grown by one tile on every side
    let grown_rows = ROWS + 64;
    let grown_cols = COLS + 64;
    let veg = Grid2::filled(grown_rows, grown_cols, 4);
    let dem = Grid2::filled(grown_rows, grown_cols, 0.0);
    let mut options = ResumeOptions::default();
    options.domain = Some((veg, dem, (-32, -32)));
    options.oob_mode = OobMode::Ignore;
    options.seed = Some(777);
    let mut grown = Propagator::from_checkpoint(&checkpoint, options).unwrap();

    assert_eq!(grown.time(), checkpoint.time);
    let after = grown.get_output().unwrap();
    // world-anchored state: every burnt cell moved by exactly the shift
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(
                before.fire_probability[(row, col)],
                after.fire_probability[(row + 32, col + 32)],
                "mismatch at ({row}, {col})"
            );
        }
    }
    // and the simulation keeps running on the larger grid
    grown.step_window(900).unwrap();
    let final_output = grown.get_output().unwrap();
    assert!(final_output.stats.area_mean > after.stats.area_mean);
}

#[test]
fn freezing_is_behaviour_neutral() {
    let freeze_dir = std::env::temp_dir().join(format!(
        "prop-core-freeze-test-{}",
        std::process::id()
    ));

    let run = |freeze: bool| {
        let (veg, dem) = flat_grass();
        let mut config = PropagatorConfig::new(veg, dem);
        config.realizations = 10;
        config.seed = Some(21);
        config.oob_mode = OobMode::Ignore;
        if freeze {
            config.freeze_dir = Some(freeze_dir.clone());
        }
        let mut sim = Propagator::new(config).unwrap();
        sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
        let mut frozen_any = 0;
        // Run to exhaustion in 1-hour windows: grassland at 20 m/cell
        // spreads ~10 min/cell crosswind, so the interior tiles only burn
        // out (and become freezable) after several hours of sim time.
        for _ in 0..48 {
            if sim.next_time().is_none() {
                break;
            }
            sim.step_window(3600).unwrap();
            if freeze {
                frozen_any += sim.freeze_inactive_tiles().unwrap();
            }
        }
        (sim.get_output().unwrap(), frozen_any, sim)
    };

    let (plain, _, _) = run(false);
    let (frozen, frozen_count, mut frozen_sim) = run(true);

    assert!(frozen_count > 0, "no tiles were ever frozen");
    // identical outputs (freezing changes neither dynamics nor RNG)
    assert_eq!(
        plain.fire_probability.as_slice(),
        frozen.fire_probability.as_slice()
    );
    assert_eq!(plain.ros_max.as_slice(), frozen.ros_max.as_slice());
    assert_eq!(
        plain.mean_arrival_time.as_slice(),
        frozen.mean_arrival_time.as_slice()
    );

    // thawing everything must not change the outputs either
    let thawed_count = frozen_sim.thaw_all().unwrap();
    assert!(thawed_count > 0);
    let thawed = frozen_sim.get_output().unwrap();
    assert_eq!(
        plain.fire_probability.as_slice(),
        thawed.fire_probability.as_slice()
    );

    std::fs::remove_dir_all(&freeze_dir).ok();
}

#[test]
fn frozen_checkpoint_restores_incrementally() {
    let freeze_dir = std::env::temp_dir().join(format!(
        "prop-core-freeze-cp-test-{}",
        std::process::id()
    ));
    let (veg, dem) = flat_grass();
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = 10;
    config.seed = Some(23);
    config.oob_mode = OobMode::Ignore;
    config.freeze_dir = Some(freeze_dir.clone());
    let mut sim = Propagator::new(config).unwrap();
    sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
    // Run until interior tiles have burned out and been frozen (see
    // `freezing_is_behaviour_neutral` for the timing rationale).
    let mut frozen = 0;
    for _ in 0..48 {
        if sim.next_time().is_none() {
            break;
        }
        sim.step_window(3600).unwrap();
        frozen += sim.freeze_inactive_tiles().unwrap();
        if frozen > 0 {
            break;
        }
    }
    let checkpoint = sim.checkpoint();
    assert!(!checkpoint.frozen_index.is_empty());
    let at_checkpoint = sim.get_output().unwrap();

    // keep running (thaws nothing here, but state diverges)
    sim.step_window(1200).unwrap();
    sim.freeze_inactive_tiles().unwrap();

    // rollback on the same session store: index-only restore
    sim.restore(&checkpoint).unwrap();
    let restored = sim.get_output().unwrap();
    assert_eq!(
        restored.fire_probability.as_slice(),
        at_checkpoint.fire_probability.as_slice()
    );

    // resume without a store: frozen tiles materialize into the pools
    let mut options = ResumeOptions::default();
    options.oob_mode = OobMode::Ignore;
    let mut resumed = Propagator::from_checkpoint(&checkpoint, options).unwrap();
    let materialized = resumed.get_output().unwrap();
    assert_eq!(
        materialized.fire_probability.as_slice(),
        at_checkpoint.fire_probability.as_slice()
    );

    std::fs::remove_dir_all(&freeze_dir).ok();
}

#[test]
fn ignition_into_frozen_area_thaws_and_burns() {
    let freeze_dir = std::env::temp_dir().join(format!(
        "prop-core-thaw-ign-test-{}",
        std::process::id()
    ));
    let (veg, dem) = flat_grass();
    let mut config = PropagatorConfig::new(veg, dem);
    config.realizations = 5;
    config.seed = Some(31);
    config.oob_mode = OobMode::Ignore;
    config.freeze_dir = Some(freeze_dir.clone());
    let mut sim = Propagator::new(config).unwrap();
    sim.set_boundary_conditions(basic_conditions((32, 32))).unwrap();
    sim.step_window(1200).unwrap();
    sim.freeze_inactive_tiles().unwrap();

    // a fresh ignition far away must thaw whatever it can reach and burn
    sim.set_boundary_conditions(BoundaryConditions {
        time: sim.time(),
        ignitions: Some(Ignitions::Points(vec![(100, 100)])),
        ..Default::default()
    })
    .unwrap();
    sim.step_window(600).unwrap();
    let output = sim.get_output().unwrap();
    assert_eq!(output.fire_probability[(100, 100)], 1.0);

    std::fs::remove_dir_all(&freeze_dir).ok();
}

#[test]
fn actions_moisture_decays_and_slows_spread() {
    // a heavily watered domain must burn less than a dry one
    let run = |water: bool| {
        let mut sim = new_sim(17);
        let mut bc = basic_conditions((64, 64));
        if water {
            bc.additional_moisture = Some(Grid2::filled(ROWS, COLS, 25.0));
        }
        sim.set_boundary_conditions(bc).unwrap();
        sim.step_window(1800).unwrap();
        sim.get_output().unwrap().stats.area_mean
    };
    let dry = run(false);
    let wet = run(true);
    assert!(
        wet < dry,
        "watering did not slow the fire (wet {wet}, dry {dry})"
    );
}

#[test]
fn vegetation_changes_apply_at_their_time() {
    let mut sim = new_sim(19);
    sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();

    // at t=600 turn the whole east half into NO_FUEL (a huge firebreak)
    let mut changes = Grid2::filled(ROWS, COLS, f64::NAN);
    for row in 0..ROWS {
        for col in 96..COLS {
            changes[(row, col)] = NO_FUEL as f64;
        }
    }
    sim.set_boundary_conditions(BoundaryConditions {
        time: 600,
        vegetation_changes: Some(changes),
        ..Default::default()
    })
    .unwrap();

    sim.step_window(24 * 3600).unwrap();
    let output = sim.get_output().unwrap();
    // nothing east of the firebreak burns after the change (the fire
    // cannot have reached col >= 100 within 600 s from col 64)
    for row in 0..ROWS {
        for col in 100..COLS {
            assert_eq!(output.fire_probability[(row, col)], 0.0);
        }
    }
}

#[test]
fn spotting_flags_only_with_spotting_enabled() {
    let run = |spotting: bool| {
        let veg = Grid2::filled(ROWS, COLS, 5); // conifers: spotting-prone
        let dem = Grid2::filled(ROWS, COLS, 0.0);
        let mut config = PropagatorConfig::new(veg, dem);
        config.realizations = 10;
        config.seed = Some(29);
        config.do_spotting = spotting;
        config.oob_mode = OobMode::Ignore;
        let mut sim = Propagator::new(config).unwrap();
        let mut bc = basic_conditions((64, 64));
        bc.wind_speed = Some(FieldInput::Scalar(40.0)); // strong wind
        sim.set_boundary_conditions(bc).unwrap();
        sim.step_window(1800).unwrap();
        let output = sim.get_output().unwrap();
        let gen: f32 = output
            .spotting_generation_probability
            .as_slice()
            .iter()
            .sum();
        let recv: f32 = output
            .spotting_receiving_probability
            .as_slice()
            .iter()
            .sum();
        (gen, recv)
    };
    let (gen_off, recv_off) = run(false);
    assert_eq!(gen_off, 0.0);
    assert_eq!(recv_off, 0.0);
    let (gen_on, recv_on) = run(true);
    assert!(gen_on > 0.0, "no embers generated with spotting on");
    assert!(recv_on > 0.0, "no embers received with spotting on");
}

#[test]
fn boundary_conditions_in_the_past_are_rejected() {
    let mut sim = new_sim(37);
    sim.set_boundary_conditions(basic_conditions((64, 64))).unwrap();
    sim.step_window(600).unwrap();
    let result = sim.set_boundary_conditions(BoundaryConditions {
        time: 0,
        wind_speed: Some(FieldInput::Scalar(20.0)),
        ..Default::default()
    });
    assert!(matches!(
        result,
        Err(PropagatorError::InvalidBoundaryConditions(_))
    ));
}

#[test]
fn growth_must_be_tile_aligned_and_containing() {
    let mut sim = new_sim(41);
    let veg = Grid2::filled(ROWS + 10, COLS + 10, 4);
    let dem = Grid2::filled(ROWS + 10, COLS + 10, 0.0);
    // shift of 10 is not a TILE_SIZE multiple
    assert!(matches!(
        sim.expand(veg, dem, (-10, -10)),
        Err(PropagatorError::InvalidGrowth(_))
    ));
    // smaller grid cannot contain the old one
    let veg = Grid2::filled(64, 64, 4);
    let dem = Grid2::filled(64, 64, 0.0);
    assert!(matches!(
        sim.expand(veg, dem, (0, 0)),
        Err(PropagatorError::InvalidGrowth(_))
    ));
}
