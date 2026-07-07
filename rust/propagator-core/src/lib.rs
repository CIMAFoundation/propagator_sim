//! PROPAGATOR core — stochastic cellular wildfire spread simulator.
//!
//! Rust rewrite of the Python/numba core (`propagator.core`), specified
//! in `docs/rust-core-spec.md`. The engine is event-driven: each of the
//! `R` stochastic realizations owns a min-heap of pending ignition
//! events and block-sparse 32x32 state tiles, so memory scales with the
//! burned area rather than the grid. Burned-out interior tiles can be
//! frozen to disk, the domain can grow in place or through world-anchored
//! checkpoints, and runs seeded with [`Propagator::reseed`] are bitwise
//! reproducible regardless of thread count.
//!
//! ```no_run
//! use propagator_core::*;
//!
//! let veg = Grid2::filled(256, 256, 4); // grassland
//! let dem = Grid2::filled(256, 256, 0.0);
//! let mut config = PropagatorConfig::new(veg, dem);
//! config.seed = Some(42);
//! let mut sim = Propagator::new(config).unwrap();
//!
//! sim.set_boundary_conditions(BoundaryConditions {
//!     time: 0,
//!     moisture: Some(FieldInput::Scalar(8.0)),   // percent
//!     wind_dir: Some(FieldInput::Scalar(180.0)), // degrees from north
//!     wind_speed: Some(FieldInput::Scalar(15.0)),// km/h
//!     ignitions: Some(Ignitions::Points(vec![(128, 128)])),
//!     ..Default::default()
//! }).unwrap();
//!
//! sim.step_window(3600).unwrap(); // one hour
//! let output = sim.get_output().unwrap();
//! println!("burned area: {:.1} ha", output.stats.area_mean / 10_000.0);
//! ```

mod checkpoint;
mod error;
mod fold;
mod front;
mod fuels;
mod grid;
mod kernel;
mod models;
mod propagator;
mod rng;
mod scheduler;
mod tile_store;
mod tiles;

pub use checkpoint::{PropagatorCheckpoint, ResumeOptions, CHECKPOINT_VERSION};
pub use error::{PropagatorError, Result};
pub use fold::{PropagatorOutput, PropagatorStats, StateFold};
pub use fuels::{
    legacy_fuel_defs, legacy_fuel_system, FuelDef, FuelSystem, HUMIDITY_UNSET,
    NO_FUEL,
};
pub use grid::Grid2;
pub use models::{MoistureModel, RosModel};
pub use propagator::{
    OobMode, Propagator, PropagatorConfig, CELLSIZE_DEFAULT, REALIZATIONS_DEFAULT,
};
pub use scheduler::{
    BoundaryConditions, FieldInput, Ignitions, Scheduler, SchedulerEvent, UpdateBatch,
};
pub use tile_store::{TileIndex, TileKey, TileStore, RECORD_SIZE};
pub use tiles::{
    FLAG_FIRE, FLAG_SPOT_GEN, FLAG_SPOT_RECV, TILE_SIZE,
};
