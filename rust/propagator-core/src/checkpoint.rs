//! Checkpointing: immutable snapshots of every dynamic quantity of a
//! running simulation, anchored in world cell coordinates so they can be
//! restored onto larger, shifted grids (domain growth).

use std::path::PathBuf;
use std::sync::Arc;

use crate::front::FrontHeap;
use crate::fuels::FuelSystem;
use crate::grid::Grid2;
use crate::models::{MoistureModel, RosModel};
use crate::scheduler::SchedulerEvent;
use crate::tile_store::{TileIndex, TileStore};
use crate::tiles::TileGrid;
use crate::OobMode;

pub const CHECKPOINT_VERSION: u32 = 1;

/// Immutable snapshot of a `Propagator`'s dynamic state.
///
/// Frozen tiles are captured *incrementally*: the checkpoint holds
/// `{key -> offset}` references into the (append-only) session tile
/// store instead of copying the records, so snapshotting a run with a
/// large frozen interior costs neither RAM nor tile I/O. Such a
/// checkpoint depends on the store — don't `clear()` it while the
/// checkpoint is still needed.
///
/// The RNG state is not captured: runs resumed from a checkpoint are
/// statistically equivalent but not bitwise reproductions of the
/// original continuation (use `reseed` for deterministic continuations).
#[derive(Clone)]
pub struct PropagatorCheckpoint {
    pub version: u32,
    pub time: i64,
    pub origin: (i64, i64),
    pub cellsize: f64,
    pub realizations: usize,
    pub do_spotting: bool,

    pub veg: Grid2<i32>,
    pub dem: Grid2<f64>,

    pub moisture: Option<Grid2<f32>>,
    pub wind_dir: Option<Grid2<f32>>,
    pub wind_speed: Option<Grid2<f32>>,
    pub actions_moisture: Option<Grid2<f32>>,

    /// per-realization front heaps and tiled state
    pub(crate) heaps: Vec<FrontHeap>,
    pub(crate) tiles: Vec<TileGrid>,

    /// pending boundary-condition queue as (time, event) pairs
    pub scheduler_events: Vec<(i64, SchedulerEvent)>,

    /// frozen tiles as {key -> record offset} into `frozen_source`
    pub frozen_index: TileIndex,
    pub(crate) frozen_source: Option<Arc<TileStore>>,
}

impl PropagatorCheckpoint {
    /// Grid shape (rows, cols) the checkpoint was taken on.
    pub fn shape(&self) -> (usize, usize) {
        self.veg.shape()
    }

    /// Inclusive world-cell bounds (row0, col0, row1, col1).
    pub fn world_bounds(&self) -> (i64, i64, i64, i64) {
        let (rows, cols) = self.shape();
        (
            self.origin.0,
            self.origin.1,
            self.origin.0 + rows as i64 - 1,
            self.origin.1 + cols as i64 - 1,
        )
    }
}

/// Options for `Propagator::from_checkpoint`.
#[derive(Default)]
pub struct ResumeOptions {
    /// Resume on a new (typically larger) grid: (veg, dem, origin). The
    /// new grid must fully contain the checkpointed one and north/west
    /// growth must be a multiple of TILE_SIZE cells. When `None` the
    /// checkpointed domain is reconstructed as-is.
    pub domain: Option<(Grid2<i32>, Grid2<f64>, (i64, i64))>,
    /// Fuel system and model functions are not serialized; pass the same
    /// ones used originally (defaults apply when omitted).
    pub fuels: Option<FuelSystem>,
    pub ros_model: Option<RosModel>,
    pub moisture_model: Option<MoistureModel>,
    pub oob_mode: OobMode,
    pub seed: Option<u64>,
    pub freeze_dir: Option<PathBuf>,
    pub n_threads: Option<usize>,
}

/// Rewrite a pending scheduler event (in place) for a re-anchored grid:
/// update coordinates shift; weather fields pad by edge replication,
/// action moisture with zeros, vegetation changes with NaN (no change).
pub(crate) fn reanchor_event(
    event: &mut SchedulerEvent,
    row_shift: usize,
    col_shift: usize,
    old_shape: (usize, usize),
    new_shape: (usize, usize),
) {
    if old_shape == new_shape && row_shift == 0 && col_shift == 0 {
        return;
    }
    for row in &mut event.updates.rows {
        *row += row_shift as i32;
    }
    for col in &mut event.updates.cols {
        *col += col_shift as i32;
    }
    for field in [
        &mut event.moisture,
        &mut event.wind_dir,
        &mut event.wind_speed,
    ] {
        if let Some(grid) = field.take() {
            *field = Some(grid.pad_edge(new_shape, row_shift, col_shift));
        }
    }
    if let Some(grid) = event.additional_moisture.take() {
        event.additional_moisture =
            Some(grid.pad_const(new_shape, row_shift, col_shift, 0.0));
    }
    if let Some(grid) = event.vegetation_changes.take() {
        event.vegetation_changes =
            Some(grid.pad_const(new_shape, row_shift, col_shift, f64::NAN));
    }
}
