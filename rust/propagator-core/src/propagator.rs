//! The main `Propagator`: stochastic cellular wildfire spread simulator.
//!
//! Behavioural port of `propagator/core/propagator.py` (see
//! `docs/rust-core-spec.md` for the full specification).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use crate::checkpoint::{
    reanchor_event, PropagatorCheckpoint, ResumeOptions, CHECKPOINT_VERSION,
};
use crate::edges::EdgeCache;
use crate::error::{PropagatorError, Result};
use crate::fold::{fold_realization, PropagatorOutput, PropagatorStats, StateFold};
use crate::fuels::{legacy_fuel_system, FuelSystem};
use crate::grid::Grid2;
use crate::kernel::{advance_all, Realization, SharedInputs};
use crate::models::{MoistureModel, RosModel};
use crate::rng::Rng;
use crate::scheduler::{
    BoundaryConditions, Ignitions, Scheduler, SchedulerEvent, UpdateBatch,
};
use crate::tile_store::{record_to_tile, TileKey, TileStore};
use crate::reach::{is_empty, live_open, FuelMasks, Reach};
use crate::tiles::{
    FrozenSummary, Tile, FLAG_FIRE, FLAG_SPOT_GEN, FLAG_SPOT_RECV, FROZEN_TILE, TILE_SHIFT,
    TILE_SIZE,
};

/// Behaviour when fire reaches the boundary ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OobMode {
    /// Record but continue; the fire stops at the boundary.
    Ignore,
    /// Halt before igniting a boundary cell and return
    /// `PropagatorError::OutOfBounds`; state is intact and resumable on
    /// a larger grid.
    #[default]
    Raise,
}

/// per-minute fractional decay of action moisture
const ACTIONS_MOISTURE_DECAY: f64 = 0.01;

pub const CELLSIZE_DEFAULT: f64 = 20.0;
pub const REALIZATIONS_DEFAULT: usize = 100;

/// Construction parameters for [`Propagator`].
pub struct PropagatorConfig {
    pub veg: Grid2<i32>,
    pub dem: Grid2<f64>,
    /// `None` selects the legacy 7-fuel table.
    pub fuels: Option<FuelSystem>,
    pub cellsize: f64,
    pub do_spotting: bool,
    pub realizations: usize,
    /// world (row, col) of local cell (0, 0)
    pub origin: (i64, i64),
    pub seed: Option<u64>,
    /// directory for freezing burned-out tiles to disk; `None` disables
    pub freeze_dir: Option<PathBuf>,
    pub ros_model: RosModel,
    pub moisture_model: MoistureModel,
    pub oob_mode: OobMode,
    /// `None` = available parallelism
    pub n_threads: Option<usize>,
}

impl PropagatorConfig {
    /// Config with default everything but the mandatory `veg`/`dem` grids
    /// (which must share a shape). Mutate the public fields to customize.
    pub fn new(veg: Grid2<i32>, dem: Grid2<f64>) -> PropagatorConfig {
        PropagatorConfig {
            veg,
            dem,
            fuels: None,
            cellsize: CELLSIZE_DEFAULT,
            do_spotting: false,
            realizations: REALIZATIONS_DEFAULT,
            origin: (0, 0),
            seed: None,
            freeze_dir: None,
            ros_model: RosModel::default(),
            moisture_model: MoistureModel::default(),
            oob_mode: OobMode::default(),
            n_threads: None,
        }
    }
}

/// Stochastic cellular wildfire spread simulator (see spec §1).
pub struct Propagator {
    veg: Grid2<i32>,
    dem: Grid2<f64>,
    fuels: FuelSystem,
    fuel_idx: Grid2<i32>,
    cellsize: f64,
    n_realizations: usize,
    do_spotting: bool,
    origin: (i64, i64),
    ros_model: RosModel,
    moisture_model: MoistureModel,
    oob_mode: OobMode,
    n_threads: usize,

    time: i64,
    moisture: Option<Grid2<f32>>,
    wind_dir: Option<Grid2<f32>>,
    wind_speed: Option<Grid2<f32>>,
    actions_moisture: Option<Grid2<f32>>,
    /// cells whose cached edges depend on `actions_moisture`: every cell
    /// with action moisture plus its 8 neighbours (edges read the target's
    /// moisture). Decay only invalidates these.
    actions_footprint: Vec<(usize, usize)>,
    scheduler: Scheduler,
    reals: Vec<Realization>,
    /// realization-independent spread quantities per cell edge, valid for
    /// the current vegetation / weather / action moisture
    edges: EdgeCache,

    store: Option<Arc<TileStore>>,
    // precomputed fold contribution of frozen tiles, one 32x32 block per
    // spatial position (world coords), aggregated across realizations
    fold_blocks: HashMap<(i64, i64), StateFold>,
    fold_dirty: HashSet<(i64, i64)>,
    fold_valid: bool,
    /// per-tile fuel masks of `veg` for the freeze/thaw floods; reset
    /// whenever the vegetation or the grid changes
    fuel_masks: OnceLock<FuelMasks>,
}

impl Propagator {
    /// Build a propagator from a config: validate the grids, resolve the
    /// fuel system (legacy table when unset; spotting stripped when
    /// disabled), pre-map vegetation codes to dense fuel indices, seed one
    /// RNG per realization, and open the freeze store if a `freeze_dir` was
    /// given. Time starts at 0 with no weather set.
    pub fn new(config: PropagatorConfig) -> Result<Propagator> {
        if config.veg.shape() != config.dem.shape() {
            return Err(PropagatorError::InvalidBoundaryConditions(
                "veg and dem must have the same shape".into(),
            ));
        }
        let mut fuels = config.fuels.unwrap_or_else(legacy_fuel_system);
        if !config.do_spotting {
            fuels.disable_spotting();
        }
        let fuel_idx = fuels.build_fuel_index_grid(&config.veg)?;
        let (rows, cols) = config.veg.shape();
        let reals = (0..config.realizations)
            .map(|r| {
                let rng = match config.seed {
                    Some(seed) => Rng::for_realization(seed, r as u64),
                    None => Rng::from_entropy(r as u64),
                };
                Realization::new(rows, cols, rng)
            })
            .collect();
        let store = match &config.freeze_dir {
            Some(dir) => Some(Arc::new(TileStore::open(dir)?)),
            None => None,
        };
        let n_threads = config.n_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
        Ok(Propagator {
            veg: config.veg,
            dem: config.dem,
            fuels,
            fuel_idx,
            cellsize: config.cellsize,
            n_realizations: config.realizations,
            do_spotting: config.do_spotting,
            origin: config.origin,
            ros_model: config.ros_model,
            moisture_model: config.moisture_model,
            oob_mode: config.oob_mode,
            n_threads,
            time: 0,
            moisture: None,
            wind_dir: None,
            wind_speed: None,
            actions_moisture: None,
            actions_footprint: Vec::new(),
            scheduler: Scheduler::new(),
            reals,
            edges: EdgeCache::for_realizations(rows, cols, config.realizations),
            store,
            fold_blocks: HashMap::new(),
            fold_dirty: HashSet::new(),
            fold_valid: true,
            fuel_masks: OnceLock::new(),
        })
    }

    // --- accessors ------------------------------------------------------

    /// Current simulation time, seconds from start.
    pub fn time(&self) -> i64 {
        self.time
    }

    /// Number of stochastic realizations.
    pub fn realizations(&self) -> usize {
        self.n_realizations
    }

    /// Current grid shape `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        self.veg.shape()
    }

    /// World (row, col) of local cell (0, 0); shifts as the domain grows.
    pub fn origin(&self) -> (i64, i64) {
        self.origin
    }

    /// Current vegetation-code grid (reflects any applied changes).
    pub fn veg(&self) -> &Grid2<i32> {
        &self.veg
    }

    /// Current boundary-hit behaviour.
    pub fn oob_mode(&self) -> OobMode {
        self.oob_mode
    }

    /// Change the boundary-hit behaviour mid-run.
    ///
    /// The point of this is to give up on growing: a caller that can no
    /// longer supply a larger grid switches to [`OobMode::Ignore`], and the
    /// next `step` pops the boundary events it had been suspending, so the
    /// fire clips at the edge and the rest of the front keeps burning.
    /// Without it a run built with [`OobMode::Raise`] whose growth stops is
    /// stuck forever: a suspended boundary event is the earliest event in its
    /// realization's heap, so nothing in that realization advances again.
    pub fn set_oob_mode(&mut self, mode: OobMode) {
        self.oob_mode = mode;
    }

    /// Tiles currently held by the shared edge cache (spread quantities
    /// per cell edge, see [`EdgeCache`]).
    pub fn edge_cache_tiles(&self) -> usize {
        self.edges.allocated_tiles()
    }

    /// Bytes currently held by the shared edge cache.
    pub fn edge_cache_bytes(&self) -> usize {
        self.edges.allocated_bytes()
    }

    /// Which domain edges the front is pressing against, as
    /// `(north, south, west, east)` — north/west are the low-row/low-col
    /// edges. A flag is set when some realization halted on that edge in
    /// `OobMode::Raise` (i.e. the last `step` returned
    /// `PropagatorError::OutOfBounds`); all-false otherwise. Lets the caller
    /// grow only the sides the fire actually reached.
    pub fn boundary_pressure(&self) -> (bool, bool, bool, bool) {
        let (n_rows, n_cols) = self.veg.shape();
        let (mut north, mut south, mut west, mut east) = (false, false, false, false);
        for real in &self.reals {
            if !real.halted_on_boundary {
                continue;
            }
            if let Some((row, col)) = real.heap.peek_cell() {
                north |= row <= 0;
                south |= row >= n_rows as i32 - 1;
                west |= col <= 0;
                east |= col >= n_cols as i32 - 1;
            }
        }
        (north, south, west, east)
    }

    /// Which edges have pending front events within `margin` cells.
    /// Unlike `boundary_pressure`, this is predictive and does not require
    /// propagation to have halted at the boundary first.
    pub fn boundary_proximity(&self, margin: usize) -> (bool, bool, bool, bool) {
        let (n_rows, n_cols) = self.veg.shape();
        let margin = margin.max(1) as i32;
        let south_limit = n_rows as i32 - margin;
        let east_limit = n_cols as i32 - margin;
        let (mut north, mut south, mut west, mut east) = (false, false, false, false);
        for real in &self.reals {
            for (row, col) in real.heap.event_cells() {
                north |= row < margin;
                south |= row >= south_limit;
                west |= col < margin;
                east |= col >= east_limit;
                if north && south && west && east {
                    return (north, south, west, east);
                }
            }
        }
        (north, south, west, east)
    }

    /// Inclusive world-cell bounds (row0, col0, row1, col1) of the grid.
    pub fn world_bounds(&self) -> (i64, i64, i64, i64) {
        let (rows, cols) = self.veg.shape();
        (
            self.origin.0,
            self.origin.1,
            self.origin.0 + rows as i64 - 1,
            self.origin.1 + cols as i64 - 1,
        )
    }

    /// Reseed every realization's RNG deterministically. Runs with the
    /// same seed are bitwise reproducible regardless of thread count.
    pub fn reseed(&mut self, seed: u64) {
        for (r, real) in self.reals.iter_mut().enumerate() {
            real.rng = Rng::for_realization(seed, r as u64);
        }
    }

    // --- boundary conditions ---------------------------------------------

    /// Enqueue boundary conditions (external units) at their time.
    pub fn set_boundary_conditions(&mut self, bc: BoundaryConditions) -> Result<()> {
        if bc.time < self.time {
            return Err(PropagatorError::InvalidBoundaryConditions(
                "boundary conditions cannot be applied in the past".into(),
            ));
        }
        let shape = self.veg.shape();
        let mut event = SchedulerEvent::default();

        if let Some(field) = &bc.moisture {
            // percent -> fraction
            event.moisture = Some(field.to_grid(shape, |v| v / 100.0)?);
        }
        if let Some(field) = &bc.wind_dir {
            // degrees -> radians
            event.wind_dir = Some(field.to_grid(shape, |v| v.to_radians())?);
        }
        if let Some(field) = &bc.wind_speed {
            event.wind_speed = Some(field.to_grid(shape, |v| v)?);
        }
        if let Some(grid) = &bc.additional_moisture {
            if grid.shape() != shape {
                return Err(PropagatorError::InvalidBoundaryConditions(
                    "additional_moisture shape mismatch".into(),
                ));
            }
            // percent -> fraction
            let data = grid.as_slice().iter().map(|v| v / 100.0).collect();
            event.additional_moisture = Some(Grid2::from_vec(shape.0, shape.1, data));
        }
        if let Some(grid) = &bc.vegetation_changes {
            if grid.shape() != shape {
                return Err(PropagatorError::InvalidBoundaryConditions(
                    "vegetation_changes shape mismatch".into(),
                ));
            }
            event.vegetation_changes = Some(grid.clone());
        }
        if let Some(ignitions) = &bc.ignitions {
            event.updates = self.ignitions_to_updates(ignitions)?;
        }

        self.scheduler.add_event(bc.time, event);
        Ok(())
    }

    fn ignitions_to_updates(&self, ignitions: &Ignitions) -> Result<UpdateBatch> {
        let (rows, cols) = self.veg.shape();
        let check = |row: usize, col: usize| -> Result<()> {
            if row >= rows || col >= cols {
                return Err(PropagatorError::InvalidBoundaryConditions(format!(
                    "ignition ({row}, {col}) outside grid {rows}x{cols}"
                )));
            }
            Ok(())
        };
        let mut updates = UpdateBatch::default();
        match ignitions {
            Ignitions::Points(points) => {
                for &(row, col) in points {
                    check(row, col)?;
                    for r in 0..self.n_realizations {
                        updates.push(row as i32, col as i32, r as u32, 0.0, 0.0);
                    }
                }
            }
            Ignitions::PointsWithRealization(points) => {
                for &(row, col, realization) in points {
                    check(row, col)?;
                    if realization >= self.n_realizations {
                        return Err(PropagatorError::InvalidBoundaryConditions(
                            format!("realization {realization} out of range"),
                        ));
                    }
                    updates.push(row as i32, col as i32, realization as u32, 0.0, 0.0);
                }
            }
            Ignitions::Mask(mask) => {
                if mask.shape() != self.veg.shape() {
                    return Err(PropagatorError::InvalidBoundaryConditions(
                        "ignition mask shape mismatch".into(),
                    ));
                }
                for row in 0..rows {
                    for col in 0..cols {
                        if mask[(row, col)] {
                            for r in 0..self.n_realizations {
                                updates.push(
                                    row as i32, col as i32, r as u32, 0.0, 0.0,
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(updates)
    }

    // --- stepping ---------------------------------------------------------

    /// Advance to the next scheduled time (boundary condition or front
    /// event, whichever is earlier). No-op when fully idle.
    pub fn step(&mut self) -> Result<()> {
        let next_bc = self.scheduler.next_time();
        let next_prop = self.next_front_time();
        let new_time = match (next_bc, next_prop) {
            (None, None) => return Ok(()),
            (Some(bc_time), None) => bc_time,
            (None, Some(prop_time)) => prop_time,
            (Some(bc_time), Some(prop_time)) => bc_time.min(prop_time),
        };

        if next_bc == Some(new_time) {
            let (bc_time, event) = self.scheduler.pop().unwrap();
            let time_delta = bc_time - self.time;
            self.update_boundary_conditions(time_delta, &event);
            self.update_vegetation(&event)?;
            self.schedule_ignitions(bc_time, &event.updates)?;
        } else if new_time > self.time {
            self.decay_actions_moisture(new_time - self.time);
        }

        self.propagate_until(new_time)?;
        self.time = new_time;
        Ok(())
    }

    /// Propagate for `seconds` of simulation time, applying every
    /// scheduler event that falls inside the window at its exact time.
    pub fn step_window(&mut self, seconds: i64) -> Result<()> {
        if seconds < 0 {
            return Err(PropagatorError::InvalidBoundaryConditions(
                "seconds must be non-negative".into(),
            ));
        }
        let target_time = self.time + seconds;
        loop {
            let next_bc = self.scheduler.next_time();
            let segment_end = match next_bc {
                Some(bc_time) if bc_time <= target_time => bc_time,
                _ => target_time,
            };

            if segment_end > self.time {
                self.decay_actions_moisture(segment_end - self.time);
            }
            self.propagate_until(segment_end)?;

            match next_bc {
                Some(bc_time) if bc_time <= target_time => {
                    let (bc_time, event) = self.scheduler.pop().unwrap();
                    debug_assert_eq!(bc_time, segment_end);
                    self.update_boundary_conditions(0, &event);
                    self.update_vegetation(&event)?;
                    self.schedule_ignitions(bc_time, &event.updates)?;
                    self.time = bc_time;
                }
                _ => {
                    self.time = segment_end;
                    return Ok(());
                }
            }
        }
    }

    /// Next simulation time (scheduler or front event); `None` when idle.
    pub fn next_time(&self) -> Option<i64> {
        match (self.scheduler.next_time(), self.next_front_time()) {
            (None, None) => None,
            (Some(t), None) | (None, Some(t)) => Some(t),
            (Some(a), Some(b)) => Some(a.min(b)),
        }
    }

    fn next_front_time(&self) -> Option<i64> {
        self.reals
            .iter()
            .filter_map(|real| real.heap.peek_time())
            .min()
    }

    fn propagate_until(&mut self, end_time: i64) -> Result<()> {
        if self.reals.iter().all(|real| real.heap.is_empty()) {
            return Ok(());
        }
        let moisture = self.effective_moisture()?;
        let wind_dir = self
            .wind_dir
            .as_ref()
            .ok_or(PropagatorError::WeatherNotSet)?;
        let wind_speed = self
            .wind_speed
            .as_ref()
            .ok_or(PropagatorError::WeatherNotSet)?;
        let halt_on_boundary = self.oob_mode == OobMode::Raise;

        // fresh flags for this kernel run (mirrors the numba out_of_bounds
        // array being zeroed per call)
        for real in &mut self.reals {
            real.out_of_bounds = false;
            real.halted_on_boundary = false;
        }

        let shared = SharedInputs {
            veg: &self.veg,
            fuel_idx: &self.fuel_idx,
            dem: &self.dem,
            moisture: &moisture,
            wind_dir,
            wind_speed,
            fuels: &self.fuels,
            cellsize: self.cellsize,
            ros_model: self.ros_model,
            moisture_model: self.moisture_model,
            do_spotting: self.do_spotting,
            edges: &self.edges,
        };
        advance_all(
            &mut self.reals,
            &shared,
            end_time,
            halt_on_boundary,
            self.n_threads,
        );

        if self.oob_mode == OobMode::Raise
            && self.reals.iter().any(|real| real.out_of_bounds)
        {
            return Err(PropagatorError::OutOfBounds);
        }
        Ok(())
    }

    fn update_boundary_conditions(&mut self, time_delta: i64, event: &SchedulerEvent) {
        if time_delta > 0 {
            self.decay_actions_moisture(time_delta);
        }
        if event.moisture.is_some() || event.wind_dir.is_some() || event.wind_speed.is_some() {
            self.edges.invalidate();
        }
        if let Some(moisture) = &event.moisture {
            self.moisture = Some(moisture.clone());
        }
        if let Some(additional) = &event.additional_moisture {
            let (rows, cols) = self.veg.shape();
            let actions = self
                .actions_moisture
                .get_or_insert_with(|| Grid2::filled(rows, cols, 0.0));
            for (a, b) in actions
                .as_mut_slice()
                .iter_mut()
                .zip(additional.as_slice().iter())
            {
                *a = (*a + *b).clamp(0.0, 1.0);
            }
            // action moisture only ever accumulates, so the new footprint
            // covers every cell whose edges changed
            self.rebuild_actions_footprint();
            self.edges
                .invalidate_cells(self.actions_footprint.iter().copied());
        }
        if let Some(wind_dir) = &event.wind_dir {
            self.wind_dir = Some(wind_dir.clone());
        }
        if let Some(wind_speed) = &event.wind_speed {
            self.wind_speed = Some(wind_speed.clone());
        }
    }

    fn update_vegetation(&mut self, event: &SchedulerEvent) -> Result<()> {
        let Some(changes) = &event.vegetation_changes else {
            return Ok(());
        };
        // vegetation changes can add fuel to burned-out areas, which
        // invalidates the freeze criterion: bring everything back
        self.thaw_all()?;
        for (cell, change) in self
            .veg
            .as_mut_slice()
            .iter_mut()
            .zip(changes.as_slice().iter())
        {
            if !change.is_nan() {
                *cell = *change as i32;
            }
        }
        self.fuel_idx = self.fuels.build_fuel_index_grid(&self.veg)?;
        self.fuel_masks = OnceLock::new();
        self.edges.invalidate();
        Ok(())
    }

    fn schedule_ignitions(&mut self, time: i64, updates: &UpdateBatch) -> Result<()> {
        // a new ignition seeds spread through areas that may have been
        // frozen as unreachable: bring its whole component back, with one
        // multi-seed flood per realization for the whole batch
        if self.store.as_ref().is_some_and(|store| !store.is_empty()) {
            let mut seeds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.n_realizations];
            for i in 0..updates.len() {
                seeds[updates.realizations[i] as usize]
                    .push((updates.rows[i] as usize, updates.cols[i] as usize));
            }
            self.thaw_for_ignitions(&seeds)?;
        }
        // push order unchanged: heap layout and RNG use are as before
        for i in 0..updates.len() {
            self.reals[updates.realizations[i] as usize].heap.push(
                time,
                updates.rows[i],
                updates.cols[i],
                updates.ros[i],
                updates.fli[i],
            );
        }
        Ok(())
    }

    fn decay_actions_moisture(&mut self, time_delta: i64) {
        let Some(actions) = &mut self.actions_moisture else {
            return;
        };
        let elapsed_minutes = (time_delta as f64 / 60.0).max(0.0);
        if elapsed_minutes == 0.0 {
            return;
        }
        let factor = (1.0 - ACTIONS_MOISTURE_DECAY).powf(elapsed_minutes) as f32;
        for value in actions.as_mut_slice() {
            *value *= factor;
        }
        // only edges into cells with action moisture changed
        self.edges
            .invalidate_cells(self.actions_footprint.iter().copied());
    }

    /// Recompute `actions_footprint` from `actions_moisture`: the cells with
    /// action moisture, dilated by one cell in every direction.
    fn rebuild_actions_footprint(&mut self) {
        self.actions_footprint.clear();
        let Some(actions) = &self.actions_moisture else {
            return;
        };
        let (rows, cols) = actions.shape();
        let mut marked = vec![false; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                if actions[(row, col)] == 0.0 {
                    continue;
                }
                for dr in -1i64..=1 {
                    for dc in -1i64..=1 {
                        let r = row as i64 + dr;
                        let c = col as i64 + dc;
                        if r >= 0 && c >= 0 && (r as usize) < rows && (c as usize) < cols {
                            marked[r as usize * cols + c as usize] = true;
                        }
                    }
                }
            }
        }
        self.actions_footprint = marked
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| (i / cols, i % cols))
            .collect();
    }

    /// Base moisture plus action-derived increments, clipped to [0, 1].
    fn effective_moisture(&self) -> Result<Grid2<f32>> {
        let moisture = self
            .moisture
            .as_ref()
            .ok_or(PropagatorError::WeatherNotSet)?;
        match &self.actions_moisture {
            None => Ok(moisture.clone()),
            Some(actions) => {
                let data = moisture
                    .as_slice()
                    .iter()
                    .zip(actions.as_slice().iter())
                    .map(|(m, a)| (m + a).clamp(0.0, 1.0))
                    .collect();
                let (rows, cols) = moisture.shape();
                Ok(Grid2::from_vec(rows, cols, data))
            }
        }
    }

    // --- outputs -----------------------------------------------------------

    /// Aggregate the tiled state (live + frozen) into per-cell maps.
    fn fold_state(&mut self) -> Result<StateFold> {
        let (rows, cols) = self.veg.shape();
        let mut fold = StateFold::empty(rows, cols);
        for real in &self.reals {
            fold_realization(&mut fold, &real.tiles);
        }
        self.fold_frozen(&mut fold)?;
        Ok(fold)
    }

    /// Assemble the current outputs and summary stats (single fold pass).
    pub fn get_output(&mut self) -> Result<PropagatorOutput> {
        let fold = self.fold_state()?;
        let n = self.n_realizations as f32;
        let (rows, cols) = self.veg.shape();
        let cells = rows * cols;

        let probability = |count: &Grid2<i32>| -> Grid2<f32> {
            let data = count.as_slice().iter().map(|c| *c as f32 / n).collect();
            Grid2::from_vec(rows, cols, data)
        };
        let masked_mean = |sum: &Grid2<f64>, count: &Grid2<i32>| -> Grid2<f32> {
            let mut data = Vec::with_capacity(cells);
            for (s, c) in sum.as_slice().iter().zip(count.as_slice().iter()) {
                data.push(if *c > 0 { (*s / *c as f64) as f32 } else { f32::NAN });
            }
            Grid2::from_vec(rows, cols, data)
        };

        let fire_probability = probability(&fold.count);
        let mut min_arrival = Vec::with_capacity(cells);
        for (arrival, count) in fold
            .arrival_min
            .as_slice()
            .iter()
            .zip(fold.count.as_slice().iter())
        {
            min_arrival.push(if *count == 0 { 0.0 } else { *arrival as f32 });
        }
        let stats = self.compute_stats(&fire_probability);

        Ok(PropagatorOutput {
            time: self.time,
            spotting_generation_probability: probability(&fold.spot_gen_count),
            spotting_receiving_probability: probability(&fold.spot_recv_count),
            min_arrival_time: Grid2::from_vec(rows, cols, min_arrival),
            mean_arrival_time: masked_mean(&fold.arrival_sum, &fold.count),
            ros_mean: masked_mean(&fold.ros_sum, &fold.count),
            ros_max: fold.ros_max,
            fli_mean: masked_mean(&fold.fli_sum, &fold.count),
            flame_height_max: fold.flame_max,
            flame_height_mean: masked_mean(&fold.flame_sum, &fold.count),
            radiant_reach_max: fold.radiant_max,
            radiant_reach_mean: masked_mean(&fold.radiant_sum, &fold.count),
            fli_max: fold.fli_max,
            fire_probability,
            stats,
        })
    }

    /// Mean burn probability across realizations for each cell.
    pub fn compute_fire_probability(&mut self) -> Result<Grid2<f32>> {
        let fold = self.fold_state()?;
        let n = self.n_realizations as f32;
        let (rows, cols) = self.veg.shape();
        let data = fold
            .count
            .as_slice()
            .iter()
            .map(|c| *c as f32 / n)
            .collect();
        Ok(Grid2::from_vec(rows, cols, data))
    }

    /// Simple area-based stats and number of active realizations.
    pub fn compute_stats(&self, probability: &Grid2<f32>) -> PropagatorStats {
        let n_active = self
            .reals
            .iter()
            .filter(|real| !real.heap.is_empty())
            .count();
        let cell_area = self.cellsize * self.cellsize;
        let mut area_mean = 0.0f64;
        let mut area_50 = 0.0f64;
        let mut area_75 = 0.0f64;
        let mut area_90 = 0.0f64;
        for &p in probability.as_slice() {
            let p = p as f64;
            area_mean += p;
            if p >= 0.5 {
                area_50 += 1.0;
            }
            if p >= 0.75 {
                area_75 += 1.0;
            }
            if p >= 0.90 {
                area_90 += 1.0;
            }
        }
        PropagatorStats {
            n_active,
            area_mean: area_mean * cell_area,
            area_50: area_50 * cell_area,
            area_75: area_75 * cell_area,
            area_90: area_90 * cell_area,
        }
    }

    /// Dense per-realization materialization of one tile component.
    fn materialize<T: Clone + 'static>(
        &self,
        zero: T,
        extract: impl Fn(&Tile, usize) -> T,
    ) -> Result<Vec<Grid2<T>>> {
        let (rows, cols) = self.veg.shape();
        let mut out: Vec<Grid2<T>> = (0..self.n_realizations)
            .map(|_| Grid2::filled(rows, cols, zero.clone()))
            .collect();
        for (r, real) in self.reals.iter().enumerate() {
            for (tile_row, tile_col, slot) in real.tiles.allocated() {
                let tile = real.tiles.tile(slot);
                scatter_tile(&mut out[r], tile, tile_row, tile_col, &extract);
            }
        }
        if let Some(store) = &self.store {
            for key in store.keys() {
                let tile = store.read(&key)?;
                let (tile_row, tile_col) = self.tile_local_pos(&key);
                scatter_tile(
                    &mut out[key.0 as usize],
                    &tile,
                    tile_row,
                    tile_col,
                    &extract,
                );
            }
        }
        Ok(out)
    }

    /// Dense per-realization burn state (0/1). Analysis/tests, not hot path.
    pub fn get_fire(&self) -> Result<Vec<Grid2<u8>>> {
        self.materialize(0u8, |tile, local| {
            (tile.flags[local] & FLAG_FIRE != 0) as u8
        })
    }

    /// Dense per-realization spotting generation mask.
    pub fn get_spotting_generation(&self) -> Result<Vec<Grid2<u8>>> {
        self.materialize(0u8, |tile, local| {
            (tile.flags[local] & FLAG_SPOT_GEN != 0) as u8
        })
    }

    /// Dense per-realization spotting receiving mask.
    pub fn get_spotting_receiving(&self) -> Result<Vec<Grid2<u8>>> {
        self.materialize(0u8, |tile, local| {
            (tile.flags[local] & FLAG_SPOT_RECV != 0) as u8
        })
    }

    /// Dense per-realization arrival times (0 where never burnt).
    pub fn get_arrival_time(&self) -> Result<Vec<Grid2<i32>>> {
        self.materialize(0i32, |tile, local| tile.arrival[local])
    }

    /// Dense per-realization rate of spread (0 where never burnt).
    pub fn get_ros(&self) -> Result<Vec<Grid2<f32>>> {
        self.materialize(0.0f32, |tile, local| tile.ros[local])
    }

    /// Dense per-realization fireline intensity (0 where never burnt).
    pub fn get_fireline_int(&self) -> Result<Vec<Grid2<f32>>> {
        self.materialize(0.0f32, |tile, local| tile.fli[local])
    }

    // --- tile freezing ------------------------------------------------------

    fn tile_world_key(
        &self,
        realization: usize,
        tile_row: usize,
        tile_col: usize,
    ) -> TileKey {
        (
            realization as u32,
            self.origin.0 + ((tile_row << TILE_SHIFT) as i64),
            self.origin.1 + ((tile_col << TILE_SHIFT) as i64),
        )
    }

    fn tile_local_pos(&self, key: &TileKey) -> (usize, usize) {
        (
            ((key.1 - self.origin.0) >> TILE_SHIFT) as usize,
            ((key.2 - self.origin.1) >> TILE_SHIFT) as usize,
        )
    }

    /// Per-tile fuel masks of the current vegetation (built on first use,
    /// dropped whenever the vegetation or the grid changes).
    fn fuel_masks(&self) -> &FuelMasks {
        self.fuel_masks.get_or_init(|| FuelMasks::build(&self.veg))
    }

    /// Freeze burned-out tiles to disk and release their memory.
    ///
    /// A tile is frozen (per realization) when propagation can never
    /// touch it again. With spotting disabled that holds for any tile
    /// with no *reachable* unburnt fuel cell (sparse flood, see
    /// [`crate::reach`]: it may only over-approximate reachability); with
    /// spotting enabled the stricter "every cell burnt or fuel-free"
    /// criterion applies. Freezing changes neither the dynamics nor the
    /// RNG streams; seeded runs stay bitwise identical. Returns the number
    /// of tiles frozen.
    pub fn freeze_inactive_tiles(&mut self) -> Result<usize> {
        let store = self
            .store
            .clone()
            .ok_or(PropagatorError::FreezeDirNotSet)?;
        let do_spotting = self.do_spotting;
        let fuel = self.fuel_masks();
        // classification is read-only and independent per realization
        let decisions = map_realizations(&self.reals, self.n_threads, |_, real| {
            freezable_tiles(real, fuel, do_spotting)
        });

        // writes and fold accumulation stay sequential, in realization
        // then tile-position order, exactly as before
        let mut frozen_total = 0usize;
        for (r, to_freeze) in decisions.into_iter().enumerate() {
            if to_freeze.is_empty() {
                continue;
            }
            for &(tile_row, tile_col, slot, _) in &to_freeze {
                let key = self.tile_world_key(r, tile_row, tile_col);
                let tile = self.reals[r].tiles.tile(slot);
                store.freeze(key, tile)?;
                fold_cache_add(
                    &mut self.fold_blocks,
                    &self.fold_dirty,
                    self.fold_valid,
                    &key,
                    tile,
                );
            }
            frozen_total += to_freeze.len();
            self.reals[r].tiles.freeze_tiles(to_freeze);
        }
        if frozen_total > 0 {
            self.release_dead_edge_tiles();
        }
        Ok(frozen_total)
    }

    /// Count, without freezing anything, how many allocated (live) tiles
    /// each freeze criterion would freeze right now. With spotting enabled
    /// `blocked_by_strict` is the number of tiles kept live only because
    /// spotting uses the strict "every fuel cell burnt" criterion instead
    /// of reachability. Needs no freeze store.
    pub fn freeze_diagnostic(&self) -> FreezeDiagnostic {
        let fuel = self.fuel_masks();
        let per_real = map_realizations(&self.reals, self.n_threads, |_, real| {
            let strict: HashSet<(usize, usize)> = freezable_tiles(real, fuel, true)
                .into_iter()
                .map(|(row, col, _, _)| (row, col))
                .collect();
            let reachability = freezable_tiles(real, fuel, false);
            let allocated = real.tiles.allocated().count();
            let blocked = reachability
                .iter()
                .filter(|t| !strict.contains(&(t.0, t.1)))
                .count();
            (allocated, strict.len(), reachability.len(), blocked)
        });
        let mut diagnostic = FreezeDiagnostic::default();
        for (allocated, strict, reachability, blocked) in per_real {
            diagnostic.allocated += allocated;
            diagnostic.strict_freezable += strict;
            diagnostic.reachability_freezable += reachability;
            diagnostic.blocked_by_strict += blocked;
        }
        diagnostic
    }

    /// Free cached edges of blocks no realization holds a live tile for:
    /// frozen everywhere (never popped again) or not yet reached (cheap to
    /// refill on first touch).
    fn release_dead_edge_tiles(&mut self) {
        let (tiles_h, tiles_w) = self.edges.tiles_shape();
        for tile_row in 0..tiles_h {
            for tile_col in 0..tiles_w {
                let live = self
                    .reals
                    .iter()
                    .any(|real| real.tiles.slot_of(tile_row, tile_col) >= 0);
                if !live {
                    self.edges.release_tile(tile_row, tile_col);
                }
            }
        }
    }

    /// Load a frozen tile back into the pool.
    fn thaw_tile(&mut self, realization: usize, tile_row: usize, tile_col: usize) -> Result<()> {
        let store = self.store.clone().expect("thaw without a tile store");
        let key = self.tile_world_key(realization, tile_row, tile_col);
        let tile = store.thaw(&key)?;
        self.reals[realization]
            .tiles
            .insert_tile(tile_row, tile_col, tile);
        self.fold_cache_mark_dirty(&key);
        Ok(())
    }

    /// Load every frozen tile back into memory; returns the count.
    pub fn thaw_all(&mut self) -> Result<usize> {
        let Some(store) = self.store.clone() else {
            return Ok(0);
        };
        let keys = store.keys();
        for key in &keys {
            let (tile_row, tile_col) = self.tile_local_pos(key);
            self.thaw_tile(key.0 as usize, tile_row, tile_col)?;
        }
        Ok(keys.len())
    }

    /// Thaw every frozen tile new ignitions could reach (`seeds[r]` are
    /// realization `r`'s ignition cells): their whole unburnt-fuel
    /// components must be live again for the kernel to propagate through
    /// them. One sparse flood per realization, from the in-RAM frozen
    /// summaries: solid tiles are never reached, only tiles that actually
    /// thaw are read from disk, and an ignition's own tile always thaws.
    fn thaw_for_ignitions(&mut self, seeds: &[Vec<(usize, usize)>]) -> Result<()> {
        if self.store.is_none() {
            return Ok(());
        }
        let fuel = self.fuel_masks();
        let to_thaw = map_realizations(&self.reals, self.n_threads, |r, real| {
            let cells = &seeds[r];
            if cells.is_empty() || real.tiles.frozen_count() == 0 {
                return Vec::new();
            }
            let mut reach = Reach::build(&real.tiles, fuel);
            for &(row, col) in cells {
                reach.seed_cell(row, col);
            }
            reach.run();
            let targets: HashSet<(usize, usize)> = cells
                .iter()
                .map(|&(row, col)| (row >> TILE_SHIFT, col >> TILE_SHIFT))
                .collect();
            real.tiles
                .frozen_positions()
                .filter(|&(tile_row, tile_col)| {
                    targets.contains(&(tile_row, tile_col)) || reach.reached(tile_row, tile_col)
                })
                .collect::<Vec<_>>()
        });
        for (realization, positions) in to_thaw.into_iter().enumerate() {
            for (tile_row, tile_col) in positions {
                self.thaw_tile(realization, tile_row, tile_col)?;
            }
        }
        Ok(())
    }

    /// Thaw frozen tiles that straddled the old south/east domain edge
    /// and gain in-domain cells when the grid grows: their summaries (and
    /// the kernel, which reads frozen tiles as burnt) know nothing of the
    /// new cells. `old_shape` is in the old grid, the shift places it in
    /// the new one.
    fn thaw_grown_edge_tiles(
        &mut self,
        old_shape: (usize, usize),
        row_shift: usize,
        col_shift: usize,
    ) -> Result<()> {
        if self.store.is_none() {
            return Ok(());
        }
        let (rows, cols) = self.veg.shape();
        let (tiles_h, tiles_w) = self.reals[0].tiles.tiles_shape();
        let mut positions: Vec<(usize, usize)> = Vec::new();
        if !old_shape.0.is_multiple_of(TILE_SIZE) && row_shift + old_shape.0 < rows {
            let tile_row = (row_shift + old_shape.0) >> TILE_SHIFT;
            positions.extend((0..tiles_w).map(|tile_col| (tile_row, tile_col)));
        }
        if !old_shape.1.is_multiple_of(TILE_SIZE) && col_shift + old_shape.1 < cols {
            let tile_col = (col_shift + old_shape.1) >> TILE_SHIFT;
            positions.extend((0..tiles_h).map(|tile_row| (tile_row, tile_col)));
        }
        for realization in 0..self.n_realizations {
            for &(tile_row, tile_col) in &positions {
                if self.reals[realization].tiles.slot_of(tile_row, tile_col) == FROZEN_TILE {
                    self.thaw_tile(realization, tile_row, tile_col)?;
                }
            }
        }
        Ok(())
    }

    // --- frozen-fold cache ---------------------------------------------------

    /// A record left this position (thaw): min/max aggregates cannot be
    /// subtracted, so the whole position is rebuilt lazily.
    fn fold_cache_mark_dirty(&mut self, key: &TileKey) {
        if self.fold_valid {
            self.fold_dirty.insert((key.1, key.2));
        }
    }

    /// Drop the cache entirely (restore swapped the frozen index).
    fn fold_cache_invalidate(&mut self) {
        self.fold_blocks.clear();
        self.fold_dirty.clear();
        self.fold_valid = false;
    }

    /// Rebuild invalid or dirty cache blocks from the tile store.
    fn ensure_fold_cache(&mut self) -> Result<()> {
        let Some(store) = self.store.clone() else {
            return Ok(());
        };
        if self.fold_valid && self.fold_dirty.is_empty() {
            return Ok(());
        }
        let rebuild: Option<HashSet<(i64, i64)>> = if self.fold_valid {
            let dirty = std::mem::take(&mut self.fold_dirty);
            for position in &dirty {
                self.fold_blocks.remove(position);
            }
            Some(dirty)
        } else {
            self.fold_blocks.clear();
            None // rebuild everything
        };
        for key in store.keys() {
            let position = (key.1, key.2);
            if let Some(rebuild) = &rebuild {
                if !rebuild.contains(&position) {
                    continue;
                }
            }
            let tile = store.read(&key)?;
            let block = self
                .fold_blocks
                .entry(position)
                .or_insert_with(|| StateFold::empty(TILE_SIZE, TILE_SIZE));
            block.accumulate_tile(&tile, 0, 0);
        }
        self.fold_dirty.clear();
        self.fold_valid = true;
        Ok(())
    }

    /// Merge the cached frozen-tile aggregates into a fold. Steady state
    /// costs no disk I/O: blocks are maintained incrementally at freeze
    /// time and only dirty/invalidated positions re-read the store.
    fn fold_frozen(&mut self, fold: &mut StateFold) -> Result<()> {
        if self.store.is_none() {
            return Ok(());
        }
        self.ensure_fold_cache()?;
        for ((world_row, world_col), block) in &self.fold_blocks {
            let row0 = (world_row - self.origin.0) as usize;
            let col0 = (world_col - self.origin.1) as usize;
            fold.merge_block(block, row0, col0);
        }
        Ok(())
    }

    // --- checkpointing and domain growth --------------------------------------

    /// Snapshot the full dynamic state as an immutable checkpoint.
    pub fn checkpoint(&self) -> PropagatorCheckpoint {
        let frozen_index = self
            .store
            .as_ref()
            .map(|store| store.snapshot_index())
            .unwrap_or_default();
        let frozen_source = match &self.store {
            Some(store) if !frozen_index.is_empty() => Some(store.clone()),
            _ => None,
        };
        PropagatorCheckpoint {
            version: CHECKPOINT_VERSION,
            time: self.time,
            origin: self.origin,
            cellsize: self.cellsize,
            realizations: self.n_realizations,
            do_spotting: self.do_spotting,
            veg: self.veg.clone(),
            dem: self.dem.clone(),
            moisture: self.moisture.clone(),
            wind_dir: self.wind_dir.clone(),
            wind_speed: self.wind_speed.clone(),
            actions_moisture: self.actions_moisture.clone(),
            heaps: self.reals.iter().map(|real| real.heap.clone()).collect(),
            tiles: self.reals.iter().map(|real| real.tiles.clone()).collect(),
            scheduler_events: self.scheduler.snapshot(),
            frozen_index,
            frozen_source,
        }
    }

    /// Roll this propagator back to a checkpoint taken on the same grid.
    /// The checkpoint stays valid and can be restored again.
    pub fn restore(&mut self, checkpoint: &PropagatorCheckpoint) -> Result<()> {
        if checkpoint.shape() != self.veg.shape() {
            return Err(PropagatorError::CheckpointMismatch(format!(
                "checkpoint grid {:?} does not match this propagator's grid \
                 {:?}; use from_checkpoint to change the domain",
                checkpoint.shape(),
                self.veg.shape()
            )));
        }
        if checkpoint.origin != self.origin {
            return Err(PropagatorError::CheckpointMismatch(format!(
                "checkpoint origin {:?} does not match {:?}",
                checkpoint.origin, self.origin
            )));
        }
        self.veg = checkpoint.veg.clone();
        self.fuel_idx = self.fuels.build_fuel_index_grid(&self.veg)?;
        self.fuel_masks = OnceLock::new();
        self.load_state(checkpoint, 0, 0)
    }

    /// Build a propagator resuming from a checkpoint, optionally on a
    /// new (typically larger) grid.
    pub fn from_checkpoint(
        checkpoint: &PropagatorCheckpoint,
        options: ResumeOptions,
    ) -> Result<Propagator> {
        let (veg, dem, origin, row_shift, col_shift) = match options.domain {
            None => (
                checkpoint.veg.clone(),
                checkpoint.dem.clone(),
                checkpoint.origin,
                0usize,
                0usize,
            ),
            Some((veg, dem, origin)) => {
                let (row_shift, col_shift) = growth_shifts(
                    checkpoint.origin,
                    checkpoint.shape(),
                    &veg,
                    &dem,
                    origin,
                )?;
                // vegetation in the overlap comes from the checkpoint,
                // preserving past vegetation_changes
                let mut veg = veg;
                veg.blit(&checkpoint.veg, row_shift, col_shift);
                (veg, dem, origin, row_shift, col_shift)
            }
        };

        let mut config = PropagatorConfig::new(veg, dem);
        config.fuels = options.fuels;
        config.cellsize = checkpoint.cellsize;
        config.do_spotting = checkpoint.do_spotting;
        config.realizations = checkpoint.realizations;
        config.origin = origin;
        config.seed = options.seed;
        config.freeze_dir = options.freeze_dir;
        if let Some(model) = options.ros_model {
            config.ros_model = model;
        }
        if let Some(model) = options.moisture_model {
            config.moisture_model = model;
        }
        config.oob_mode = options.oob_mode;
        config.n_threads = options.n_threads;

        let mut propagator = Propagator::new(config)?;
        propagator.load_state(checkpoint, row_shift, col_shift)?;
        Ok(propagator)
    }

    /// Grow the domain in place, preserving all simulation state. This
    /// is the cheap growth path: only the small tile-index grids are
    /// re-anchored and 2D fields padded. The new grid must contain the
    /// old one and north/west growth must be a multiple of TILE_SIZE
    /// cells. Vegetation in the overlap keeps the current (possibly
    /// mutated) values; weather pads by edge replication until fresh
    /// boundary conditions arrive.
    pub fn expand(
        &mut self,
        veg: Grid2<i32>,
        dem: Grid2<f64>,
        origin: (i64, i64),
    ) -> Result<()> {
        let old_shape = self.veg.shape();
        let (row_shift, col_shift) =
            growth_shifts(self.origin, old_shape, &veg, &dem, origin)?;
        let new_shape = veg.shape();

        let mut veg = veg;
        veg.blit(&self.veg, row_shift, col_shift);
        self.veg = veg;
        self.dem = dem;
        self.fuel_idx = self.fuels.build_fuel_index_grid(&self.veg)?;
        self.fuel_masks = OnceLock::new();
        self.origin = origin;
        self.edges.reset(new_shape.0, new_shape.1);

        let tile_row0 = row_shift >> TILE_SHIFT;
        let tile_col0 = col_shift >> TILE_SHIFT;
        for real in &mut self.reals {
            real.heap.shift_coords(row_shift as i32, col_shift as i32);
            real.tiles
                .reanchor(new_shape.0, new_shape.1, tile_row0, tile_col0);
        }
        self.thaw_grown_edge_tiles(old_shape, row_shift, col_shift)?;

        for field in [&mut self.moisture, &mut self.wind_dir, &mut self.wind_speed] {
            if let Some(grid) = field.take() {
                *field = Some(grid.pad_edge(new_shape, row_shift, col_shift));
            }
        }
        if let Some(grid) = self.actions_moisture.take() {
            self.actions_moisture =
                Some(grid.pad_const(new_shape, row_shift, col_shift, 0.0));
        }
        self.rebuild_actions_footprint();

        for (_, event) in self.scheduler.events_mut() {
            reanchor_event(event, row_shift, col_shift, old_shape, new_shape);
        }
        Ok(())
    }

    /// Replace all dynamic state with the checkpoint's, re-anchored by
    /// (row_shift, col_shift) into this propagator's grid.
    fn load_state(
        &mut self,
        checkpoint: &PropagatorCheckpoint,
        row_shift: usize,
        col_shift: usize,
    ) -> Result<()> {
        if checkpoint.version > CHECKPOINT_VERSION {
            return Err(PropagatorError::CheckpointMismatch(format!(
                "checkpoint version {} is newer than supported version {}",
                checkpoint.version, CHECKPOINT_VERSION
            )));
        }
        if checkpoint.realizations != self.n_realizations {
            return Err(PropagatorError::CheckpointMismatch(
                "realizations mismatch".into(),
            ));
        }
        if checkpoint.do_spotting != self.do_spotting {
            return Err(PropagatorError::CheckpointMismatch(
                "do_spotting mismatch".into(),
            ));
        }
        if checkpoint.cellsize != self.cellsize {
            return Err(PropagatorError::CheckpointMismatch(
                "cellsize mismatch".into(),
            ));
        }

        let old_shape = checkpoint.shape();
        let new_shape = self.veg.shape();
        let grew = old_shape != new_shape || row_shift != 0 || col_shift != 0;
        self.time = checkpoint.time;
        self.edges.reset(new_shape.0, new_shape.1);

        let tile_row0 = row_shift >> TILE_SHIFT;
        let tile_col0 = col_shift >> TILE_SHIFT;
        for (r, real) in self.reals.iter_mut().enumerate() {
            real.heap = checkpoint.heaps[r].clone();
            real.heap.shift_coords(row_shift as i32, col_shift as i32);
            real.tiles = checkpoint.tiles[r].clone();
            if grew {
                real.tiles
                    .reanchor(new_shape.0, new_shape.1, tile_row0, tile_col0);
            }
            real.out_of_bounds = false;
            real.halted_on_boundary = false;
        }

        // weather grows by edge replication, action moisture with zeros
        for (target, source) in [
            (&mut self.moisture, &checkpoint.moisture),
            (&mut self.wind_dir, &checkpoint.wind_dir),
            (&mut self.wind_speed, &checkpoint.wind_speed),
        ] {
            *target = source.as_ref().map(|grid| {
                if grew {
                    grid.pad_edge(new_shape, row_shift, col_shift)
                } else {
                    grid.clone()
                }
            });
        }
        self.actions_moisture = checkpoint.actions_moisture.as_ref().map(|grid| {
            if grew {
                grid.pad_const(new_shape, row_shift, col_shift, 0.0)
            } else {
                grid.clone()
            }
        });
        self.rebuild_actions_footprint();

        self.scheduler = Scheduler::new();
        for (event_time, event) in &checkpoint.scheduler_events {
            let mut event = event.clone();
            reanchor_event(&mut event, row_shift, col_shift, old_shape, new_shape);
            self.scheduler.add_event(*event_time, event);
        }

        self.fuel_masks = OnceLock::new();
        self.restore_frozen(checkpoint)?;
        if grew {
            self.thaw_grown_edge_tiles(old_shape, row_shift, col_shift)?;
        }
        Ok(())
    }

    /// Re-attach a checkpoint's frozen tiles to this propagator: same
    /// session store — roll the index back (records are append-only);
    /// other store — stream-copy the records in; no store — load the
    /// records straight into the tile pools instead.
    fn restore_frozen(&mut self, checkpoint: &PropagatorCheckpoint) -> Result<()> {
        self.fold_cache_invalidate();
        if checkpoint.frozen_index.is_empty() {
            if let Some(store) = &self.store {
                store.restore_index(Default::default());
            }
            return Ok(());
        }
        let source = checkpoint.frozen_source.as_ref().ok_or_else(|| {
            PropagatorError::CheckpointMismatch(
                "checkpoint references frozen tiles but has no source".into(),
            )
        })?;

        if let Some(store) = self.store.clone() {
            if Arc::ptr_eq(&store, source) {
                store.restore_index(checkpoint.frozen_index.clone());
                return Ok(());
            }
            store.restore_index(Default::default());
            for (key, offset) in &checkpoint.frozen_index {
                let payload = source.read_bytes(*offset)?;
                store.write_record(*key, &payload)?;
            }
            return Ok(());
        }

        // no store: materialize the frozen tiles into the pools
        for (key, offset) in &checkpoint.frozen_index {
            let tile = record_to_tile(&source.read_bytes(*offset)?);
            let (tile_row, tile_col) = self.tile_local_pos(key);
            self.reals[key.0 as usize]
                .tiles
                .insert_tile(tile_row, tile_col, tile);
        }
        Ok(())
    }
}

/// Validate a domain-growth request and return the (row, col) shift of
/// the old grid inside the new one.
fn growth_shifts(
    old_origin: (i64, i64),
    old_shape: (usize, usize),
    veg: &Grid2<i32>,
    dem: &Grid2<f64>,
    origin: (i64, i64),
) -> Result<(usize, usize)> {
    if veg.shape() != dem.shape() {
        return Err(PropagatorError::InvalidGrowth(
            "veg and dem must have the same shape".into(),
        ));
    }
    let row_shift = old_origin.0 - origin.0;
    let col_shift = old_origin.1 - origin.1;
    if row_shift < 0 || col_shift < 0 {
        return Err(PropagatorError::InvalidGrowth(
            "new origin must not be south/east of the current origin: the \
             new grid must contain the old one"
                .into(),
        ));
    }
    if row_shift % TILE_SIZE as i64 != 0 || col_shift % TILE_SIZE as i64 != 0 {
        return Err(PropagatorError::InvalidGrowth(format!(
            "north/west growth must be a multiple of TILE_SIZE ({TILE_SIZE}) \
             cells; got shift ({row_shift}, {col_shift})"
        )));
    }
    let (row_shift, col_shift) = (row_shift as usize, col_shift as usize);
    if row_shift + old_shape.0 > veg.rows() || col_shift + old_shape.1 > veg.cols() {
        return Err(PropagatorError::InvalidGrowth(
            "new grid does not fully contain the current grid".into(),
        ));
    }
    Ok((row_shift, col_shift))
}

/// Scatter one tile into a dense grid through `extract`.
fn scatter_tile<T: Clone>(
    out: &mut Grid2<T>,
    tile: &Tile,
    tile_row: usize,
    tile_col: usize,
    extract: &impl Fn(&Tile, usize) -> T,
) {
    let (rows, cols) = out.shape();
    let row0 = tile_row << TILE_SHIFT;
    let col0 = tile_col << TILE_SHIFT;
    let height = TILE_SIZE.min(rows.saturating_sub(row0));
    let width = TILE_SIZE.min(cols.saturating_sub(col0));
    for local_row in 0..height {
        for local_col in 0..width {
            out[(row0 + local_row, col0 + local_col)] =
                extract(tile, local_row * TILE_SIZE + local_col);
        }
    }
}

/// Freeze-criterion counts over every realization's live tiles, from
/// [`Propagator::freeze_diagnostic`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FreezeDiagnostic {
    /// live (allocated) tiles
    pub allocated: usize,
    /// tiles the strict (spotting) criterion would freeze: every
    /// in-domain fuel cell burnt
    pub strict_freezable: usize,
    /// tiles the reachability (no-spotting) criterion would freeze
    pub reachability_freezable: usize,
    /// tiles reachability would freeze but the strict criterion keeps
    pub blocked_by_strict: usize,
}

/// `f(r, &reals[r])` for every realization, in order: on scoped threads
/// when `n_threads > 1`, else sequentially (no thread is ever spawned,
/// so this works on targets without threads such as wasm32).
fn map_realizations<T, F>(reals: &[Realization], n_threads: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, &Realization) -> T + Sync,
{
    let n_threads = n_threads.max(1).min(reals.len().max(1));
    if n_threads <= 1 {
        return reals.iter().enumerate().map(|(r, real)| f(r, real)).collect();
    }
    let chunk_size = reals.len().div_ceil(n_threads);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = reals
            .chunks(chunk_size)
            .enumerate()
            .map(|(chunk, slice)| {
                scope.spawn(move || {
                    slice
                        .iter()
                        .enumerate()
                        .map(|(i, real)| f(chunk * chunk_size + i, real))
                        .collect::<Vec<T>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("freeze worker panicked"))
            .collect()
    })
}

/// Live tiles of one realization that can be frozen, in tile-position
/// order, each with its frozen summary. `strict`: every in-domain fuel
/// cell burnt (the spotting criterion); otherwise no cell reachable from
/// a pending front event (sparse over-approximation).
fn freezable_tiles(
    real: &Realization,
    fuel: &FuelMasks,
    strict: bool,
) -> Vec<(usize, usize, i32, FrozenSummary)> {
    let tiles = &real.tiles;
    if strict || real.heap.is_empty() {
        // no flood needed: open masks alone decide (an empty heap reaches
        // nothing, so every tile freezes)
        return tiles
            .allocated()
            .filter_map(|(tile_row, tile_col, slot)| {
                let open = live_open(tiles.tile(slot), fuel.get(tile_row, tile_col));
                if strict && !is_empty(&open) {
                    return None;
                }
                Some((tile_row, tile_col, slot, FrozenSummary::from_open(&open)))
            })
            .collect();
    }
    let mut reach = Reach::build(tiles, fuel);
    let (tiles_h, tiles_w) = tiles.tiles_shape();
    for (row, col) in real.heap.event_cells() {
        if row >= 0
            && col >= 0
            && ((row as usize) >> TILE_SHIFT) < tiles_h
            && ((col as usize) >> TILE_SHIFT) < tiles_w
        {
            reach.seed_cell(row as usize, col as usize);
        }
    }
    reach.run();
    tiles
        .allocated()
        .filter(|&(tile_row, tile_col, _)| !reach.reached(tile_row, tile_col))
        .map(|(tile_row, tile_col, slot)| {
            let open = reach.open_mask(tile_row, tile_col);
            (tile_row, tile_col, slot, FrozenSummary::from_open(&open))
        })
        .collect()
}

/// Accumulate a newly frozen tile into the frozen-fold cache.
fn fold_cache_add(
    blocks: &mut HashMap<(i64, i64), StateFold>,
    dirty: &HashSet<(i64, i64)>,
    valid: bool,
    key: &TileKey,
    tile: &Tile,
) {
    if !valid {
        return;
    }
    let position = (key.1, key.2);
    if dirty.contains(&position) {
        // the block will be rebuilt from the store anyway
        return;
    }
    blocks
        .entry(position)
        .or_insert_with(|| StateFold::empty(TILE_SIZE, TILE_SIZE))
        .accumulate_tile(tile, 0, 0);
}
