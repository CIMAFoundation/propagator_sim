//! Per-cell aggregates of the tiled state across realizations, and the
//! derived output maps.

use crate::grid::Grid2;
use crate::tiles::{
    Tile, TileGrid, FLAG_FIRE, FLAG_SPOT_GEN, FLAG_SPOT_RECV, TILE_SHIFT, TILE_SIZE,
};

/// Per-cell aggregates across realizations (see spec §3.4).
///
/// Every grid is `rows x cols`. Counts and sums are reduced across all
/// realizations; dividing a `*_sum` by `count` yields the mean over the
/// realizations that burned the cell. `arrival_min` starts at `i32::MAX` so
/// the min reduction is well-defined for never-burnt cells.
#[derive(Clone)]
pub struct StateFold {
    /// number of realizations in which the cell burned
    pub count: Grid2<i32>,
    /// number of realizations in which the cell emitted embers (spotting)
    pub spot_gen_count: Grid2<i32>,
    /// number of realizations in which the cell was ignited by embers
    pub spot_recv_count: Grid2<i32>,
    /// earliest arrival time across realizations (`i32::MAX` if never burnt)
    pub arrival_min: Grid2<i32>,
    /// sum of arrival times over burning realizations
    pub arrival_sum: Grid2<f64>,
    /// sum of rate of spread over burning realizations
    pub ros_sum: Grid2<f64>,
    /// maximum rate of spread across realizations
    pub ros_max: Grid2<f32>,
    /// sum of fireline intensity over burning realizations
    pub fli_sum: Grid2<f64>,
    /// maximum fireline intensity across realizations
    pub fli_max: Grid2<f32>,
}

impl StateFold {
    /// A zeroed fold (with `arrival_min` seeded to `i32::MAX`).
    pub fn empty(rows: usize, cols: usize) -> StateFold {
        StateFold {
            count: Grid2::filled(rows, cols, 0),
            spot_gen_count: Grid2::filled(rows, cols, 0),
            spot_recv_count: Grid2::filled(rows, cols, 0),
            arrival_min: Grid2::filled(rows, cols, i32::MAX),
            arrival_sum: Grid2::filled(rows, cols, 0.0),
            ros_sum: Grid2::filled(rows, cols, 0.0),
            ros_max: Grid2::filled(rows, cols, 0.0),
            fli_sum: Grid2::filled(rows, cols, 0.0),
            fli_max: Grid2::filled(rows, cols, 0.0),
        }
    }

    /// Accumulate one cell's state at `(row, col)`.
    #[inline]
    pub fn accumulate_cell(
        &mut self,
        row: usize,
        col: usize,
        flags: u8,
        arrival: i32,
        ros: f32,
        fli: f32,
    ) {
        if flags == 0 {
            return;
        }
        if flags & FLAG_SPOT_GEN != 0 {
            self.spot_gen_count[(row, col)] += 1;
        }
        if flags & FLAG_SPOT_RECV != 0 {
            self.spot_recv_count[(row, col)] += 1;
        }
        if flags & FLAG_FIRE == 0 {
            return;
        }
        self.count[(row, col)] += 1;
        if arrival < self.arrival_min[(row, col)] {
            self.arrival_min[(row, col)] = arrival;
        }
        self.arrival_sum[(row, col)] += arrival as f64;
        if !ros.is_nan() {
            self.ros_sum[(row, col)] += ros as f64;
            if ros > self.ros_max[(row, col)] {
                self.ros_max[(row, col)] = ros;
            }
        }
        if !fli.is_nan() {
            self.fli_sum[(row, col)] += fli as f64;
            if fli > self.fli_max[(row, col)] {
                self.fli_max[(row, col)] = fli;
            }
        }
    }

    /// Accumulate one full tile placed with its top-left cell at
    /// `(row0, col0)` (may be clipped by the fold bounds; negative
    /// offsets are not supported — frozen tiles are always in-domain).
    pub fn accumulate_tile(&mut self, tile: &Tile, row0: usize, col0: usize) {
        let (rows, cols) = self.count.shape();
        let height = TILE_SIZE.min(rows.saturating_sub(row0));
        let width = TILE_SIZE.min(cols.saturating_sub(col0));
        for local_row in 0..height {
            for local_col in 0..width {
                let local = local_row * TILE_SIZE + local_col;
                self.accumulate_cell(
                    row0 + local_row,
                    col0 + local_col,
                    tile.flags[local],
                    tile.arrival[local],
                    tile.ros[local],
                    tile.fli[local],
                );
            }
        }
    }

    /// Merge a (TILE_SIZE x TILE_SIZE) aggregate block placed at
    /// `(row0, col0)` (frozen-fold cache blocks).
    pub fn merge_block(&mut self, block: &StateFold, row0: usize, col0: usize) {
        debug_assert_eq!(block.count.shape(), (TILE_SIZE, TILE_SIZE));
        let (rows, cols) = self.count.shape();
        let height = TILE_SIZE.min(rows.saturating_sub(row0));
        let width = TILE_SIZE.min(cols.saturating_sub(col0));
        for local_row in 0..height {
            for local_col in 0..width {
                let src = (local_row, local_col);
                let dst = (row0 + local_row, col0 + local_col);
                self.count[dst] += block.count[src];
                self.spot_gen_count[dst] += block.spot_gen_count[src];
                self.spot_recv_count[dst] += block.spot_recv_count[src];
                if block.arrival_min[src] < self.arrival_min[dst] {
                    self.arrival_min[dst] = block.arrival_min[src];
                }
                self.arrival_sum[dst] += block.arrival_sum[src];
                self.ros_sum[dst] += block.ros_sum[src];
                self.fli_sum[dst] += block.fli_sum[src];
                if block.ros_max[src] > self.ros_max[dst] {
                    self.ros_max[dst] = block.ros_max[src];
                }
                if block.fli_max[src] > self.fli_max[dst] {
                    self.fli_max[dst] = block.fli_max[src];
                }
            }
        }
    }
}

/// Fold every allocated tile of one realization into `fold` (frozen
/// tiles are folded separately from the cache).
pub fn fold_realization(fold: &mut StateFold, tiles: &TileGrid) {
    for (tile_row, tile_col, slot) in tiles.allocated() {
        let tile = tiles.tile(slot);
        fold.accumulate_tile(tile, tile_row << TILE_SHIFT, tile_col << TILE_SHIFT);
    }
}

/// Summary statistics for the current state (spec §6.5).
#[derive(Clone, Debug, PartialEq)]
pub struct PropagatorStats {
    /// realizations with a non-empty front heap
    pub n_active: usize,
    /// expected burned area, m^2
    pub area_mean: f64,
    /// area with burn probability >= 0.5 / 0.75 / 0.9, m^2
    pub area_50: f64,
    pub area_75: f64,
    pub area_90: f64,
}

/// Snapshot of simulation outputs at a given time.
#[derive(Clone)]
pub struct PropagatorOutput {
    /// seconds from simulation start
    pub time: i64,
    pub fire_probability: Grid2<f32>,
    pub spotting_generation_probability: Grid2<f32>,
    pub spotting_receiving_probability: Grid2<f32>,
    /// 0 where never burnt
    pub min_arrival_time: Grid2<f32>,
    /// NaN where never burnt
    pub mean_arrival_time: Grid2<f32>,
    pub ros_mean: Grid2<f32>,
    pub ros_max: Grid2<f32>,
    pub fli_mean: Grid2<f32>,
    pub fli_max: Grid2<f32>,
    pub stats: PropagatorStats,
}
