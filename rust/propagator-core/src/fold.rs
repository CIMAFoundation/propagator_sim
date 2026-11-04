//! Per-cell aggregates of the tiled state across realizations, and the
//! derived output maps.

use crate::grid::Grid2;
use crate::models::{flame_height, radiant_reach};
use crate::tiles::{
    Tile, TileGrid, FLAG_FIRE, FLAG_SPOT_GEN, FLAG_SPOT_RECV, TILE_SHIFT, TILE_SIZE,
};

/// Fold one cell's state into the per-cell aggregates.
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_into(
    count: &mut i32,
    spot_gen_count: &mut i32,
    spot_recv_count: &mut i32,
    arrival_min: &mut i32,
    arrival_sum: &mut f64,
    ros_sum: &mut f64,
    ros_max: &mut f32,
    fli_sum: &mut f64,
    fli_max: &mut f32,
    flame_sum: &mut f64,
    flame_max: &mut f32,
    radiant_sum: &mut f64,
    radiant_max: &mut f32,
    flags: u8,
    arrival: i32,
    ros: f32,
    fli: f32,
) {
    if flags == 0 {
        return;
    }
    if flags & FLAG_SPOT_GEN != 0 {
        *spot_gen_count += 1;
    }
    if flags & FLAG_SPOT_RECV != 0 {
        *spot_recv_count += 1;
    }
    if flags & FLAG_FIRE == 0 {
        return;
    }
    *count += 1;
    if arrival < *arrival_min {
        *arrival_min = arrival;
    }
    *arrival_sum += arrival as f64;
    if !ros.is_nan() {
        *ros_sum += ros as f64;
        if ros > *ros_max {
            *ros_max = ros;
        }
    }
    if !fli.is_nan() {
        *fli_sum += fli as f64;
        if fli > *fli_max {
            *fli_max = fli;
        }
        let flame = flame_height(fli as f64);
        *flame_sum += flame;
        if flame as f32 > *flame_max {
            *flame_max = flame as f32;
        }
        let reach = radiant_reach(fli as f64);
        *radiant_sum += reach;
        if reach as f32 > *radiant_max {
            *radiant_max = reach as f32;
        }
    }
}

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
    /// sum of per-realization flame height (m) over burning realizations
    pub flame_sum: Grid2<f64>,
    /// maximum flame height (m) across realizations
    pub flame_max: Grid2<f32>,
    /// sum of per-realization radiant reach (m) over burning realizations
    pub radiant_sum: Grid2<f64>,
    /// maximum radiant reach (m) across realizations
    pub radiant_max: Grid2<f32>,
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
            flame_sum: Grid2::filled(rows, cols, 0.0),
            flame_max: Grid2::filled(rows, cols, 0.0),
            radiant_sum: Grid2::filled(rows, cols, 0.0),
            radiant_max: Grid2::filled(rows, cols, 0.0),
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
        accumulate_into(
            &mut self.count[(row, col)],
            &mut self.spot_gen_count[(row, col)],
            &mut self.spot_recv_count[(row, col)],
            &mut self.arrival_min[(row, col)],
            &mut self.arrival_sum[(row, col)],
            &mut self.ros_sum[(row, col)],
            &mut self.ros_max[(row, col)],
            &mut self.fli_sum[(row, col)],
            &mut self.fli_max[(row, col)],
            &mut self.flame_sum[(row, col)],
            &mut self.flame_max[(row, col)],
            &mut self.radiant_sum[(row, col)],
            &mut self.radiant_max[(row, col)],
            flags,
            arrival,
            ros,
            fli,
        );
    }

    /// Split the fold into disjoint horizontal bands of `rows_per_band`
    /// rows (the last may be shorter) that can be filled concurrently.
    pub fn bands_mut(&mut self, rows_per_band: usize) -> Vec<FoldBand<'_>> {
        let (rows, cols) = self.count.shape();
        let chunk = rows_per_band.max(1) * cols;
        let count = self.count.as_mut_slice().chunks_mut(chunk);
        let mut spot_gen_count = self.spot_gen_count.as_mut_slice().chunks_mut(chunk);
        let mut spot_recv_count = self.spot_recv_count.as_mut_slice().chunks_mut(chunk);
        let mut arrival_min = self.arrival_min.as_mut_slice().chunks_mut(chunk);
        let mut arrival_sum = self.arrival_sum.as_mut_slice().chunks_mut(chunk);
        let mut ros_sum = self.ros_sum.as_mut_slice().chunks_mut(chunk);
        let mut ros_max = self.ros_max.as_mut_slice().chunks_mut(chunk);
        let mut fli_sum = self.fli_sum.as_mut_slice().chunks_mut(chunk);
        let mut fli_max = self.fli_max.as_mut_slice().chunks_mut(chunk);
        let mut flame_sum = self.flame_sum.as_mut_slice().chunks_mut(chunk);
        let mut flame_max = self.flame_max.as_mut_slice().chunks_mut(chunk);
        let mut radiant_sum = self.radiant_sum.as_mut_slice().chunks_mut(chunk);
        let mut radiant_max = self.radiant_max.as_mut_slice().chunks_mut(chunk);
        let mut bands = Vec::with_capacity(rows.div_ceil(rows_per_band.max(1)));
        let mut row0 = 0;
        for count in count {
            let rows = count.len() / cols;
            bands.push(FoldBand {
                row0,
                rows,
                cols,
                count,
                spot_gen_count: spot_gen_count.next().unwrap(),
                spot_recv_count: spot_recv_count.next().unwrap(),
                arrival_min: arrival_min.next().unwrap(),
                arrival_sum: arrival_sum.next().unwrap(),
                ros_sum: ros_sum.next().unwrap(),
                ros_max: ros_max.next().unwrap(),
                fli_sum: fli_sum.next().unwrap(),
                fli_max: fli_max.next().unwrap(),
                flame_sum: flame_sum.next().unwrap(),
                flame_max: flame_max.next().unwrap(),
                radiant_sum: radiant_sum.next().unwrap(),
                radiant_max: radiant_max.next().unwrap(),
            });
            row0 += rows;
        }
        bands
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
                self.flame_sum[dst] += block.flame_sum[src];
                self.radiant_sum[dst] += block.radiant_sum[src];
                if block.ros_max[src] > self.ros_max[dst] {
                    self.ros_max[dst] = block.ros_max[src];
                }
                if block.fli_max[src] > self.fli_max[dst] {
                    self.fli_max[dst] = block.fli_max[src];
                }
                if block.flame_max[src] > self.flame_max[dst] {
                    self.flame_max[dst] = block.flame_max[src];
                }
                if block.radiant_max[src] > self.radiant_max[dst] {
                    self.radiant_max[dst] = block.radiant_max[src];
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

/// A horizontal band of a [`StateFold`], see [`StateFold::bands_mut`].
pub struct FoldBand<'a> {
    /// first grid row of the band
    pub row0: usize,
    /// rows in the band
    pub rows: usize,
    cols: usize,
    count: &'a mut [i32],
    spot_gen_count: &'a mut [i32],
    spot_recv_count: &'a mut [i32],
    arrival_min: &'a mut [i32],
    arrival_sum: &'a mut [f64],
    ros_sum: &'a mut [f64],
    ros_max: &'a mut [f32],
    fli_sum: &'a mut [f64],
    fli_max: &'a mut [f32],
    flame_sum: &'a mut [f64],
    flame_max: &'a mut [f32],
    radiant_sum: &'a mut [f64],
    radiant_max: &'a mut [f32],
}

impl FoldBand<'_> {
    /// Accumulate a tile whose top-left cell is at grid `(row0, col0)`,
    /// clipped to the band and the grid.
    pub fn accumulate_tile(&mut self, tile: &Tile, row0: usize, col0: usize) {
        let band_end = self.row0 + self.rows;
        let first = row0.max(self.row0);
        let last = (row0 + TILE_SIZE).min(band_end);
        if first >= last {
            return;
        }
        let width = TILE_SIZE.min(self.cols.saturating_sub(col0));
        for row in first..last {
            let local_row = row - row0;
            let base = (row - self.row0) * self.cols + col0;
            for local_col in 0..width {
                let local = local_row * TILE_SIZE + local_col;
                let i = base + local_col;
                accumulate_into(
                    &mut self.count[i],
                    &mut self.spot_gen_count[i],
                    &mut self.spot_recv_count[i],
                    &mut self.arrival_min[i],
                    &mut self.arrival_sum[i],
                    &mut self.ros_sum[i],
                    &mut self.ros_max[i],
                    &mut self.fli_sum[i],
                    &mut self.fli_max[i],
                    &mut self.flame_sum[i],
                    &mut self.flame_max[i],
                    &mut self.radiant_sum[i],
                    &mut self.radiant_max[i],
                    tile.flags[local],
                    tile.arrival[local],
                    tile.ros[local],
                    tile.fli[local],
                );
            }
        }
    }

    /// Fold the tiles of every realization that intersect this band. Bands
    /// of `TILE_SIZE` rows starting at a tile boundary touch exactly one
    /// tile row each.
    pub fn fold_realizations<'t>(&mut self, tiles: impl Iterator<Item = &'t TileGrid>) {
        let tile_row0 = self.row0 >> TILE_SHIFT;
        let tile_row1 = (self.row0 + self.rows - 1) >> TILE_SHIFT;
        for grid in tiles {
            let (tiles_h, tiles_w) = grid.tiles_shape();
            for tile_row in tile_row0..=tile_row1.min(tiles_h - 1) {
                for tile_col in 0..tiles_w {
                    let slot = grid.slot_of(tile_row, tile_col);
                    if slot >= 0 {
                        self.accumulate_tile(
                            grid.tile(slot),
                            tile_row << TILE_SHIFT,
                            tile_col << TILE_SHIFT,
                        );
                    }
                }
            }
        }
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
    /// mean Byram flame height over burning realizations, m — NaN where
    /// never burnt
    pub flame_height_mean: Grid2<f32>,
    /// max Byram flame height across realizations, m — 0 where never burnt
    pub flame_height_max: Grid2<f32>,
    /// mean radiant reach over burning realizations, m — NaN where never
    /// burnt
    pub radiant_reach_mean: Grid2<f32>,
    /// max radiant reach across realizations, m — 0 where never burnt
    pub radiant_reach_max: Grid2<f32>,
    pub stats: PropagatorStats,
}
