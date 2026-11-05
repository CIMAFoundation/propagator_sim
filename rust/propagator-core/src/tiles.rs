//! Block-sparse (tiled) per-realization state.
//!
//! Per-cell tracking (flags, arrival time, ROS, fireline intensity) lives
//! in 32x32 tiles allocated on demand as fire reaches them, so memory
//! scales with burned area instead of grid size x realizations.

use std::collections::HashMap;

pub const TILE_SHIFT: usize = 5;
pub const TILE_SIZE: usize = 1 << TILE_SHIFT;
pub const TILE_MASK: usize = TILE_SIZE - 1;
pub const TILE_CELLS: usize = TILE_SIZE * TILE_SIZE;

/// One bit per cell of a tile: `mask[local_row] >> local_col & 1`.
pub type RowMask = [u32; TILE_SIZE];

// flags bitfield
pub const FLAG_FIRE: u8 = 1;
pub const FLAG_SPOT_GEN: u8 = 2;
pub const FLAG_SPOT_RECV: u8 = 4;

// tile index sentinels
pub const UNALLOCATED: i32 = -1;
/// Frozen to disk: every in-domain cell provably can never ignite again,
/// so reads treat the whole tile as burnt without touching the contents.
pub const FROZEN_TILE: i32 = -2;
/// Cold (fuzzy-frozen) to disk: stored like a frozen tile but may still
/// burn, so it is never read as burnt; the kernel faults it back in
/// before touching any of its cells.
pub const COLD_TILE: i32 = -3;

/// One tile's per-cell state (struct-of-arrays, ~13 KiB boxed).
#[derive(Clone)]
pub struct Tile {
    pub flags: [u8; TILE_CELLS],
    pub arrival: [i32; TILE_CELLS],
    pub ros: [f32; TILE_CELLS],
    pub fli: [f32; TILE_CELLS],
}

impl Tile {
    pub fn zeroed() -> Box<Tile> {
        Box::new(Tile {
            flags: [0; TILE_CELLS],
            arrival: [0; TILE_CELLS],
            ros: [0.0; TILE_CELLS],
            fli: [0.0; TILE_CELLS],
        })
    }
}

#[inline]
pub fn local_index(row: usize, col: usize) -> usize {
    (row & TILE_MASK) * TILE_SIZE + (col & TILE_MASK)
}

/// In-RAM digest of a frozen tile, recorded at freeze time so
/// reachability never has to read the record back from disk.
#[derive(Clone, Debug, PartialEq)]
pub enum FrozenSummary {
    /// no open cell (in-domain, fuel, unburnt): can never be reached
    Solid,
    /// the tile's open cells
    Open(Box<RowMask>),
}

impl FrozenSummary {
    pub fn from_open(open: &RowMask) -> FrozenSummary {
        if open.iter().all(|&row| row == 0) {
            FrozenSummary::Solid
        } else {
            FrozenSummary::Open(Box::new(*open))
        }
    }

    /// Open-cell mask (all zero when solid).
    #[cfg(test)]
    pub fn open(&self) -> RowMask {
        match self {
            FrozenSummary::Solid => [0; TILE_SIZE],
            FrozenSummary::Open(mask) => **mask,
        }
    }
}

/// One realization's tiled state: an index grid mapping every spatial
/// 32x32 block to `UNALLOCATED`, `FROZEN_TILE`, or a slot in the pool.
#[derive(Clone)]
pub struct TileGrid {
    tiles_h: usize,
    tiles_w: usize,
    idx: Vec<i32>,
    pool: Vec<Box<Tile>>,
    /// summary of every `FROZEN_TILE` and `COLD_TILE` position, keyed by
    /// (tile_row, tile_col); travels with checkpoints and re-anchors with
    /// the grid
    frozen: HashMap<(usize, usize), FrozenSummary>,
    /// number of `COLD_TILE` positions (0 keeps the kernel's checks cheap)
    cold: usize,
}

impl TileGrid {
    pub fn new(rows: usize, cols: usize) -> TileGrid {
        let tiles_h = rows.div_ceil(TILE_SIZE);
        let tiles_w = cols.div_ceil(TILE_SIZE);
        TileGrid {
            tiles_h,
            tiles_w,
            idx: vec![UNALLOCATED; tiles_h * tiles_w],
            pool: Vec::new(),
            frozen: HashMap::new(),
            cold: 0,
        }
    }

    #[inline]
    pub fn tiles_shape(&self) -> (usize, usize) {
        (self.tiles_h, self.tiles_w)
    }

    #[inline]
    pub fn slot_of(&self, tile_row: usize, tile_col: usize) -> i32 {
        self.idx[tile_row * self.tiles_w + tile_col]
    }

    #[inline]
    pub fn set_slot(&mut self, tile_row: usize, tile_col: usize, slot: i32) {
        self.idx[tile_row * self.tiles_w + tile_col] = slot;
    }

    #[allow(dead_code)]
    pub fn pool(&self) -> &[Box<Tile>] {
        &self.pool
    }

    #[allow(dead_code)]
    pub fn pool_len(&self) -> usize {
        self.pool.len()
    }

    #[inline]
    pub fn tile(&self, slot: i32) -> &Tile {
        &self.pool[slot as usize]
    }

    #[inline]
    pub fn tile_mut(&mut self, slot: i32) -> &mut Tile {
        &mut self.pool[slot as usize]
    }

    /// Flags byte for a cell. Unallocated tiles read as 0; frozen tiles
    /// read as burnt (freeze criterion guarantees it).
    #[inline]
    pub fn read_flags(&self, row: usize, col: usize) -> u8 {
        let slot = self.slot_of(row >> TILE_SHIFT, col >> TILE_SHIFT);
        debug_assert_ne!(slot, COLD_TILE, "reading a cold tile");
        if slot == FROZEN_TILE {
            return FLAG_FIRE;
        }
        if slot < 0 {
            return 0;
        }
        self.pool[slot as usize].flags[local_index(row, col)]
    }

    /// Slot for a cell's tile, allocating (zeroed) if needed. Must not be
    /// called on a frozen tile — the caller thaws first (the kernel never
    /// hits this: frozen cells read as burnt and are skipped).
    #[inline]
    pub fn ensure_slot(&mut self, row: usize, col: usize) -> i32 {
        let tile_row = row >> TILE_SHIFT;
        let tile_col = col >> TILE_SHIFT;
        let slot = self.slot_of(tile_row, tile_col);
        debug_assert_ne!(slot, FROZEN_TILE, "writing into a frozen tile");
        debug_assert_ne!(slot, COLD_TILE, "writing into a cold tile");
        if slot >= 0 {
            return slot;
        }
        let new_slot = self.pool.len() as i32;
        self.pool.push(Tile::zeroed());
        self.set_slot(tile_row, tile_col, new_slot);
        new_slot
    }

    /// Insert an existing tile (thaw / checkpoint load) at a position,
    /// dropping its frozen summary if it had one.
    pub fn insert_tile(
        &mut self,
        tile_row: usize,
        tile_col: usize,
        tile: Box<Tile>,
    ) -> i32 {
        if self.slot_of(tile_row, tile_col) == COLD_TILE {
            self.cold -= 1;
        }
        self.frozen.remove(&(tile_row, tile_col));
        let slot = self.pool.len() as i32;
        self.pool.push(tile);
        self.set_slot(tile_row, tile_col, slot);
        slot
    }

    /// Remove a set of pool slots (marking their positions with
    /// `new_sentinel`), compacting the pool and remapping the index.
    pub fn remove_slots(&mut self, remove: &[i32], new_sentinel: i32) {
        if remove.is_empty() {
            return;
        }
        let old_len = self.pool.len();
        let mut removed = vec![false; old_len];
        for &slot in remove {
            removed[slot as usize] = true;
        }
        let mut remap = vec![-1i32; old_len];
        let mut kept = 0i32;
        for (slot, &is_removed) in removed.iter().enumerate() {
            if !is_removed {
                remap[slot] = kept;
                kept += 1;
            }
        }
        let old_pool = std::mem::take(&mut self.pool);
        self.pool = old_pool
            .into_iter()
            .enumerate()
            .filter(|(slot, _)| !removed[*slot])
            .map(|(_, tile)| tile)
            .collect();
        for entry in &mut self.idx {
            if *entry >= 0 {
                let mapped = remap[*entry as usize];
                *entry = if mapped >= 0 { mapped } else { new_sentinel };
            }
        }
    }

    /// Mark allocated tiles frozen, `(tile_row, tile_col, slot, summary)`
    /// each: their slots are released and the summaries recorded.
    #[cfg(test)]
    pub fn freeze_tiles(&mut self, tiles: Vec<(usize, usize, i32, FrozenSummary)>) {
        self.store_tiles(tiles, Vec::new());
    }

    /// Release allocated tiles to disk in one pass (slots are only valid
    /// until the pool compacts): `dead` become `FROZEN_TILE`, `cold`
    /// become `COLD_TILE`, each with its summary recorded.
    pub fn store_tiles(
        &mut self,
        dead: Vec<(usize, usize, i32, FrozenSummary)>,
        cold: Vec<(usize, usize, i32, FrozenSummary)>,
    ) {
        let slots: Vec<i32> = dead.iter().chain(cold.iter()).map(|t| t.2).collect();
        self.remove_slots(&slots, FROZEN_TILE);
        for (tile_row, tile_col, _, summary) in dead {
            debug_assert_eq!(self.slot_of(tile_row, tile_col), FROZEN_TILE);
            self.frozen.insert((tile_row, tile_col), summary);
        }
        self.cold += cold.len();
        for (tile_row, tile_col, _, summary) in cold {
            debug_assert_eq!(self.slot_of(tile_row, tile_col), FROZEN_TILE);
            self.set_slot(tile_row, tile_col, COLD_TILE);
            self.frozen.insert((tile_row, tile_col), summary);
        }
    }

    /// Number of cold tiles.
    #[inline]
    pub fn cold_count(&self) -> usize {
        self.cold
    }

    /// Number of dead (frozen, never burning again) tiles.
    pub fn dead_count(&self) -> usize {
        self.frozen.len() - self.cold
    }

    /// Summary of a frozen position (`None` if not frozen, or frozen
    /// without one: callers must then assume every fuel cell is open).
    #[inline]
    pub fn frozen_summary(&self, tile_row: usize, tile_col: usize) -> Option<&FrozenSummary> {
        self.frozen.get(&(tile_row, tile_col))
    }

    /// Number of tiles on disk (dead and cold) with a summary.
    pub fn frozen_count(&self) -> usize {
        self.frozen.len()
    }

    /// Iterate allocated tiles: (tile_row, tile_col, slot).
    pub fn allocated(&self) -> impl Iterator<Item = (usize, usize, i32)> + '_ {
        let tiles_w = self.tiles_w;
        self.idx
            .iter()
            .enumerate()
            .filter(|(_, &slot)| slot >= 0)
            .map(move |(flat, &slot)| (flat / tiles_w, flat % tiles_w, slot))
    }

    /// Iterate frozen (dead) tile positions: (tile_row, tile_col).
    pub fn frozen_positions(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.positions_of(FROZEN_TILE)
    }

    /// Iterate positions holding `sentinel`: (tile_row, tile_col).
    pub fn positions_of(&self, sentinel: i32) -> impl Iterator<Item = (usize, usize)> + '_ {
        let tiles_w = self.tiles_w;
        self.idx
            .iter()
            .enumerate()
            .filter(move |(_, &slot)| slot == sentinel)
            .map(move |(flat, _)| (flat / tiles_w, flat % tiles_w))
    }

    /// Re-anchor into a larger domain: the old index grid is embedded at
    /// `(tile_row0, tile_col0)`; the pool transfers untouched.
    pub fn reanchor(&mut self, rows: usize, cols: usize, tile_row0: usize, tile_col0: usize) {
        let tiles_h = rows.div_ceil(TILE_SIZE);
        let tiles_w = cols.div_ceil(TILE_SIZE);
        let mut idx = vec![UNALLOCATED; tiles_h * tiles_w];
        for tile_row in 0..self.tiles_h {
            for tile_col in 0..self.tiles_w {
                idx[(tile_row + tile_row0) * tiles_w + (tile_col + tile_col0)] =
                    self.idx[tile_row * self.tiles_w + tile_col];
            }
        }
        self.tiles_h = tiles_h;
        self.tiles_w = tiles_w;
        self.idx = idx;
        self.frozen = std::mem::take(&mut self.frozen)
            .into_iter()
            .map(|((tile_row, tile_col), summary)| {
                ((tile_row + tile_row0, tile_col + tile_col0), summary)
            })
            .collect();
    }
}

/// 8-connected flood fill over `open` cells from the given seeds, marking
/// `reachable`. Grids are `grid_h x grid_w` row-major; seeds outside
/// `open` are ignored.
#[cfg(test)]
pub fn flood_reachable(
    open: &[bool],
    grid_h: usize,
    grid_w: usize,
    seeds: &[(usize, usize)],
    reachable: &mut [bool],
) {
    debug_assert_eq!(open.len(), grid_h * grid_w);
    debug_assert_eq!(reachable.len(), grid_h * grid_w);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for &(row, col) in seeds {
        let flat = row * grid_w + col;
        if open[flat] && !reachable[flat] {
            reachable[flat] = true;
            stack.push((row, col));
        }
    }
    while let Some((row, col)) = stack.pop() {
        for dr in -1i64..=1 {
            for dc in -1i64..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nr = row as i64 + dr;
                let nc = col as i64 + dc;
                if nr < 0 || nc < 0 || nr as usize >= grid_h || nc as usize >= grid_w {
                    continue;
                }
                let flat = nr as usize * grid_w + nc as usize;
                if open[flat] && !reachable[flat] {
                    reachable[flat] = true;
                    stack.push((nr as usize, nc as usize));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_flags_sentinels() {
        let mut grid = TileGrid::new(100, 100);
        assert_eq!(grid.read_flags(50, 50), 0);
        let slot = grid.ensure_slot(50, 50);
        grid.tile_mut(slot).flags[local_index(50, 50)] = FLAG_FIRE;
        assert_eq!(grid.read_flags(50, 50), FLAG_FIRE);
        assert_eq!(grid.read_flags(50, 51), 0);
        grid.set_slot(0, 0, FROZEN_TILE);
        assert_eq!(grid.read_flags(0, 0), FLAG_FIRE);
    }

    #[test]
    fn remove_slots_compacts_and_remaps() {
        let mut grid = TileGrid::new(128, 128);
        let a = grid.ensure_slot(0, 0);
        let b = grid.ensure_slot(40, 40);
        let c = grid.ensure_slot(80, 80);
        grid.tile_mut(c).arrival[0] = 77;
        grid.remove_slots(&[b], FROZEN_TILE);
        assert_eq!(grid.slot_of(1, 1), FROZEN_TILE);
        assert_eq!(grid.pool_len(), 2);
        let _ = a;
        // c moved to slot 1, content preserved
        let c_slot = grid.slot_of(2, 2);
        assert_eq!(c_slot, 1);
        assert_eq!(grid.tile(c_slot).arrival[0], 77);
    }

    #[test]
    fn flood_respects_barriers() {
        // 3x3, middle column closed
        let open = vec![
            true, false, true, //
            true, false, true, //
            true, false, true,
        ];
        let mut reachable = vec![false; 9];
        flood_reachable(&open, 3, 3, &[(0, 0)], &mut reachable);
        assert!(reachable[0] && reachable[3] && reachable[6]);
        assert!(!reachable[2] && !reachable[5] && !reachable[8]);
    }
}
