//! Block-sparse (tiled) per-realization state.
//!
//! Per-cell tracking (flags, arrival time, ROS, fireline intensity) lives
//! in 32x32 tiles allocated on demand as fire reaches them, so memory
//! scales with burned area instead of grid size x realizations.

pub const TILE_SHIFT: usize = 5;
pub const TILE_SIZE: usize = 1 << TILE_SHIFT;
pub const TILE_MASK: usize = TILE_SIZE - 1;
pub const TILE_CELLS: usize = TILE_SIZE * TILE_SIZE;

// flags bitfield
pub const FLAG_FIRE: u8 = 1;
pub const FLAG_SPOT_GEN: u8 = 2;
pub const FLAG_SPOT_RECV: u8 = 4;

// tile index sentinels
pub const UNALLOCATED: i32 = -1;
/// Frozen to disk: every in-domain cell provably can never ignite again,
/// so reads treat the whole tile as burnt without touching the contents.
pub const FROZEN_TILE: i32 = -2;

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

/// One realization's tiled state: an index grid mapping every spatial
/// 32x32 block to `UNALLOCATED`, `FROZEN_TILE`, or a slot in the pool.
#[derive(Clone)]
pub struct TileGrid {
    tiles_h: usize,
    tiles_w: usize,
    idx: Vec<i32>,
    pool: Vec<Box<Tile>>,
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
        if slot >= 0 {
            return slot;
        }
        let new_slot = self.pool.len() as i32;
        self.pool.push(Tile::zeroed());
        self.set_slot(tile_row, tile_col, new_slot);
        new_slot
    }

    /// Insert an existing tile (thaw / checkpoint load) at a position.
    pub fn insert_tile(
        &mut self,
        tile_row: usize,
        tile_col: usize,
        tile: Box<Tile>,
    ) -> i32 {
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

    /// Iterate allocated tiles: (tile_row, tile_col, slot).
    pub fn allocated(&self) -> impl Iterator<Item = (usize, usize, i32)> + '_ {
        let tiles_w = self.tiles_w;
        self.idx
            .iter()
            .enumerate()
            .filter(|(_, &slot)| slot >= 0)
            .map(move |(flat, &slot)| (flat / tiles_w, flat % tiles_w, slot))
    }

    /// Iterate frozen tile positions: (tile_row, tile_col).
    #[allow(dead_code)]
    pub fn frozen_positions(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let tiles_w = self.tiles_w;
        self.idx
            .iter()
            .enumerate()
            .filter(|(_, &slot)| slot == FROZEN_TILE)
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
    }
}

/// 8-connected flood fill over `open` cells from the given seeds, marking
/// `reachable`. Grids are `grid_h x grid_w` row-major; seeds outside
/// `open` are ignored.
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
