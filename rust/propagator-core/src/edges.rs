//! Shared cache of the deterministic spread quantities of every
//! cell -> neighbour edge.
//!
//! For a given set of inputs (vegetation, fuels, DEM, wind, moisture) the
//! ignition probability, transition time and the ROS / fireline intensity
//! carried into a neighbour depend only on the source cell and the
//! direction, not on the realization. The heap kernel therefore evaluates
//! the spread models once per cell per *generation* of the inputs and
//! every realization reuses the result; only the Bernoulli draw stays per
//! realization.
//!
//! The cache is block-sparse like the state tiles: a 32x32 tile is
//! allocated the first time any realization pops a cell inside it, and
//! cells are filled lazily, so the work done is never more than what the
//! uncached kernel would do (the first realization to reach a cell pays
//! for it, the others read). Each cell carries the generation stamp it was
//! filled for, so invalidating after a boundary-condition change is O(1)
//! and needs no reallocation. All accesses go through atomics so the
//! realization threads can fill and read concurrently without locks: a
//! race between two fillers writes identical bits.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::kernel::N_NEIGHBOURS;
use crate::tiles::{local_index, TILE_CELLS, TILE_SHIFT, TILE_SIZE};

/// Spread quantities for one directed edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    /// ignition probability; negative when the edge can never carry fire
    /// (target off-grid, unburnable or not a spreading fuel)
    pub p: f64,
    /// transition time, seconds (>= 1)
    pub transition: i64,
    /// rate of spread carried into the target, m/min
    pub ros: f32,
    /// fireline intensity carried into the target, kW/m
    pub fli: f32,
}

impl Edge {
    /// An edge that never carries fire.
    pub const BLOCKED: Edge = Edge {
        p: -1.0,
        transition: 0,
        ros: 0.0,
        fli: 0.0,
    };

    #[inline]
    pub fn is_blocked(&self) -> bool {
        self.p < 0.0
    }
}

const EDGES_PER_TILE: usize = TILE_CELLS * N_NEIGHBOURS;

/// Generation stamp meaning "never filled".
const UNFILLED: u32 = 0;

/// One tile of cached edges (struct-of-arrays, cell-major: edge `d` of the
/// cell with local index `l` sits at `l * 8 + d`).
struct EdgeTile {
    /// generation each cell was filled for (`UNFILLED` = never)
    ready: Box<[AtomicU32]>,
    p: Box<[AtomicU64]>,
    /// transition time, saturated to `u32::MAX` by `get_or_fill`
    transition: Box<[AtomicU32]>,
    ros: Box<[AtomicU32]>,
    fli: Box<[AtomicU32]>,
}

fn zeroed_u32(n: usize) -> Box<[AtomicU32]> {
    (0..n).map(|_| AtomicU32::new(0)).collect()
}

fn zeroed_u64(n: usize) -> Box<[AtomicU64]> {
    (0..n).map(|_| AtomicU64::new(0)).collect()
}

impl EdgeTile {
    fn new() -> Box<EdgeTile> {
        Box::new(EdgeTile {
            ready: zeroed_u32(TILE_CELLS),
            p: zeroed_u64(EDGES_PER_TILE),
            transition: zeroed_u32(EDGES_PER_TILE),
            ros: zeroed_u32(EDGES_PER_TILE),
            fli: zeroed_u32(EDGES_PER_TILE),
        })
    }

    /// Bytes held by one tile (for memory accounting).
    pub const BYTES: usize = TILE_CELLS * 4 + EDGES_PER_TILE * (8 + 4 + 4 + 4);

    #[inline]
    fn read(&self, local: usize) -> [Edge; N_NEIGHBOURS] {
        let base = local * N_NEIGHBOURS;
        let mut edges = [Edge::BLOCKED; N_NEIGHBOURS];
        for (d, edge) in edges.iter_mut().enumerate() {
            let i = base + d;
            *edge = Edge {
                p: f64::from_bits(self.p[i].load(Ordering::Relaxed)),
                transition: self.transition[i].load(Ordering::Relaxed) as i64,
                ros: f32::from_bits(self.ros[i].load(Ordering::Relaxed)),
                fli: f32::from_bits(self.fli[i].load(Ordering::Relaxed)),
            };
        }
        edges
    }

    #[inline]
    fn write(&self, local: usize, edges: &[Edge; N_NEIGHBOURS]) {
        let base = local * N_NEIGHBOURS;
        for (d, edge) in edges.iter().enumerate() {
            let i = base + d;
            self.p[i].store(edge.p.to_bits(), Ordering::Relaxed);
            self.transition[i].store(edge.transition as u32, Ordering::Relaxed);
            self.ros[i].store(edge.ros.to_bits(), Ordering::Relaxed);
            self.fli[i].store(edge.fli.to_bits(), Ordering::Relaxed);
        }
    }
}

/// Block-sparse, generation-stamped cache of the edges of a grid.
pub struct EdgeCache {
    tiles_h: usize,
    tiles_w: usize,
    /// generation of the inputs the cache currently describes (never
    /// `UNFILLED`)
    generation: u32,
    tiles: Vec<OnceLock<Box<EdgeTile>>>,
}

impl EdgeCache {
    /// An empty cache for a `rows x cols` grid.
    pub fn new(rows: usize, cols: usize) -> EdgeCache {
        let tiles_h = rows.div_ceil(TILE_SIZE);
        let tiles_w = cols.div_ceil(TILE_SIZE);
        EdgeCache {
            tiles_h,
            tiles_w,
            generation: UNFILLED + 1,
            tiles: (0..tiles_h * tiles_w).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Current generation of the inputs.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Mark every cached edge stale (the inputs changed). O(1): tiles stay
    /// allocated and are refilled lazily.
    pub fn invalidate(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == UNFILLED {
            // wrapped around: stamps from the first generation could now
            // read as fresh, so drop everything
            self.generation = UNFILLED + 1;
            for tile in &mut self.tiles {
                *tile = OnceLock::new();
            }
        }
    }

    /// Drop everything and adopt a new grid shape (tile indices change).
    pub fn reset(&mut self, rows: usize, cols: usize) {
        *self = EdgeCache::new(rows, cols);
    }

    /// Free the tile covering block `(tile_row, tile_col)`; it is
    /// reallocated on the next touch.
    pub fn release_tile(&mut self, tile_row: usize, tile_col: usize) {
        self.tiles[tile_row * self.tiles_w + tile_col] = OnceLock::new();
    }

    /// Number of allocated tiles.
    pub fn allocated_tiles(&self) -> usize {
        self.tiles.iter().filter(|t| t.get().is_some()).count()
    }

    /// Bytes held by the allocated tiles.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_tiles() * EdgeTile::BYTES
    }

    /// The eight edges leaving `(row, col)`, filling the cell with `fill`
    /// when it is missing or stale. Safe to call concurrently.
    #[inline]
    pub fn get_or_fill(
        &self,
        row: usize,
        col: usize,
        fill: impl FnOnce() -> [Edge; N_NEIGHBOURS],
    ) -> [Edge; N_NEIGHBOURS] {
        let tile_row = row >> TILE_SHIFT;
        let tile_col = col >> TILE_SHIFT;
        let tile = self.tiles[tile_row * self.tiles_w + tile_col].get_or_init(EdgeTile::new);
        let local = local_index(row, col);
        if tile.ready[local].load(Ordering::Acquire) == self.generation {
            return tile.read(local);
        }
        let mut edges = fill();
        for edge in &mut edges {
            // stored as u32; never reached within a simulation (arrival
            // times are i32), so saturating keeps every observable the same
            edge.transition = edge.transition.clamp(0, u32::MAX as i64);
        }
        tile.write(local, &edges);
        tile.ready[local].store(self.generation, Ordering::Release);
        edges
    }

    /// Tile-grid shape.
    pub fn tiles_shape(&self) -> (usize, usize) {
        (self.tiles_h, self.tiles_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn sample(seed: f64) -> [Edge; N_NEIGHBOURS] {
        let mut edges = [Edge::BLOCKED; N_NEIGHBOURS];
        for (d, edge) in edges.iter_mut().enumerate() {
            *edge = Edge {
                p: seed + d as f64 * 0.01,
                transition: 100 + d as i64,
                ros: 1.5 + d as f32,
                fli: 2.5 + d as f32,
            };
        }
        edges[3] = Edge::BLOCKED;
        edges
    }

    #[test]
    fn fills_once_then_reads_back_identical_edges() {
        let cache = EdgeCache::new(100, 70);
        assert_eq!(cache.tiles_shape(), (4, 3));
        let calls = Cell::new(0);
        let fill = || {
            calls.set(calls.get() + 1);
            sample(0.25)
        };
        let first = cache.get_or_fill(40, 65, fill);
        let second = cache.get_or_fill(40, 65, fill);
        assert_eq!(calls.get(), 1);
        assert_eq!(first, sample(0.25));
        assert_eq!(second, first);
        assert!(second[3].is_blocked());
        assert_eq!(cache.allocated_tiles(), 1);
        assert_eq!(cache.allocated_bytes(), EdgeTile::BYTES);
    }

    #[test]
    fn invalidate_refills_without_reallocating() {
        let mut cache = EdgeCache::new(64, 64);
        let calls = Cell::new(0);
        let fill = || {
            calls.set(calls.get() + 1);
            sample(0.5)
        };
        cache.get_or_fill(1, 1, fill);
        cache.get_or_fill(1, 2, fill);
        assert_eq!(calls.get(), 2);
        cache.invalidate();
        assert_eq!(cache.allocated_tiles(), 1);
        cache.get_or_fill(1, 1, fill);
        assert_eq!(calls.get(), 3);
        // the neighbour cell is still stale, the refilled one is fresh
        cache.get_or_fill(1, 1, fill);
        assert_eq!(calls.get(), 3);
        cache.get_or_fill(1, 2, fill);
        assert_eq!(calls.get(), 4);
    }

    #[test]
    fn release_and_reset_drop_tiles() {
        let mut cache = EdgeCache::new(64, 64);
        cache.get_or_fill(0, 0, || sample(0.1));
        cache.get_or_fill(40, 40, || sample(0.1));
        assert_eq!(cache.allocated_tiles(), 2);
        cache.release_tile(0, 0);
        assert_eq!(cache.allocated_tiles(), 1);
        cache.reset(128, 128);
        assert_eq!(cache.allocated_tiles(), 0);
        assert_eq!(cache.tiles_shape(), (4, 4));
    }

    #[test]
    fn transition_saturates_to_u32() {
        let cache = EdgeCache::new(32, 32);
        let mut edges = sample(0.2);
        edges[0].transition = i64::MAX;
        let read = cache.get_or_fill(0, 0, || edges);
        assert_eq!(read[0].transition, u32::MAX as i64);
        assert_eq!(read[1], edges[1]);
    }

    #[test]
    fn concurrent_fill_is_consistent() {
        let cache = EdgeCache::new(64, 64);
        std::thread::scope(|scope| {
            for t in 0..8 {
                let cache = &cache;
                scope.spawn(move || {
                    for i in 0..64usize {
                        let edges = cache.get_or_fill(i, (i + t) % 64, || sample(i as f64));
                        assert_eq!(edges, sample(i as f64));
                    }
                });
            }
        });
    }
}
