//! Shared cache of the deterministic spread quantities of every
//! cell -> neighbour edge.
//!
//! For a given set of inputs (vegetation, fuels, DEM, wind, moisture) the
//! ignition probability, transition time and the ROS / fireline intensity
//! carried into a neighbour depend only on the source cell and the
//! direction, not on the realization. The heap kernel therefore evaluates
//! the spread models once per edge per *generation* of the inputs and
//! every realization reuses the result; only the Bernoulli draw stays per
//! realization.
//!
//! The cache is block-sparse like the state tiles: a 32x32 tile is
//! allocated the first time any realization pops a cell inside it. Edges
//! are filled lazily and in two stages, probability first and fire
//! behaviour only once a draw accepts the edge, so the cached kernel never
//! evaluates more of the models than the uncached one did: the first
//! realization to need a quantity pays for it, the others read it. Each
//! cell carries the generation stamp its entries are valid for, so
//! invalidating after a boundary-condition change is O(1) and needs no
//! reallocation.
//!
//! All accesses go through atomics so the realization threads can fill
//! and read concurrently without locks: every edge quantity has its own
//! word (or its own half of a word, updated with bitwise RMWs), a cell's
//! readiness bits are published with one compare-and-swap per lookup, and
//! a race between two fillers writes identical bits. A cell's record is 21
//! consecutive 64-bit words, so a lookup touches at most three cache lines.
//! With a single realization the cache is bypassed: a cell is popped at
//! most once per realization, so nothing would be reused.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::kernel::N_NEIGHBOURS;
use crate::tiles::{local_index, TILE_CELLS, TILE_SHIFT, TILE_SIZE};

/// Words per cell record: 8 x `p` (f64 bits), 8 x (`transition` << 32 |
/// `ros`), 4 x two packed `fli` f32, 1 stamp (`generation` << 32 | ready
/// mask: bit `d` = probability of edge `d` ready, bit `8 + d` = behaviour).
const WORDS_PER_CELL: usize = N_NEIGHBOURS + N_NEIGHBOURS + N_NEIGHBOURS / 2 + 1;
const P_OFFSET: usize = 0;
const TR_OFFSET: usize = N_NEIGHBOURS;
const FLI_OFFSET: usize = 2 * N_NEIGHBOURS;
const STAMP_OFFSET: usize = WORDS_PER_CELL - 1;
const MASK_BITS: u64 = 0xffff;

/// Bytes held by one allocated 32x32 tile of the cache: a packed 168-byte
/// record per cell. Shared by all realizations (not multiplied by their
/// count).
pub const EDGE_TILE_BYTES: usize = TILE_CELLS * WORDS_PER_CELL * 8;

/// Generation stamp meaning "never filled".
const UNFILLED: u32 = 0;

/// Probability stored for an edge fire can never cross (target off-grid,
/// unburnable or not a spreading fuel).
pub const BLOCKED: f64 = -1.0;

/// One tile of cached edges: `TILE_CELLS` consecutive cell records.
struct EdgeTile {
    words: Box<[AtomicU64]>,
}

impl EdgeTile {
    fn new() -> Box<EdgeTile> {
        Box::new(EdgeTile {
            words: (0..TILE_CELLS * WORDS_PER_CELL)
                .map(|_| AtomicU64::new(0))
                .collect(),
        })
    }

    #[inline]
    fn record(&self, local: usize) -> &[AtomicU64] {
        let base = local * WORDS_PER_CELL;
        &self.words[base..base + WORDS_PER_CELL]
    }
}

/// Block-sparse, generation-stamped cache of the edges of a grid.
pub struct EdgeCache {
    tiles_h: usize,
    tiles_w: usize,
    /// generation of the inputs the cache currently describes (never
    /// `UNFILLED`)
    generation: u32,
    /// compute every lookup instead of caching (a single realization pops
    /// each cell once per generation, so there is nothing to reuse)
    bypass: bool,
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
            bypass: false,
            tiles: (0..tiles_h * tiles_w).map(|_| OnceLock::new()).collect(),
        }
    }

    /// A cache for `realizations` realizations: caching is bypassed for a
    /// single realization, where no edge is ever looked up twice.
    pub fn for_realizations(rows: usize, cols: usize, realizations: usize) -> EdgeCache {
        let mut cache = EdgeCache::new(rows, cols);
        cache.bypass = realizations <= 1;
        cache
    }

    /// `true` when lookups are computed rather than cached.
    pub fn is_bypassed(&self) -> bool {
        self.bypass
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

    /// Mark the records of the given cells stale, leaving everything else
    /// valid. For a change confined to a few cells (decaying action
    /// moisture) this replaces a full [`invalidate`](Self::invalidate).
    /// Cells in unallocated tiles need nothing.
    pub fn invalidate_cells(&mut self, cells: impl IntoIterator<Item = (usize, usize)>) {
        if self.bypass {
            return;
        }
        for (row, col) in cells {
            let tile_row = row >> TILE_SHIFT;
            let tile_col = col >> TILE_SHIFT;
            if let Some(tile) = self.tiles[tile_row * self.tiles_w + tile_col].get() {
                let record = tile.record(local_index(row, col));
                record[STAMP_OFFSET].store((UNFILLED as u64) << 32, Ordering::Release);
            }
        }
    }

    /// Drop everything and adopt a new grid shape (tile indices change).
    pub fn reset(&mut self, rows: usize, cols: usize) {
        let bypass = self.bypass;
        *self = EdgeCache::new(rows, cols);
        self.bypass = bypass;
    }

    /// Free the tile covering block `(tile_row, tile_col)`; it is
    /// reallocated on the next touch.
    pub fn release_tile(&mut self, tile_row: usize, tile_col: usize) {
        self.tiles[tile_row * self.tiles_w + tile_col] = OnceLock::new();
    }

    /// Whether the tile covering block `(tile_row, tile_col)` is allocated.
    pub fn is_allocated(&self, tile_row: usize, tile_col: usize) -> bool {
        self.tiles[tile_row * self.tiles_w + tile_col].get().is_some()
    }

    /// Number of allocated tiles.
    pub fn allocated_tiles(&self) -> usize {
        self.tiles.iter().filter(|t| t.get().is_some()).count()
    }

    /// Bytes held by the allocated tiles.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_tiles() * EDGE_TILE_BYTES
    }

    /// Tile-grid shape.
    pub fn tiles_shape(&self) -> (usize, usize) {
        (self.tiles_h, self.tiles_w)
    }

    /// Open the record of `(row, col)` for reading and lazy filling. Call
    /// [`CellEdges::commit`] afterwards to publish what was filled. Safe to
    /// call concurrently for the same cell.
    #[inline]
    pub fn lookup(&self, row: usize, col: usize) -> CellEdges<'_> {
        if self.bypass {
            return CellEdges {
                record: None,
                generation: self.generation,
                mask: 0,
                new_bits: 0,
            };
        }
        let tile_row = row >> TILE_SHIFT;
        let tile_col = col >> TILE_SHIFT;
        let tile = self.tiles[tile_row * self.tiles_w + tile_col].get_or_init(EdgeTile::new);
        let record = tile.record(local_index(row, col));
        let stamp = record[STAMP_OFFSET].load(Ordering::Acquire);
        let mask = if (stamp >> 32) as u32 == self.generation {
            (stamp & MASK_BITS) as u16
        } else {
            0
        };
        CellEdges {
            record: Some(record),
            generation: self.generation,
            mask,
            new_bits: 0,
        }
    }
}

/// One cell's cached edges, opened by [`EdgeCache::lookup`]. Readers see
/// what the stamp published at open time; what this handle fills becomes
/// visible to others on `commit`.
pub struct CellEdges<'a> {
    /// `None` when the cache is bypassed
    record: Option<&'a [AtomicU64]>,
    generation: u32,
    /// ready bits valid for `generation` at open time
    mask: u16,
    /// ready bits filled through this handle
    new_bits: u16,
}

impl CellEdges<'_> {
    /// Probability of edge `d` (`BLOCKED` when fire cannot cross it), if
    /// cached.
    #[inline]
    pub fn p(&self, d: usize) -> Option<f64> {
        let record = self.record?;
        if self.mask & (1 << d) == 0 {
            return None;
        }
        Some(f64::from_bits(record[P_OFFSET + d].load(Ordering::Relaxed)))
    }

    /// Cache the probability of edge `d`.
    #[inline]
    pub fn set_p(&mut self, d: usize, p: f64) {
        let Some(record) = self.record else {
            return;
        };
        record[P_OFFSET + d].store(p.to_bits(), Ordering::Relaxed);
        self.new_bits |= 1 << d;
    }

    /// (transition time [s], ROS [m/min], FLI [kW/m]) of edge `d`, if
    /// cached.
    #[inline]
    pub fn behavior(&self, d: usize) -> Option<(i64, f32, f32)> {
        let record = self.record?;
        if self.mask & (1 << (N_NEIGHBOURS + d)) == 0 {
            return None;
        }
        let tr = record[TR_OFFSET + d].load(Ordering::Relaxed);
        let fli_pair = record[FLI_OFFSET + d / 2].load(Ordering::Relaxed);
        let fli = (fli_pair >> (32 * (d % 2))) as u32;
        Some((
            (tr >> 32) as i64,
            f32::from_bits(tr as u32),
            f32::from_bits(fli),
        ))
    }

    /// Cache the fire behaviour of edge `d`. The transition time is stored
    /// as `u32`; saturating is unobservable within a simulation (arrival
    /// times are `i32`), and callers must use the returned value so that a
    /// first fill and later reads agree.
    #[inline]
    pub fn set_behavior(&mut self, d: usize, transition: i64, ros: f32, fli: f32) -> i64 {
        let transition = transition.clamp(0, u32::MAX as i64);
        let Some(record) = self.record else {
            return transition;
        };
        let tr = (transition as u64) << 32 | ros.to_bits() as u64;
        record[TR_OFFSET + d].store(tr, Ordering::Relaxed);
        // the fli word is shared with the paired edge, which may be filled
        // concurrently: replace only this half, atomically, so a reader
        // never observes a cleared half
        let shift = 32 * (d % 2);
        let half = 0xffff_ffffu64 << shift;
        let value = (fli.to_bits() as u64) << shift;
        let _ =
            record[FLI_OFFSET + d / 2].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |word| {
                Some((word & !half) | value)
            });
        self.new_bits |= 1 << (N_NEIGHBOURS + d);
        transition
    }

    /// Publish the entries filled through this handle.
    #[inline]
    pub fn commit(self) {
        let Some(record) = self.record else {
            return;
        };
        if self.new_bits == 0 {
            return;
        }
        let generation = (self.generation as u64) << 32;
        let bits = self.new_bits as u64;
        // merge with bits other threads published meanwhile; a stale stamp
        // (older generation) is replaced outright
        let _ = record[STAMP_OFFSET].fetch_update(Ordering::AcqRel, Ordering::Acquire, |stamp| {
            if stamp & !MASK_BITS == generation {
                Some(stamp | bits)
            } else {
                Some(generation | bits)
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_all(cache: &EdgeCache, row: usize, col: usize, seed: f64) -> usize {
        let mut edges = cache.lookup(row, col);
        let mut computed = 0;
        for d in 0..N_NEIGHBOURS {
            if edges.p(d).is_none() {
                edges.set_p(
                    d,
                    if d == 3 {
                        BLOCKED
                    } else {
                        seed + d as f64 * 0.01
                    },
                );
                computed += 1;
            }
            if edges.behavior(d).is_none() {
                edges.set_behavior(d, 100 + d as i64, 1.5 + d as f32, 2.5 + d as f32);
                computed += 1;
            }
        }
        edges.commit();
        computed
    }

    fn check_all(cache: &EdgeCache, row: usize, col: usize, seed: f64) {
        let edges = cache.lookup(row, col);
        for d in 0..N_NEIGHBOURS {
            let expected = if d == 3 {
                BLOCKED
            } else {
                seed + d as f64 * 0.01
            };
            assert_eq!(edges.p(d), Some(expected), "p of edge {d}");
            assert_eq!(
                edges.behavior(d),
                Some((100 + d as i64, 1.5 + d as f32, 2.5 + d as f32)),
                "behaviour of edge {d}"
            );
        }
    }

    #[test]
    fn fills_once_then_reads_back_identical_edges() {
        let cache = EdgeCache::new(100, 70);
        assert_eq!(cache.tiles_shape(), (4, 3));
        assert_eq!(fill_all(&cache, 40, 65, 0.25), 16);
        assert_eq!(fill_all(&cache, 40, 65, 0.25), 0);
        check_all(&cache, 40, 65, 0.25);
        assert_eq!(cache.allocated_tiles(), 1);
        assert_eq!(cache.allocated_bytes(), EDGE_TILE_BYTES);
    }

    #[test]
    fn probability_and_behaviour_fill_independently() {
        let cache = EdgeCache::new(64, 64);
        let mut edges = cache.lookup(2, 2);
        edges.set_p(5, 0.4);
        edges.commit();
        let mut edges = cache.lookup(2, 2);
        assert_eq!(edges.p(5), Some(0.4));
        assert_eq!(edges.p(4), None);
        assert_eq!(edges.behavior(5), None);
        // the paired edge 4 shares the fli word with edge 5
        edges.set_behavior(4, 7, 0.5, 9.0);
        edges.commit();
        let mut edges = cache.lookup(2, 2);
        assert_eq!(edges.behavior(4), Some((7, 0.5, 9.0)));
        assert_eq!(edges.behavior(5), None);
        edges.set_behavior(5, 8, 0.6, 10.0);
        edges.commit();
        let edges = cache.lookup(2, 2);
        assert_eq!(edges.behavior(4), Some((7, 0.5, 9.0)));
        assert_eq!(edges.behavior(5), Some((8, 0.6, 10.0)));
    }

    #[test]
    fn uncommitted_fills_are_not_visible() {
        let cache = EdgeCache::new(64, 64);
        {
            let mut edges = cache.lookup(1, 1);
            edges.set_p(0, 0.9);
            // dropped without commit
        }
        assert_eq!(cache.lookup(1, 1).p(0), None);
    }

    #[test]
    fn invalidate_refills_without_reallocating() {
        let mut cache = EdgeCache::new(64, 64);
        assert_eq!(fill_all(&cache, 1, 1, 0.5), 16);
        assert_eq!(fill_all(&cache, 1, 2, 0.5), 16);
        cache.invalidate();
        assert_eq!(cache.allocated_tiles(), 1);
        assert_eq!(cache.lookup(1, 1).p(0), None);
        assert_eq!(fill_all(&cache, 1, 1, 0.7), 16);
        check_all(&cache, 1, 1, 0.7);
        // the neighbour cell is still stale
        assert_eq!(fill_all(&cache, 1, 2, 0.7), 16);
    }

    #[test]
    fn invalidate_cells_is_local() {
        let mut cache = EdgeCache::new(64, 64);
        fill_all(&cache, 1, 1, 0.5);
        fill_all(&cache, 1, 2, 0.5);
        fill_all(&cache, 40, 40, 0.5);
        // cells in unallocated tiles are fine to name
        cache.invalidate_cells([(1, 1), (40, 40), (33, 1)]);
        assert_eq!(cache.allocated_tiles(), 2);
        assert_eq!(cache.lookup(1, 1).p(0), None);
        assert_eq!(cache.lookup(40, 40).p(0), None);
        check_all(&cache, 1, 2, 0.5);
        assert_eq!(fill_all(&cache, 1, 1, 0.6), 16);
        check_all(&cache, 1, 1, 0.6);
    }

    #[test]
    fn release_and_reset_drop_tiles() {
        let mut cache = EdgeCache::new(64, 64);
        fill_all(&cache, 0, 0, 0.1);
        fill_all(&cache, 40, 40, 0.1);
        assert_eq!(cache.allocated_tiles(), 2);
        cache.release_tile(0, 0);
        assert_eq!(cache.allocated_tiles(), 1);
        cache.reset(128, 128);
        assert_eq!(cache.allocated_tiles(), 0);
        assert_eq!(cache.tiles_shape(), (4, 4));
    }

    #[test]
    fn single_realization_bypasses_the_cache() {
        let cache = EdgeCache::for_realizations(64, 64, 1);
        assert!(cache.is_bypassed());
        assert_eq!(fill_all(&cache, 5, 5, 0.3), 16);
        assert_eq!(fill_all(&cache, 5, 5, 0.3), 16);
        assert_eq!(cache.allocated_tiles(), 0);
        let mut cache = cache;
        cache.reset(32, 32);
        assert!(cache.is_bypassed());
        assert!(!EdgeCache::for_realizations(64, 64, 2).is_bypassed());
    }

    #[test]
    fn record_layout_is_three_cache_lines() {
        assert_eq!(WORDS_PER_CELL * 8, 168);
        assert_eq!(EDGE_TILE_BYTES, TILE_CELLS * 168);
    }

    #[test]
    fn transition_saturates_to_u32() {
        let cache = EdgeCache::new(32, 32);
        let mut edges = cache.lookup(0, 0);
        assert_eq!(edges.set_behavior(0, i64::MAX, 1.0, 2.0), u32::MAX as i64);
        assert_eq!(edges.set_behavior(1, 42, 1.0, 2.0), 42);
        edges.commit();
        let edges = cache.lookup(0, 0);
        assert_eq!(edges.behavior(0), Some((u32::MAX as i64, 1.0, 2.0)));
        assert_eq!(edges.behavior(1), Some((42, 1.0, 2.0)));
    }

    #[test]
    fn concurrent_fill_is_consistent() {
        let cache = EdgeCache::new(64, 64);
        std::thread::scope(|scope| {
            for t in 0..8 {
                let cache = &cache;
                scope.spawn(move || {
                    for i in 0..64usize {
                        // each thread fills a different edge of the same
                        // cells first, then everything that is missing
                        let mut edges = cache.lookup(i, i);
                        let d = t % N_NEIGHBOURS;
                        if edges.p(d).is_none() {
                            edges.set_p(
                                d,
                                if d == 3 {
                                    BLOCKED
                                } else {
                                    i as f64 + d as f64 * 0.01
                                },
                            );
                        }
                        if edges.behavior(d).is_none() {
                            edges.set_behavior(d, 100 + d as i64, 1.5 + d as f32, 2.5 + d as f32);
                        }
                        edges.commit();
                        fill_all(cache, i, i, i as f64);
                        check_all(cache, i, i, i as f64);
                    }
                });
            }
        });
        for i in 0..64usize {
            check_all(&cache, i, i, i as f64);
        }
    }
}
