//! Sparse 8-connected reachability over one realization's tiled state.
//!
//! The freeze and thaw decisions need to know which cells unburnt fuel
//! connects to a seed (a pending front event, or a new ignition). Instead
//! of flooding full-grid boolean masks, the flood runs over per-tile
//! 1024-bit masks of the *allocated* tiles only (live tiles plus frozen
//! tiles with open cells, from their [`FrozenSummary`]):
//!
//! - every unallocated tile is unburnt, so all its fuel cells are open;
//!   all unallocated territory collapses into one virtual **sink** node.
//!   The flood never descends into it: stepping from an open cell into a
//!   fuel cell of an unallocated tile, or a seed landing in one, marks the
//!   sink reached;
//! - once the sink is reached, every open cell 8-adjacent to a fuel cell
//!   of an unallocated tile is reached too, and the flood continues from
//!   there. This merges all unallocated regions into one component.
//!
//! The result is a deliberate **over**-approximation of exact cell-level
//! reachability (any exact path through unallocated cells maps to a path
//! through the sink), never an under-approximation, which is what keeps
//! freezing and thawing safe.

use crate::fuels::NO_FUEL;
use crate::grid::Grid2;
use crate::tiles::{
    FrozenSummary, RowMask, Tile, TileGrid, FLAG_FIRE, FROZEN_TILE, TILE_MASK, TILE_SHIFT,
    TILE_SIZE,
};

pub(crate) const EMPTY: RowMask = [0; TILE_SIZE];
const LAST: usize = TILE_SIZE - 1;
const HIGH_BIT: u32 = 1 << LAST;

/// Fuel cells (in-domain, `veg != NO_FUEL`) of every tile position.
/// Realization-independent; padding cells past the domain edge are 0.
#[derive(Clone)]
pub(crate) struct FuelMasks {
    tiles_h: usize,
    tiles_w: usize,
    masks: Vec<RowMask>,
}

impl FuelMasks {
    pub fn build(veg: &Grid2<i32>) -> FuelMasks {
        let (rows, cols) = veg.shape();
        let tiles_h = rows.div_ceil(TILE_SIZE);
        let tiles_w = cols.div_ceil(TILE_SIZE);
        let mut masks = vec![EMPTY; tiles_h * tiles_w];
        let data = veg.as_slice();
        for row in 0..rows {
            let base = (row >> TILE_SHIFT) * tiles_w;
            let local_row = row & TILE_MASK;
            for (col, &code) in data[row * cols..(row + 1) * cols].iter().enumerate() {
                if code != NO_FUEL {
                    masks[base + (col >> TILE_SHIFT)][local_row] |= 1 << (col & TILE_MASK);
                }
            }
        }
        FuelMasks {
            tiles_h,
            tiles_w,
            masks,
        }
    }

    #[inline]
    pub fn get(&self, tile_row: usize, tile_col: usize) -> &RowMask {
        &self.masks[tile_row * self.tiles_w + tile_col]
    }
}

#[inline]
pub(crate) fn is_empty(mask: &RowMask) -> bool {
    mask.iter().all(|&row| row == 0)
}

/// Open cells (in-domain fuel, unburnt) of a live tile.
pub(crate) fn live_open(tile: &Tile, fuel: &RowMask) -> RowMask {
    let mut open = *fuel;
    for (local_row, bits) in open.iter_mut().enumerate() {
        if *bits == 0 {
            continue;
        }
        let flags = &tile.flags[local_row * TILE_SIZE..(local_row + 1) * TILE_SIZE];
        let mut burnt = 0u32;
        for (local_col, &flag) in flags.iter().enumerate() {
            burnt |= ((flag & FLAG_FIRE != 0) as u32) << local_col;
        }
        *bits &= !burnt;
    }
    open
}

#[inline]
fn dilate(bits: u32) -> u32 {
    bits | (bits << 1) | (bits >> 1)
}

/// Bits of `open` in runs (horizontal, toward higher columns) that start
/// at or below a seed: the carry of `open + seeds` ripples through each
/// run from its lowest seed.
#[inline]
fn fill_up(seeds: u32, open: u32) -> u32 {
    let (seeds, open) = (seeds as u64, open as u64);
    ((((open + seeds) ^ open) | seeds) & open) as u32
}

/// Every cell of `open` in the same horizontal run as a seed.
#[inline]
fn fill_row(seeds: u32, open: u32) -> u32 {
    let seeds = seeds & open;
    if seeds == 0 {
        return 0;
    }
    fill_up(seeds, open)
        | fill_up(seeds.reverse_bits(), open.reverse_bits()).reverse_bits()
}

/// Close `reach` under 8-connectivity within the tile's `open` cells.
fn fill_tile(reach: &mut RowMask, open: &RowMask) {
    loop {
        for i in 0..TILE_SIZE {
            let mut seeds = reach[i];
            if i > 0 {
                seeds |= dilate(reach[i - 1]);
            }
            reach[i] = fill_row(seeds, open[i]);
        }
        let mut changed_up = false;
        for i in (0..LAST).rev() {
            let filled = fill_row(reach[i] | dilate(reach[i + 1]), open[i]);
            if filled != reach[i] {
                reach[i] = filled;
                changed_up = true;
            }
        }
        // a downward sweep after an unchanged upward one adds nothing new
        if !changed_up {
            return;
        }
    }
}

/// The 8 tile-neighbour offsets.
const DIRS: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// Cells of the tile at offset `(dr, dc)` from `src`'s tile that are
/// 8-adjacent to a set cell of `src` (only border cells can be).
fn spill(src: &RowMask, dr: isize, dc: isize) -> RowMask {
    let mut out = EMPTY;
    match (dr, dc) {
        (-1, 0) => out[LAST] = dilate(src[0]),
        (1, 0) => out[0] = dilate(src[LAST]),
        (0, -1) | (0, 1) => {
            // src's west (east) column against the neighbour's east (west)
            let (from, to) = if dc < 0 { (1, HIGH_BIT) } else { (HIGH_BIT, 1) };
            for i in 0..TILE_SIZE {
                let mut near = src[i];
                if i > 0 {
                    near |= src[i - 1];
                }
                if i < LAST {
                    near |= src[i + 1];
                }
                if near & from != 0 {
                    out[i] = to;
                }
            }
        }
        (-1, -1) if src[0] & 1 != 0 => out[LAST] = HIGH_BIT,
        (-1, 1) if src[0] & HIGH_BIT != 0 => out[LAST] = 1,
        (1, -1) if src[LAST] & 1 != 0 => out[0] = HIGH_BIT,
        (1, 1) if src[LAST] & HIGH_BIT != 0 => out[0] = 1,
        _ => {}
    }
    out
}

#[inline]
fn border_empty(mask: &RowMask) -> bool {
    const EDGE: u32 = 1 | HIGH_BIT;
    mask[0] == 0 && mask[LAST] == 0 && mask.iter().all(|&row| row & EDGE == 0)
}

const SINK: i32 = -1;
const WALL: i32 = -2;

/// Sparse flood over one realization's tiles (see the module docs).
pub(crate) struct Reach<'a> {
    fuel: &'a FuelMasks,
    tiles_h: usize,
    tiles_w: usize,
    /// per tile position: node id, `SINK` (unallocated with fuel) or
    /// `WALL` (no open cell)
    pos: Vec<i32>,
    at: Vec<(usize, usize)>,
    open: Vec<RowMask>,
    /// open cells 8-adjacent to a fuel cell of an unallocated tile
    sink_adj: Vec<RowMask>,
    reach: Vec<RowMask>,
    queued: Vec<bool>,
    work: Vec<u32>,
    sink: bool,
}

impl<'a> Reach<'a> {
    /// Graph of a realization's live and frozen tiles. A frozen tile
    /// without a summary is treated as fully open (conservative).
    pub fn build(tiles: &TileGrid, fuel: &'a FuelMasks) -> Reach<'a> {
        let (tiles_h, tiles_w) = tiles.tiles_shape();
        debug_assert_eq!((tiles_h, tiles_w), (fuel.tiles_h, fuel.tiles_w));
        let mut pos = vec![WALL; tiles_h * tiles_w];
        let mut at = Vec::new();
        let mut open = Vec::new();
        for tile_row in 0..tiles_h {
            for tile_col in 0..tiles_w {
                let slot = tiles.slot_of(tile_row, tile_col);
                let fuel_mask = fuel.get(tile_row, tile_col);
                let mask = if slot >= 0 {
                    live_open(tiles.tile(slot), fuel_mask)
                } else if slot == FROZEN_TILE {
                    match tiles.frozen_summary(tile_row, tile_col) {
                        Some(FrozenSummary::Solid) => EMPTY,
                        Some(FrozenSummary::Open(mask)) => **mask,
                        None => *fuel_mask,
                    }
                } else {
                    if !is_empty(fuel_mask) {
                        pos[tile_row * tiles_w + tile_col] = SINK;
                    }
                    continue;
                };
                if is_empty(&mask) {
                    continue;
                }
                pos[tile_row * tiles_w + tile_col] = at.len() as i32;
                at.push((tile_row, tile_col));
                open.push(mask);
            }
        }
        let n = at.len();
        let mut reach = Reach {
            fuel,
            tiles_h,
            tiles_w,
            pos,
            at,
            open,
            sink_adj: vec![EMPTY; n],
            reach: vec![EMPTY; n],
            queued: vec![false; n],
            work: Vec::new(),
            sink: false,
        };
        for node in 0..n {
            let (tile_row, tile_col) = reach.at[node];
            let mut adj = EMPTY;
            for &(dr, dc) in &DIRS {
                if let Some((nr, nc)) = reach.neighbour(tile_row, tile_col, dr, dc) {
                    if reach.pos[nr * tiles_w + nc] == SINK {
                        let near = spill(fuel.get(nr, nc), -dr, -dc);
                        for (a, b) in adj.iter_mut().zip(near.iter()) {
                            *a |= b;
                        }
                    }
                }
            }
            for (a, b) in adj.iter_mut().zip(reach.open[node].iter()) {
                *a &= b;
            }
            reach.sink_adj[node] = adj;
        }
        reach
    }

    #[inline]
    fn neighbour(
        &self,
        tile_row: usize,
        tile_col: usize,
        dr: isize,
        dc: isize,
    ) -> Option<(usize, usize)> {
        let nr = tile_row as isize + dr;
        let nc = tile_col as isize + dc;
        if nr < 0 || nc < 0 || nr as usize >= self.tiles_h || nc as usize >= self.tiles_w {
            return None;
        }
        Some((nr as usize, nc as usize))
    }

    #[inline]
    fn node(&self, tile_row: usize, tile_col: usize) -> i32 {
        if tile_row >= self.tiles_h || tile_col >= self.tiles_w {
            return WALL;
        }
        self.pos[tile_row * self.tiles_w + tile_col]
    }

    fn enqueue(&mut self, node: usize) {
        if !self.queued[node] {
            self.queued[node] = true;
            self.work.push(node as u32);
        }
    }

    /// Seed the flood at a cell; ignored unless it is open.
    pub fn seed_cell(&mut self, row: usize, col: usize) {
        let (tile_row, tile_col) = (row >> TILE_SHIFT, col >> TILE_SHIFT);
        let (local_row, bit) = (row & TILE_MASK, 1u32 << (col & TILE_MASK));
        match self.node(tile_row, tile_col) {
            SINK => {
                if self.fuel.get(tile_row, tile_col)[local_row] & bit != 0 {
                    self.seed_sink();
                }
            }
            WALL => {}
            node => {
                let node = node as usize;
                if self.open[node][local_row] & bit != 0
                    && self.reach[node][local_row] & bit == 0
                {
                    self.reach[node][local_row] |= bit;
                    self.enqueue(node);
                }
            }
        }
    }

    /// Mark the sink reached: every open cell bordering unallocated fuel
    /// becomes reached.
    pub fn seed_sink(&mut self) {
        if self.sink {
            return;
        }
        self.sink = true;
        for node in 0..self.at.len() {
            let mut changed = false;
            for i in 0..TILE_SIZE {
                let add = self.sink_adj[node][i] & !self.reach[node][i];
                if add != 0 {
                    self.reach[node][i] |= add;
                    changed = true;
                }
            }
            if changed {
                self.enqueue(node);
            }
        }
    }

    /// Run the flood to its fixed point.
    pub fn run(&mut self) {
        while let Some(node) = self.work.pop() {
            let node = node as usize;
            self.queued[node] = false;
            let open = self.open[node];
            let mut reach = self.reach[node];
            fill_tile(&mut reach, &open);
            self.reach[node] = reach;
            if !self.sink
                && reach
                    .iter()
                    .zip(self.sink_adj[node].iter())
                    .any(|(r, a)| r & a != 0)
            {
                self.seed_sink();
            }
            if border_empty(&reach) {
                continue;
            }
            let (tile_row, tile_col) = self.at[node];
            for &(dr, dc) in &DIRS {
                let Some((nr, nc)) = self.neighbour(tile_row, tile_col, dr, dc) else {
                    continue;
                };
                let other = self.pos[nr * self.tiles_w + nc];
                if other < 0 {
                    // walls stop the flood; the sink is handled above
                    continue;
                }
                let other = other as usize;
                let incoming = spill(&reach, dr, dc);
                let mut changed = false;
                let open = &self.open[other];
                for ((target, &bits), &allowed) in
                    self.reach[other].iter_mut().zip(incoming.iter()).zip(open.iter())
                {
                    let add = bits & allowed & !*target;
                    if add != 0 {
                        *target |= add;
                        changed = true;
                    }
                }
                if changed {
                    self.enqueue(other);
                }
            }
        }
    }

    /// Whether any cell of the tile was reached.
    pub fn reached(&self, tile_row: usize, tile_col: usize) -> bool {
        match self.node(tile_row, tile_col) {
            node if node >= 0 => !is_empty(&self.reach[node as usize]),
            _ => false,
        }
    }

    /// Reached cells of a tile (empty for walls and unallocated tiles).
    #[cfg(test)]
    pub fn reach_mask(&self, tile_row: usize, tile_col: usize) -> RowMask {
        match self.node(tile_row, tile_col) {
            node if node >= 0 => self.reach[node as usize],
            _ => EMPTY,
        }
    }

    /// Open cells of a live or frozen tile (empty for walls).
    pub fn open_mask(&self, tile_row: usize, tile_col: usize) -> RowMask {
        match self.node(tile_row, tile_col) {
            node if node >= 0 => self.open[node as usize],
            _ => EMPTY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tiles::{flood_reachable, local_index};

    #[test]
    fn fill_row_selects_seeded_runs() {
        let open = 0b0111_0110_1111u32;
        assert_eq!(fill_row(0b0000_0000_0100, open), 0b0000_0000_1111);
        assert_eq!(fill_row(0b0010_0000_0000, open), 0b0111_0000_0000);
        assert_eq!(fill_row(0b0010_0000_0101, open), 0b0111_0000_1111);
        assert_eq!(fill_row(0, open), 0);
        assert_eq!(fill_row(HIGH_BIT, u32::MAX), u32::MAX);
        assert_eq!(fill_row(1, u32::MAX), u32::MAX);
    }

    /// A random tiled scenario and its exact full-grid reachability.
    struct Scenario {
        veg: Grid2<i32>,
        tiles: TileGrid,
        seeds: Vec<(usize, usize)>,
    }

    fn random_scenario(seed: u64) -> Scenario {
        let mut rng = Rng::for_realization(seed, 0);
        // deliberately not a multiple of the tile size
        let rows = 32 * (2 + (rng.next_u64() % 5) as usize) - (rng.next_u64() % 20) as usize;
        let cols = 32 * (2 + (rng.next_u64() % 5) as usize) - (rng.next_u64() % 20) as usize;
        let no_fuel = rng.uniform() * 0.4;
        let mut veg = Grid2::filled(rows, cols, 4);
        for row in 0..rows {
            for col in 0..cols {
                if rng.uniform() < no_fuel {
                    veg[(row, col)] = NO_FUEL;
                }
            }
        }
        let fuel = FuelMasks::build(&veg);
        let mut tiles = TileGrid::new(rows, cols);
        let (tiles_h, tiles_w) = tiles.tiles_shape();
        let burn = rng.uniform();
        let mut frozen = Vec::new();
        for tile_row in 0..tiles_h {
            for tile_col in 0..tiles_w {
                let kind = rng.uniform();
                if kind < 0.3 {
                    continue; // unallocated
                }
                let slot = tiles.ensure_slot(tile_row << TILE_SHIFT, tile_col << TILE_SHIFT);
                let tile = tiles.tile_mut(slot);
                for local in 0..TILE_SIZE * TILE_SIZE {
                    if rng.uniform() < burn {
                        tile.flags[local] = FLAG_FIRE;
                    }
                }
                if kind > 0.75 {
                    let open = live_open(tiles.tile(slot), fuel.get(tile_row, tile_col));
                    frozen.push((tile_row, tile_col, slot, FrozenSummary::from_open(&open)));
                }
            }
        }
        tiles.freeze_tiles(frozen);
        let n_seeds = 1 + (rng.next_u64() % 6) as usize;
        let seeds = (0..n_seeds)
            .map(|_| {
                (
                    (rng.next_u64() % rows as u64) as usize,
                    (rng.next_u64() % cols as u64) as usize,
                )
            })
            .collect();
        Scenario { veg, tiles, seeds }
    }

    /// Exact reference: full-grid open mask (frozen tiles from their
    /// summaries) and an 8-connected flood from the seeds.
    fn exact_reach(scenario: &Scenario) -> (Vec<bool>, usize) {
        let (tiles_h, tiles_w) = scenario.tiles.tiles_shape();
        let grid_w = tiles_w * TILE_SIZE;
        let grid_h = tiles_h * TILE_SIZE;
        let fuel = FuelMasks::build(&scenario.veg);
        let mut open = vec![false; grid_h * grid_w];
        for tile_row in 0..tiles_h {
            for tile_col in 0..tiles_w {
                let slot = scenario.tiles.slot_of(tile_row, tile_col);
                let mask = if slot >= 0 {
                    live_open(scenario.tiles.tile(slot), fuel.get(tile_row, tile_col))
                } else if slot == FROZEN_TILE {
                    scenario.tiles.frozen_summary(tile_row, tile_col).unwrap().open()
                } else {
                    *fuel.get(tile_row, tile_col)
                };
                for (local_row, bits) in mask.iter().enumerate() {
                    for local_col in 0..TILE_SIZE {
                        let row = (tile_row << TILE_SHIFT) + local_row;
                        let col = (tile_col << TILE_SHIFT) + local_col;
                        open[row * grid_w + col] = bits >> local_col & 1 != 0;
                    }
                }
            }
        }
        let mut reachable = vec![false; grid_h * grid_w];
        flood_reachable(&open, grid_h, grid_w, &scenario.seeds, &mut reachable);
        (reachable, grid_w)
    }

    #[test]
    fn sparse_reach_over_approximates_exact_flood() {
        let mut exact_unreached_tiles = 0usize;
        let mut sparse_unreached_tiles = 0usize;
        for seed in 0..300u64 {
            let scenario = random_scenario(seed);
            let (exact, grid_w) = exact_reach(&scenario);
            let fuel = FuelMasks::build(&scenario.veg);
            let mut reach = Reach::build(&scenario.tiles, &fuel);
            for &(row, col) in &scenario.seeds {
                reach.seed_cell(row, col);
            }
            reach.run();

            let (tiles_h, tiles_w) = scenario.tiles.tiles_shape();
            for tile_row in 0..tiles_h {
                for tile_col in 0..tiles_w {
                    let slot = scenario.tiles.slot_of(tile_row, tile_col);
                    if slot == crate::tiles::UNALLOCATED {
                        continue;
                    }
                    let sparse = reach.reach_mask(tile_row, tile_col);
                    let mut exact_any = false;
                    for local_row in 0..TILE_SIZE {
                        for local_col in 0..TILE_SIZE {
                            let row = (tile_row << TILE_SHIFT) + local_row;
                            let col = (tile_col << TILE_SHIFT) + local_col;
                            let e = exact[row * grid_w + col];
                            let s = sparse[local_row] >> local_col & 1 != 0;
                            exact_any |= e;
                            assert!(
                                !e || s,
                                "seed {seed}: sparse missed exact-reachable cell ({row}, {col})"
                            );
                        }
                    }
                    if slot >= 0 {
                        exact_unreached_tiles += !exact_any as usize;
                        sparse_unreached_tiles += !reach.reached(tile_row, tile_col) as usize;
                    }
                }
            }
        }
        eprintln!("unreached live tiles: exact {exact_unreached_tiles}, sparse {sparse_unreached_tiles}");
        // sparse may freeze fewer tiles than exact, never more
        assert!(sparse_unreached_tiles <= exact_unreached_tiles);
        assert!(sparse_unreached_tiles > 0);
    }

    #[test]
    fn enclosed_island_stays_unreached() {
        // a burnt ring around an unburnt island inside one live tile,
        // seeded from outside the ring: the island must not be reached
        let veg = Grid2::filled(96, 96, 4);
        let fuel = FuelMasks::build(&veg);
        let mut tiles = TileGrid::new(96, 96);
        for tile_row in 0..3 {
            for tile_col in 0..3 {
                tiles.ensure_slot(tile_row * 32, tile_col * 32);
            }
        }
        let slot = tiles.slot_of(1, 1);
        for row in 40..56 {
            for col in 40..56 {
                let ring = row == 40 || row == 55 || col == 40 || col == 55;
                if ring {
                    tiles.tile_mut(slot).flags[local_index(row, col)] = FLAG_FIRE;
                }
            }
        }
        let mut reach = Reach::build(&tiles, &fuel);
        reach.seed_cell(5, 5);
        reach.run();
        let mask = reach.reach_mask(1, 1);
        assert!(mask[(41 - 32) as usize] >> (41 - 32) & 1 == 0, "island reached");
        assert!(mask[0] & 1 != 0, "outside of the ring not reached");
        // seeding inside the island reaches it but not the outside
        let mut inner = Reach::build(&tiles, &fuel);
        inner.seed_cell(47, 47);
        inner.run();
        assert!(inner.reached(1, 1));
        assert!(!inner.reached(0, 0));
    }
}
