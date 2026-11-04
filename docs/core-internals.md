# Core Internals

This page explains how the simulation **core** works under the hood: how it
stores per-cell state, schedules the fire front, grows the domain at
runtime, and pages burned-out interior to disk. It covers both
interchangeable cores — the Python/numba core (`propagator.core`) and the
Rust core (`propagator-core`, exposed to Python as `propagator_rust` behind
the `propagator.rust_core` adapter) — and calls out where their internal
layouts diverge.

It is the implementation companion to the user-facing
[Checkpoints, Rollback & Domain Growth](checkpoints.md) page and to the
[Rust Core Specification](rust-core-spec.md); read those first for the
*what*, and this page for the *how*. For choosing between the two cores and
their observable trade-offs, see [Rust vs numba Core](rust-vs-numba.md).

## The shared model

Both cores implement the same stochastic cellular-automaton model as an
**event-driven** simulation: each of the `R` realizations owns a min-heap of
pending ignition events, and the kernel repeatedly pops the earliest event,
marks the cell burnt, and pushes spread (and ember) events for its
neighbours. Realizations are independent given the shared read-only inputs
(vegetation, terrain, weather, fuels), so both cores parallelize over them.

They agree on the observable model — outputs match to Monte-Carlo noise —
but make different internal choices for storage, front-buffer growth,
random-number generation and parallelism (compared in detail in
[Rust vs numba Core](rust-vs-numba.md)). Everything below — tiling, world
coordinates, freezing, growth rules — is shared design; the per-core notes
call out the layout differences.

## Block-sparse tiling

### Why tiles

A dense per-cell, per-realization store costs `rows × cols × R` for every
tracked quantity. At 1000×1000 cells × 100 realizations that is 100 M cells
per array — several GB across flags, arrival time, rate of spread and
fireline intensity — even though at any instant the fire only touches a thin
front and a growing burned interior.

Instead, per-cell state lives in fixed **32×32 tiles** allocated on demand
as the fire reaches them, so memory scales with *burned area* rather than
grid area. This is what cut peak memory ~5× at scale (see the
[Migration Guide](migration.md)) and what makes checkpointing, domain
growth and disk freezing tractable.

### Coordinate decomposition

The tile size is a power of two (`TILE_SIZE = 32 = 1 << TILE_SHIFT`,
`TILE_MASK = 31`), so splitting a cell coordinate into *which tile* and
*where inside it* is pure bit-twiddling, no division:

```
row  ──▶  tile_row = row >> TILE_SHIFT   (which tile block)
          local_row = row & TILE_MASK    (0..31 inside the tile)
col  ──▶  tile_col = col >> TILE_SHIFT
          local_col = col & TILE_MASK
```

A tile's top-left cell is therefore at `(tile_row << TILE_SHIFT,
tile_col << TILE_SHIFT)`. Four coordinate spaces coexist:

- **world** — absolute cell coordinates, stable across domain growth (see
  [World coordinates](#world-coordinates-and-origin)).
- **local** — cell coordinates in the current grid (`world − origin`).
- **tile** — the `(tile_row, tile_col)` block index.
- **in-tile** — the `(local_row, local_col)` offset within a 32×32 tile.

### The tile grid: an index plus a pool

Each realization has a small dense **tile-index grid** — one `i32` per
32×32 block — that maps a block to one of:

| value | meaning |
| --- | --- |
| `-1` (`UNALLOCATED`) | never touched; reads as empty |
| `-2` (`FROZEN_TILE`) | paged to disk; reads as fully **burnt** |
| `≥ 0` | slot index into the realization's tile **pool** |

The actual per-cell arrays (a `uint8` flags byte, `int32` arrival time,
`float32` rate of spread, `float32` fireline intensity) live in the pool,
one 32×32 block per array per allocated tile. The flags byte is a bitfield:

```
FLAG_FIRE      = 1   cell has burnt
FLAG_SPOT_GEN  = 2   cell emitted embers (spotting source)
FLAG_SPOT_RECV = 4   cell was ignited by an ember
```

Reading a cell is: look up its tile in the index; if `FROZEN_TILE` return
"burnt"; if `< 0` return "empty"; otherwise index the pool at
`[slot][local_row][local_col]`. The frozen sentinel reading as burnt is
exact — a tile is only frozen once every in-domain cell is provably burnt or
fuel-free (see [Freezing](#freezing-burned-out-tiles)), so the kernel never
needs the stored bytes to know the answer.

### numba layout: shared pools + capacity doubling

The numba core keeps everything in shared N-D NumPy arrays so the JIT
kernels can index them without Python objects:

```
_tile_idx      (R, tiles_h, tiles_w)          int32   the index grid
_tile_counts   (R,)                            int32   next free slot per realization
_tile_capacity  scalar                                 pool depth (shared across R)
_tile_flags    (R, capacity, 32, 32)          uint8
_tile_arrival  (R, capacity, 32, 32)          int32
_tile_ros      (R, capacity, 32, 32)          float32
_tile_fli      (R, capacity, 32, 32)          float32
```

Allocating a tile bumps `_tile_counts[r]` and stores the new slot in the
index. When a realization's count would exceed `_tile_capacity`,
`_grow_tiles()` **doubles** the pool depth (capped at `tiles_h × tiles_w`)
and copies the old contents forward. Pool depth is shared across
realizations, so it tracks the busiest realization.

### Rust layout: per-realization owned pools

The Rust core gives each realization its own `TileGrid`:

```rust
struct TileGrid {
    idx:  Vec<i32>,          // tiles_h * tiles_w, row-major index grid
    pool: Vec<Box<Tile>>,    // one heap-boxed tile per allocated block
}
struct Tile {                // struct-of-arrays, ~13 KiB
    flags:   [u8;  1024],
    arrival: [i32; 1024],
    ros:     [f32; 1024],
    fli:     [f32; 1024],
}
```

`ensure_slot()` pushes a zeroed `Box<Tile>` and records its slot; the `Vec`
grows naturally, so there is no capacity ceiling and no doubling copy —
each realization allocates exactly what it uses. `remove_slots()` compacts
the pool and remaps the index when tiles are frozen out.

### Aggregating outputs (the fold)

Outputs (`get_output()`, the dense getters) are produced by a single
**fold** pass that reduces the tiled state across realizations into 2D
per-cell maps: burn `count`, spotting counts, `arrival_min`/`arrival_sum`,
`ros_sum`/`ros_max`, `fli_sum`/`fli_max`. Dividing a `*_sum` by `count`
yields the per-cell mean over the realizations that burned it.

- numba folds in parallel **over spatial tiles** (`fold_state_tiles`, a
  `prange` kernel): each output cell is owned by exactly one tile, so the
  in-place accumulation is race-free.
- Rust folds each realization's allocated tiles into a shared `StateFold`
  (`fold_realization`), then merges the frozen-tile cache blocks.

Frozen tiles are folded from the on-disk store (numba) or the in-memory
fold cache (both), never re-simulated — see below.

## The front event heaps

Each realization's pending events are a **binary min-heap keyed by event
time**, stored struct-of-arrays (`times`, `rows`, `cols`, `ros`, `fli`
permuted together). The heap ordering is by `times` alone; the other arrays
are payload that rides along with its key. `push` sifts up, `pop_min` swaps
the root with the last leaf and sifts down — both `O(log n)`.

The cores differ in *how the buffers grow*:

### numba: preallocation with suspend-and-regrow

The numba heaps are shared preallocated arrays (`(R, capacity)`), driven
inside one JIT kernel. Because a kernel invocation cannot resize a NumPy
array mid-flight, the kernel guarantees it never overruns: before each pop
it checks there is headroom for the **worst case** that pop can generate —
`FRONT_RESERVE = 8 neighbours + 32 embers = 40` new heap entries, and
`TILE_RESERVE = 1 + 32 = 33` new tile allocations. If either would exceed
capacity it sets `overflow[r] = 1` and **suspends that realization with its
heap intact**. Control returns to the driver (`_propagate_until`), which
grows the front and/or tile buffers (`_grow_front`, `_grow_tiles`) and
re-enters the kernel to resume. This is the *suspend-and-regrow* protocol.

### Rust: grow on demand

The Rust heaps are per-realization `Vec`s (`FrontHeap`). Pushing simply
grows the `Vec`, so there is **no capacity suspension and no reserve
accounting** — the kernel runs a realization to completion in one call. The
only reason a Rust realization suspends mid-window is the boundary halt
below, which both cores share.

## Growing the domain

### World coordinates and origin

Every simulator is anchored in an absolute **world** cell coordinate
system. `origin = (row, col)` is the world position of local cell `(0, 0)`;
`world_bounds()` returns the inclusive world extent. Local and world
coordinates coincide for the default `origin = (0, 0)`. The origin is what
lets state captured on one grid be re-anchored onto another — state
transfers where the world coordinates overlap.

Frozen-tile store keys are `(realization, world_row, world_col)` of the
tile's top-left cell, so they too survive growth unchanged.

### Boundary halt

With the default `out_of_bounds_mode = "raise"` (`OobMode::Raise` in Rust),
the kernel **suspends a realization before igniting a boundary-ring cell**
— any cell with `row ≤ 0`, `col ≤ 0`, `row ≥ n_rows−1` or `col ≥ n_cols−1`
that is not already burnt. The triggering event is left at the heap root
(not popped), the realization is flagged `out_of_bounds` / `halted_on_
boundary`, and `step()` raises `PropagatorOutOfBoundsError`. Nothing is
lost: the exact spread that would have crossed the edge is still pending.

`boundary_pressure()` reports **which** edges are under pressure as
`(north, south, west, east)` by inspecting the heap-root cell of every
halted realization. A domain-growth wrapper uses this to enlarge only the
sides the fire actually reached, rather than padding all four.

### The rules

Both growth paths enforce the same constraints (`growth_shifts` in Rust,
`_growth_shifts` in numba):

- **Containment** — the new grid must fully contain the old one; the origin
  may only move north/west (toward smaller world coordinates), never
  south/east. Formally `row_shift = old_origin − new_origin ≥ 0` on both
  axes.
- **Tile alignment** — the north/west shift must be a multiple of
  `TILE_SIZE` (32 cells). This is the crucial one (see below). Growth to
  the south/east has no alignment constraint.
- **Invariant fields** — `realizations`, `cellsize` and `do_spotting` must
  match (checkpoint growth).

### Why north/west growth must be tile-aligned

Growing south/east only appends new blocks past the old extent, so the
existing tile grid keeps its indices. Growing north/west **shifts every
existing cell** to a higher local coordinate. If that shift were not a
multiple of 32, cells would move to a different in-tile offset and every
tile's contents would have to be re-binned — an expensive full rewrite.

By forcing the shift to whole tiles, the shift in *tile* space is exactly
`tile_row0 = row_shift >> TILE_SHIFT`, `tile_col0 = col_shift >>
TILE_SHIFT`. The tile **pools transfer byte-for-byte**; only the small
tile-index grid is re-anchored — the old index is copied into a fresh,
larger `-1`-filled grid at `(tile_row0, tile_col0)`. This is what makes
growth cheap regardless of how much has already burnt.

### In-place growth: `expand()`

`expand(veg, dem, origin)` grows the live simulator without copying heaps or
pools. Step by step it:

1. Validates the request and computes `(row_shift, col_shift)`.
2. Blits the current vegetation into the new grid at the shift (so past
   `vegetation_changes` in the overlap survive), replaces `dem`, rebuilds
   the fuel-index grid, and updates `origin`.
3. Shifts pending front events: `rows += row_shift`, `cols += col_shift`.
4. Re-anchors each realization's tile-index grid into the enlarged shape at
   `(tile_row0, tile_col0)`; the pools are untouched.
5. Pads the 2D fields — `moisture`/`wind_dir`/`wind_speed` by **edge
   replication** (so the boundary keeps sane weather until fresh conditions
   arrive), `actions_moisture` with **zeros**.
6. Re-anchors every queued scheduler event: shift its ignition coordinates
   and pad any attached field grids (`reanchor_event`).

### Growth through a checkpoint

`checkpoint()` + `from_checkpoint(...)` performs the same re-anchoring but
through an immutable snapshot, for when growth coincides with a save/restore
boundary (e.g. moving the run to another process). `_load_state` (Rust:
`load_state`) applies the identical shift/pad/re-anchor logic to the
checkpoint's heaps, tiles, fields and scheduler events, then re-attaches its
frozen tiles. A same-grid `restore()` is just the zero-shift special case.

## Freezing burned-out tiles

On long runs the burned interior grows without bound while the kernel only
works at the front. With a `freeze_dir`, a tile can be **frozen** — paged to
an append-only disk file and dropped from memory, its index entry set to the
`FROZEN_TILE` sentinel — once propagation can provably never touch it again:

- with spotting **disabled**: no pending front event can reach any unburnt
  fuel cell of the tile (decided by an 8-connected `flood_reachable` flood
  from the pending events through unburnt fuel);
- with spotting **enabled**: the stricter "every in-domain cell of the tile
  is burnt or fuel-free" criterion.

Because frozen tiles read as fully burnt and the criterion guarantees that
is correct, freezing changes neither the dynamics nor the RNG streams — a
seeded run is bitwise-identical with or without it. Records are fixed-size
(**~13 KB**: 1024 flags bytes + three 32×32 `int32`/`float32` arrays),
world-keyed, and never overwritten within a session, so a checkpoint can
reference them by offset for **incremental** snapshots.

Rendering outputs never re-reads frozen tiles from disk: each tile's
contribution to the fold aggregates is kept in an in-memory **fold cache**,
one 32×32 `StateFold` block per world tile position. Freezing adds a block;
thawing marks its position dirty (min/max cannot be un-merged, so it is
rebuilt lazily from the remaining records); restoring invalidates the cache.

See [Checkpoints, Rollback & Domain Growth](checkpoints.md#freezing-inactive-tiles-to-disk)
for the operational view, sidecar persistence and the floating-point caveat.

## Where to look in the source

| Concern | numba core | Rust core |
| --- | --- | --- |
| Tile constants, index/pool, fold, flood | `core/numba/tiles.py` | `propagator-core/src/tiles.rs`, `fold.rs` |
| Front heap | `core/numba/front_tracking.py` | `propagator-core/src/front.rs` |
| Spread kernel | `core/numba/front_tracking.py`, `propagation.py` | `propagator-core/src/kernel.rs` |
| Driver, growth, freezing | `core/propagator.py` | `propagator-core/src/propagator.rs` |
| Frozen-tile store | `core/tile_store.py` | `propagator-core/src/tile_store.rs` |
| Checkpoints & re-anchoring | `core/checkpoint.py` | `propagator-core/src/checkpoint.rs` |
| Python ↔ Rust bindings | — | `propagator-py/src/lib.rs`, `propagator/rust_core.py` |
