# Migration Guide (from 0.1.x / `main`)

This release reworks how simulation state is stored: per-cell state
moved from dense per-realization arrays into block-sparse 32×32 tiles
allocated on demand, which cut peak memory ~5× at scale (4.7 GB →
0.94 GB at 1000×1000 cells × 100 realizations) and enabled
checkpointing, rollback, domain growth and freezing burned-out interior
tiles to disk. Most of the public API is unchanged; the breaking
changes below all stem from the dense state arrays no longer existing.

## Breaking changes

### 1. Dense state fields → getter methods (and a new axis order)

The public arrays `fire`, `arrival_time`, `ros`, `fireline_int`,
`spotting_generation` and `spotting_receiving` no longer exist as
attributes. Dense views are materialized on demand by getters — and the
layout changed from `(rows, cols, realizations)` to
`(realizations, rows, cols)` so each realization's grid is contiguous:

| 0.1.x | now |
| --- | --- |
| `sim.fire[row, col, r]` | `sim.get_fire()[r, row, col]` |
| `sim.arrival_time[row, col, r]` | `sim.get_arrival_time()[r, row, col]` |
| `sim.ros[row, col, r]` | `sim.get_ros()[r, row, col]` |
| `sim.fireline_int[row, col, r]` | `sim.get_fireline_int()[r, row, col]` |
| `sim.spotting_generation[...]` | `sim.get_spotting_generation()[r, row, col]` |
| `sim.spotting_receiving[...]` | `sim.get_spotting_receiving()[r, row, col]` |

Notes:

- The getters allocate a full dense array per call — they are meant for
  analysis and tests, not for per-step hot loops. Cache the result if
  you need several reads.
- The spotting getters return zero-filled arrays (never `None`) when
  spotting is disabled; code that checked `spotting_generation is None`
  should check `sim.do_spotting` instead.
- Aggregations you computed manually from the dense arrays (e.g.
  `np.mean(sim.fire, axis=2)`) have first-class equivalents that are
  much faster: `compute_fire_probability()`, `compute_ros_max()`,
  `compute_arrival_time_min()`, … or a single `get_output()`.
- Writing into the state arrays directly is no longer possible; enqueue
  ignitions through `BoundaryConditions(ignitions=...)`.

`PropagatorOutput` and all `compute_*` methods are unchanged: 2D maps
with the same meanings and dtypes as before.

### 2. `out_of_bounds_mode="raise"` now suspends instead of burning the edge

Previously the boundary cell burned, its out-of-grid spreads were
silently dropped, and the error was raised afterwards. Now the kernel
suspends the affected realization *before* igniting the boundary-ring
cell, leaving its event heap intact, and `step()` raises
`PropagatorOutOfBoundsError`. The simulation state remains valid and
resumable: grow the domain with `expand()` /
`Propagator.from_checkpoint()` and call `step()` again, or set
`out_of_bounds_mode="ignore"` to let the fire stop at the boundary as
before.

If you catch this error today just to terminate the run, no change is
needed — but note the fire no longer burns the outermost ring in
`"raise"` mode before raising.

### 3. Removed low-level kernel API

The `UpdateBatch`-based spread path is gone: `next_updates_fn` and
`single_cell_updates` no longer exist. Propagation runs entirely inside
the `advance_front_until` front-tracking kernel, whose signature changed
(tiled state pools instead of dense arrays). If you called these
directly, drive the simulation through `Propagator.step()` /
`set_boundary_conditions()` instead.

### 4. Construct `Propagator` with keyword arguments

New optional fields (`origin`, `seed`, `freeze_dir`) were inserted
before `p_time_fn` / `p_moist_fn` in the dataclass, so positional
construction beyond `veg, dem` will silently bind the wrong parameters.
Use keywords (the documented style) for everything after `veg` and
`dem`.

## New capabilities (opt-in, no action required)

- **Checkpoints & rollback** — `checkpoint()`, `restore()`,
  `PropagatorCheckpoint.save()/load()`,
  `Propagator.from_checkpoint()`.
- **World coordinates & domain growth** — `origin=(row, col)`,
  `world_bounds()`, in-place `expand()`.
- **Deterministic seeding** — `seed=...` / `reseed()`; same seed,
  machine and numba thread count → bitwise-identical runs.
- **Tile freezing** — `freeze_dir=...` + `freeze_inactive_tiles()`
  pages burned-out interior tiles to disk with a precomputed output
  cache, keeping RAM proportional to the active front.

See [Checkpoints, Rollback & Domain Growth](checkpoints.md) for all of
these.

## Behavioural notes

- Memory is allocated on demand (front heaps and tile pools start small
  and grow), so fresh instances are cheap even on large grids;
  first-step latency can include a growth step or two.
- `get_output()` computes all maps from a single aggregation pass and
  is dramatically faster (~20 ms vs ~1.4 s at 1000×1000 × 100
  realizations).
- Kernel event throughput is at or above the 0.1.x baseline
  (~800–860k events/s vs 788–829k in the benchmark scenario).
