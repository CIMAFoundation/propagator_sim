# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- State persistence and rollback: `Propagator.checkpoint()` snapshots the
  full dynamic state (front heaps, tiled state, weather, vegetation and the
  pending boundary-condition queue) into an immutable
  `PropagatorCheckpoint`; `Propagator.restore()` rolls back in place, and
  `PropagatorCheckpoint.save()`/`load()` persist to compressed `.npz`.
  Checkpoint size scales with the burned area, not the grid.
- World cell coordinate system: `Propagator(origin=(row, col))` anchors the
  grid in absolute cell coordinates; `world_bounds()` is available on both
  `Propagator` and `PropagatorCheckpoint`.
- Domain growth: `Propagator.from_checkpoint(cp, veg=..., dem=...,
  origin=...)` resumes a checkpoint on a larger grid. North/west growth
  must be a multiple of the tile size (32 cells) so the block-sparse tiles
  transfer without being rewritten; weather fields are padded by edge
  replication and pending scheduler events are re-anchored automatically.
- Boundary suspension: with `out_of_bounds_mode="raise"` the kernel now
  suspends a realization *before* igniting a boundary-ring cell, leaving
  its event heap intact, so a checkpoint taken after
  `PropagatorOutOfBoundsError` can be resumed on a grown domain without
  losing any spread (previously the boundary cell burned and its
  out-of-grid spreads were silently dropped).

### Changed

- Peak memory reduced ~5x at scale (4.7 GB -> 0.94 GB at 1000x1000 with
  100 realizations): the front event heap starts small and grows on
  demand, and all per-cell state (arrival time, rate of spread, fireline
  intensity, burn state and spotting masks) lives in on-demand 32x32
  block-sparse tiles, so memory scales with the burned area instead of
  grid size times realizations. `get_output()` aggregates all output maps
  in a single fold pass (~20 ms vs 1.4 s).
- **Breaking:** the dense per-realization state fields (`fire`,
  `arrival_time`, `ros`, `fireline_int`, `spotting_generation`,
  `spotting_receiving`) were removed from `Propagator`; use the
  `get_fire()`, `get_arrival_time()`, `get_ros()`, `get_fireline_int()`,
  `get_spotting_generation()` and `get_spotting_receiving()` accessors
  for dense `(realizations, rows, cols)` views.
- **Breaking:** the `UpdateBatch`-based spread path (`next_updates_fn`,
  `single_cell_updates`) was removed; propagation runs entirely in the
  `advance_front_until` front-tracking kernel.
- Propagation kernel performance (~1.6x event throughput): fuel properties
  are read directly from `FuelSystem` arrays through a precomputed
  vegetation-to-fuel index grid instead of per-cell `Fuel` jitclass
  allocations and typed-Dict lookups; spread updates are pushed directly
  onto the event heap instead of building intermediate per-cell lists.
- **Breaking:** simulation state arrays now use
  `(realizations, rows, cols)` layout so each realization's grid is
  contiguous in memory. `PropagatorOutput` 2D maps are unaffected.
- **Breaking:** the lower-level kernel functions take an additional
  `fuel_idx` argument (see `propagator.core.numba.build_fuel_index_grid`).

### Fixed

- Infinite suspend/regrow loop in the propagation driver on domains
  smaller than the per-pop tile reserve (grids up to roughly 180x180
  cells): the kernel demanded more tile headroom than the domain could
  ever contain. Worst-case tile demand is now bounded by the number of
  tiles that can still be allocated.

## [0.1.0] - 2026-06-12

### Added

- Fire spotting tracking: per-realization ember generation and receiving
  states, exposed as `spotting_generation_probability` and
  `spotting_receiving_probability` output fields and CLI rasters.
- Arrival time metrics: per-cell `min_arrival_time` and `mean_arrival_time`
  output fields and CLI rasters.
- Benchmark suite with large-domain scenarios, comparison tooling, and
  profiling scripts.
- Validation for ignition formats in `BoundaryConditions`, including 3D
  ignition masks and simplified scalar declarations.
- Spotting example with output visualization
  (`example/example_spotting_dynamics.py`).
- Documentation pages for simulation outputs and the fire spotting model.

### Changed

- Propagation scheduling redesigned around a front-tracking kernel:
  propagation events are processed in batched time windows with
  Numba-parallel execution across realizations.
- Scheduler optimized with lazy bounding boxes and bisect insertion.
- `Propagator.step` advances using the configured time resolution with
  streamlined output handling.

### Fixed

- Unknown fuel IDs in the vegetation raster now fall back to the
  non-vegetated fuel instead of failing.
- Numba JIT function caching disabled to avoid
  `ReferenceError: underlying object has vanished` issues.
- Deprecation warnings from outdated imports removed.

## [0.0.2] - 2025-11-05

Enhanced simulation features and improved documentation (#15).

## [0.0.1] - 2025-11-03

First tagged release of the rewritten PROPAGATOR simulation engine
(`propagator.core`, `propagator.io`, CLI), including Shapely-based geometry
handling, 3D ignition masks, time expressed in seconds, statistics in
hectares, a core test suite, and pre-commit tooling with ruff.
