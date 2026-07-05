# Checkpoints, Rollback & Domain Growth

The simulator can snapshot its complete dynamic state into an immutable
`PropagatorCheckpoint`. A checkpoint supports three workflows:

- **Rollback** — restore a running simulation to an earlier point in place.
- **Persistence** — save the state to disk and resume it in another process.
- **Domain growth** — resume the state on a larger grid when the fire
  reaches a boundary, enabling effectively unbounded simulations driven by
  a wrapper that loads terrain data on demand.

A checkpoint captures the simulation clock, the per-realization front event
heaps, the block-sparse tile state (burn flags, spotting flags, arrival
time, rate of spread, fireline intensity), the weather fields, the current
vegetation (including any `vegetation_changes` applied so far), the
firefighting action moisture, and the pending boundary-condition queue.
Front heaps and tile pools are stored trimmed, so checkpoint size scales
with the active front and burned area — not with the grid.

What a checkpoint does **not** capture:

- `fuels`, `p_time_fn` and `p_moist_fn` — these contain JIT-compiled
  functions and must be passed again when rebuilding a simulator.
- The random number generator state. A run resumed from a checkpoint is
  statistically equivalent to the original but not a bitwise replay.

## Rollback

```python
sim.step(seconds=3600)
cp = sim.checkpoint()          # immutable, isolated snapshot

sim.step(seconds=7200)         # keep simulating...

sim.restore(cp)                # ...and roll back to the snapshot
sim.step(seconds=7200)         # explore an alternative evolution
sim.restore(cp)                # a checkpoint can be restored many times
```

`restore()` requires the same grid shape and origin; use
`Propagator.from_checkpoint` to move a checkpoint to a different domain.

## Saving and loading

```python
from propagator.core import Propagator, PropagatorCheckpoint

cp = sim.checkpoint()
cp.save("run-042.npz")                      # compressed npz

cp = PropagatorCheckpoint.load("run-042.npz")
sim = Propagator.from_checkpoint(cp)        # same domain, same origin
sim.step(seconds=3600)
```

Pass `fuels`, `p_time_fn` and `p_moist_fn` to `from_checkpoint` if the
original simulation used non-default ones.

## World coordinates

Every simulator is anchored in an absolute *world* cell coordinate system:
`origin=(row, col)` is the world position of local cell `(0, 0)`, and
`world_bounds()` returns the inclusive world bounds of the grid. Local and
world coordinates coincide for the default `origin=(0, 0)`.

The origin is what allows a checkpoint taken on one grid to be re-anchored
onto another: state is transferred where the world coordinates overlap.

## Growing the domain

With the default `out_of_bounds_mode="raise"`, the propagation kernel
*suspends* a realization just before it would ignite a cell on the boundary
ring, leaving its event heap intact, and `step()` raises
`PropagatorOutOfBoundsError`. Nothing is lost: checkpoint, rebuild on a
larger grid, and resume.

```python
import numpy as np
from propagator.core import Propagator, PropagatorOutOfBoundsError
from propagator.core.numba import TILE_SIZE

sim = Propagator(veg=veg, dem=dem, origin=(4096, 4096), realizations=100)
...

try:
    sim.step(seconds=3600)
except PropagatorOutOfBoundsError:
    cp = sim.checkpoint()

    # grow by one tile (32 cells) on every side; the wrapper loads the
    # larger rasters, e.g. from a cloud-optimized GeoTIFF
    margin = TILE_SIZE
    new_origin = (cp.origin[0] - margin, cp.origin[1] - margin)
    new_shape = (cp.shape[0] + 2 * margin, cp.shape[1] + 2 * margin)
    new_veg, new_dem = load_window(new_origin, new_shape)  # your loader

    sim = Propagator.from_checkpoint(
        cp, veg=new_veg, dem=new_dem, origin=new_origin
    )
    sim.step(seconds=3600)  # fire continues across the old boundary
```

Rules enforced by `from_checkpoint`:

- The new grid must fully contain the checkpointed grid (the origin can
  only move north/west, never south/east).
- North/west growth — `checkpoint.origin - origin` — must be a multiple of
  the tile size (`TILE_SIZE`, 32 cells). This lets the block-sparse tile
  pools transfer byte-for-byte; only the small tile-index grid is
  re-anchored. Growth to the south/east has no alignment constraint.
- `realizations`, `cellsize` and `do_spotting` must match the checkpoint.

During growth:

- Vegetation in the overlap region is taken from the checkpoint, so past
  `vegetation_changes` survive; outside it, the caller-provided rasters
  are used.
- Weather fields (`moisture`, `wind_dir`, `wind_speed`) are padded by edge
  replication until fresh boundary conditions are set; action moisture is
  padded with zeros; pending scheduler events are shifted and padded
  automatically.

## Checkpoint format

`save()` writes a compressed NumPy `.npz` archive with a `version` field.
All state arrays are stored natively; the pending boundary-condition queue
is embedded as a pickled blob. Checkpoints written by a newer format
version are rejected on load.
