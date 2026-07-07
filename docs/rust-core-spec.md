# PROPAGATOR Core — Specification for a Rust Rewrite

Status: draft, derived from `src/propagator/core/` at commit `9792966`.

This document specifies the **core simulation engine** only (the Python
package `propagator.core`, including the numba kernels). The IO layer
(GeoTIFF/COG loaders, writers, CLI) is out of scope; the Rust core should
expose an API equivalent to the one described in §10 so any host (CLI,
FFI, service) can drive it.

---

## 1. Overview

PROPAGATOR is a **stochastic cellular-automaton wildfire spread simulator**.
It evolves a binary burn state on a regular square grid across `R`
independent realizations. Fire spreads cell-to-cell (8-connected) with a
probability that depends on fuel type, wind, slope, and moisture; the
*time* of each transition comes from a pluggable rate-of-spread (ROS)
model. Outputs are per-cell aggregates over realizations (burn
probability, arrival times, ROS, fireline intensity).

The engine is **event-driven** (discrete-event simulation), not
step-synchronous: each realization owns a binary min-heap of pending
ignition events keyed by integer arrival time (seconds). Advancing the
simulation pops events in time order until a target time.

Key architectural features that must be preserved:

- **Per-realization block-sparse state** — per-cell tracking lives in
  on-demand 32×32 tiles so memory scales with burned area, not grid size.
- **Tile freezing** — provably-inactive tiles are written to an
  append-only disk store and dropped from RAM.
- **World-anchored coordinates + domain growth** — the grid can be
  expanded (in place or via checkpoint restore) while preserving all
  state, enabling "grow the domain when the fire hits the edge".
- **Incremental checkpointing** — full dynamic state snapshots that
  reference frozen tiles by offset instead of copying them.
- **Scheduler for boundary conditions** — time-ordered queue of weather
  changes, ignitions, firefighting actions, and vegetation changes.

## 2. Conventions and units

| Quantity | Convention / unit |
|---|---|
| Grid indexing | `(row, col)`, row 0 at top (north). |
| Time | `i64` seconds from simulation start. Event times are integers; transition times are truncated to int and clamped to ≥ 1 s. |
| Cell size | meters, square cells. Default 20 m. |
| Wind direction (external API) | degrees, meteorological: clockwise, 0 = from north. |
| Wind direction (internal) | radians; converted at ingestion as `radians(deg)`. Internal angle convention: 0 = north→south propagation, π/2 = east→west (see `NEIGHBOURS_ANGLE`). |
| Wind speed | km/h externally and internally; divided by 3.6 to m/s inside models. |
| Moisture (external API) | percent (0–100). |
| Moisture (internal) | fraction (0–1); converted at ingestion (`/100`). |
| ROS | m/min internally (fuel `v0` given in m/h in config dicts, divided by 60 at fuel-system build). |
| Fireline intensity | kW/m. |
| Elevation (DEM) | meters. |
| Realizations | default `R = 100`. |

Neighbour lattice (8-connected), with per-neighbour distance
`dist = hypot(dr, dc) * cellsize` and angle
`angle = (atan2(dc, -dr) + π) mod 2π` (meteorological convention: 0 means
propagation toward south, π/2 toward west).

## 3. Domain model

### 3.1 Static inputs

- `veg: Array2<i32>` — vegetation/fuel codes. Code `0` (`NO_FUEL`) is
  reserved: not combustible, blocks spread.
- `dem: Array2<f64>` — elevation.
- `fuels: FuelSystem` — see §4.
- `cellsize: f64`, `realizations: usize`, `do_spotting: bool`.
- `origin: (i64, i64)` — world (row, col) of local cell (0, 0). Anchors
  the grid in an absolute cell coordinate system so state survives domain
  growth.
- `seed: Option<u64>` — deterministic RNG seeding (§9).
- `freeze_dir: Option<PathBuf>` — enables the frozen-tile store (§7).
- `out_of_bounds_mode: Ignore | Raise` (default Raise) — behaviour when
  fire reaches the boundary ring (§6.4).
- Pluggable model functions: `p_time_fn` (ROS/travel-time model, §5.2)
  and `p_moist_fn` (moisture probability correction, §5.3). In Rust these
  should be enum-selected or trait objects usable inside the hot kernel.

### 3.2 Dynamic state

Per simulation:

- `time: i64` — current simulation clock.
- Weather fields, full-grid `Array2<f32>`: `moisture` (fraction),
  `wind_dir` (radians), `wind_speed` (km/h). Unset until the first
  boundary conditions arrive (stepping before they are set is a usage
  error; Python would crash — Rust should return an error).
- `actions_moisture: Option<Array2<f32>>` — extra moisture from
  firefighting actions, decays over time (§6.3).
- `scheduler` — pending boundary-condition queue (§6.2).

Per realization `r`:

- **Front heap** — a binary min-heap keyed by event time, stored as five
  parallel arrays of capacity `front_capacity` (shared across
  realizations, grown by doubling; initial 4096):
  `times: i32`, `rows: i32`, `cols: i32`, `ros: f32`, `fli: f32`,
  plus `front_sizes[r]: i32`. An event means "cell (row, col) ignites at
  `time` with this ROS/FLI **unless it is already burnt when popped**"
  (lazy deletion: duplicates are allowed and skipped at pop).
- **Tiled state** (§3.3).

### 3.3 Block-sparse tiled state

Constants: `TILE_SHIFT = 5`, `TILE_SIZE = 32`, `TILE_MASK = 31`.

Per-cell tracked quantities, each stored as a pool of
`(capacity, 32, 32)` tiles per realization (pool shared-capacity across
realizations, grown by doubling, initial capacity `min(32, total_tiles)`):

- `flags: u8` — bitfield: `FLAG_FIRE = 1` (burnt), `FLAG_SPOT_GEN = 2`
  (emitted embers), `FLAG_SPOT_RECV = 4` (received an ember).
- `arrival: i32` — ignition time (s). 0 for never-burnt cells.
- `ros: f32` — ROS when ignited (m/min). 0 for ignitions/never-burnt.
- `fli: f32` — fireline intensity when ignited (kW/m).

`tile_idx: Array3<i32>` of shape `(R, tiles_h, tiles_w)` maps each
spatial tile to a pool slot, with sentinels:

- `-1` — never allocated. Cells read as zero/unburnt.
- `-2` (`FROZEN_TILE`) — frozen to disk. **Cells read as burnt**
  (`FLAG_FIRE`): the freeze criterion guarantees no cell in the tile can
  ever ignite again, so the kernel may treat the whole tile as burnt
  without reading it.
- `>= 0` — slot into the pools. `tile_counts[r]` tracks allocation;
  freed slots are compacted (§7.3). Newly allocated slots must be
  **zeroed** — the kernel assumes pristine pool memory.

`read_flags(r, row, col)` — the single primitive the kernel uses to test
cell state — implements exactly the sentinel semantics above.

### 3.4 Fold (output aggregation)

`StateFold` — per-cell aggregates across realizations, all shape
`(rows, cols)`:

| field | dtype | init | update per burnt cell |
|---|---|---|---|
| `count` | i32 | 0 | +1 if `FLAG_FIRE` |
| `spot_gen_count` | i32 | 0 | +1 if `FLAG_SPOT_GEN` (regardless of fire) |
| `spot_recv_count` | i32 | 0 | +1 if `FLAG_SPOT_RECV` |
| `arrival_min` | i32 | i32::MAX | min(arrival) over burnt |
| `arrival_sum` | f64 | 0 | += arrival over burnt |
| `ros_sum` | f64 | 0 | += ros over burnt, skipping NaN |
| `ros_max` | f32 | 0 | max over burnt, skipping NaN |
| `fli_sum` | f64 | 0 | += fli over burnt, skipping NaN |
| `fli_max` | f32 | 0 | max over burnt, skipping NaN |

Folding runs in one parallel pass over spatial tiles (each output cell
owned by one thread — race-free accumulation), then merges the frozen
contribution (§7.4).

## 4. Fuel system

A `FuelSystem` is a dense table of `n` fuel types plus an id→index map:

- Per fuel: `v0` (nominal ROS, m/min), `d0` (dead fuel load kg/m²),
  `d1` (live/canopy fuel load kg/m², default 0), `hhv` (higher heating
  value kJ/kg), `humidity` (live fuel moisture as fraction; sentinel
  `-9999` = none), `spotting: bool`, `prob_ign_by_embers: f64`,
  `burn: bool`, `name`.
- `spread_probability: Array2<f64>` — base transition probability from
  fuel i to fuel j.
- `non_vegetated` — the id of the (single) `burn=false` fuel; used as
  fallback for unknown vegetation codes.

Construction from a config map (`fuelsystem_from_dict` semantics):
`v0` is given in m/h and divided by 60; `humidity` given in percent and
divided by 100 (sentinel passed through); it is an error for a fuel to
have `d1 != 0` with no humidity.

`disable_spotting()` clears all `spotting` flags and ember probabilities;
called at construction when `do_spotting == false`.

**Fuel index grid**: `build_fuel_index_grid(fuels, veg) -> Array2<i64>`
maps every cell's vegetation code to its dense fuel index once, with
unknown codes falling back to the non-vegetated fuel (error if there is
no non-vegetated fuel and unknown codes exist). Rebuilt whenever `veg`
changes.

The legacy 7-fuel table (`FUEL_SYSTEM_LEGACY_DICT` in
`core/constants.py`) must be reproduced as the default fuel system.

## 5. Spread physics

### 5.1 Per-neighbour ignition test (`try_spread_to_neighbour`)

For a freshly burnt source cell `(row, col)` and each of the 8 neighbours:

1. Discard if out of grid, `NO_FUEL`, or already burnt (`read_flags`).
2. Compute `dh = dem[to] - dem[from]`, `dist`, `angle` (lattice
   constants × cellsize), `moist = moisture[to]`,
   `p0 = spread_probability[fuel_from, fuel_to]`.
3. Ignition probability:
   ```
   alpha_wh = max(w_h_effect_on_probability(angle, clip(w_speed,0,60), w_dir, dh, dist), 0)
   p = 1 - (1 - p0)^alpha_wh
   p = clip(p * p_moist_fn(moist), 0, 1)
   ```
4. Bernoulli draw `p > U(0,1)`; on success compute fire behaviour:
   - `(t, ros) = p_time_fn(v0[fuel_from], dh, angle, dist, moist, w_dir, w_speed)`,
     `t = max(int(t), 1)` seconds.
   - Fireline intensity from the **target** fuel:
     `lhv_dead = hhv_to*(1-moist) - Q*moist`,
     `lhv_canopy = hhv_to*(1-humidity_to) - Q*humidity_to`,
     `fli = (ros/60) * (lhv_dead*d0_to + lhv_canopy*d1_to)`, `Q = 2442.0`.
5. Push event `(time + t, row_to, col_to, ros, fli)` onto the heap.

Wind and moisture are sampled at the **source** cell for wind
(`wind_dir/wind_speed[row, col]`) and at the **target** cell for
moisture and dh.

### 5.2 ROS / travel-time models (`p_time_fn`)

Signature: `(v0, dh, angle, dist, moist, w_dir, w_speed) -> (time_s, ros_m_min)`.
`real_dist = sqrt(dist² + dh²)`; `time_s = 60 * real_dist / ros`;
moisture effect `exp(-0.014 * moist)` multiplies ROS in all models; final
ROS clipped to `[0.01, 100]` m/min.

Three implementations, selected by code (`"wang"` default, `"rothermel"`,
`"standard"`):

- **wang**: wind factor `clip(exp(0.1783 * w_spd_ms), 0.01, 10)`; slope
  factor `clip(exp(sign(dh) * 3.533 * tan(|slope_angle|)^1.2), 0.01, 10)`;
  `ros = v0 * wf * sf * moist_eff`.
- **rothermel**: slope factor `clip(exp(0.0693 * slope_deg), 0.01, 10)`;
  wind factor `clip(exp(0.0576 * flame_angle_deg)/13, 1, 20)` where
  `flame_angle = atan(0.4226 * w_spd_ms)`; `ros = v0 * sf * wf * moist_eff`.
- **standard**: `ros = v0 * w_h_effect(...) * moist_eff` (see below).

`w_spd_ms = w_speed * cos(w_dir - angle) / 3.6` (projected onto the
propagation direction; may be negative).

**`w_h_effect`** (combined wind+slope factor, used by "standard" and by
the probability modulation):
```
D1=0.5 D2=1.4 D3=8.2 D4=2.0 D5=50.0
A = 1 - D1*D2*tanh(-D4)            (evaluated at w_speed=0)
module = A + D1*D2*tanh(w_speed/D3 - D4) + w_speed/D5
a = (module - 1) / 4
w_dir_effect = (a+1)(1-a²) / (1 - a*cos(w_dir - angle))
slope = dh/dist
h_effect = 2^( tanh((3*slope)²) * sign(slope) )   # note: sign applied to tanh arg pattern below
w_h = h_effect * w_dir_effect
```
(Exact formula: `h_effect = 2 ** (tanh((slope * 3) ** 2.0 * np.sign(slope)))`.)

**`w_h_effect_on_probability`**: `wh = w_h_effect(angle, clip(w_speed,0,60), ...) - 1`;
positive values divided by 2.13, negative by 1.12; returns `wh + 1`.

### 5.3 Moisture probability models (`p_moist_fn`)

Signature `(moist_fraction) -> factor`.

- **trucchia** (default): 5th-degree polynomial in `x = moist / 0.3`
  (moisture of extinction): `-11.507x⁵ + 22.963x⁴ − 17.331x³ + 6.598x²
  − 1.7211x + 1.0003`, clipped to `[0, 1]`.
- **baghino**: `-3.5995 m³ + 5.2389 m² − 2.6355 m + 1.019` (not clipped).

### 5.4 Spotting (ember transport)

Only when `do_spotting` and the **source** fuel has `spotting = true`.
After processing neighbours of a popped cell:

1. `n_embers = min(Poisson(λ=2.0), 32)` (`MAX_SPOTTING_EMBERS = 32`,
   cap makes per-pop work bounded).
2. Per ember: angle `U(0, 2π)`; thrust `r_n ~ Normal(100, 25)` m;
   if `w_speed <= 0` distance is 0 (discarded). Distance
   `d = r_n * exp(w_spd_ms * 0.191 * (cos(w_dir - angle) - 1))`;
   landing time `d / w_spd_ms` seconds, `max(int(·), 1)`.
3. Discard embers with `d < 2*cellsize`, landing off-grid, on burnt
   cells, or on `NO_FUEL`.
4. Ignition success draw: `U(0,1) <= P_C0 * (1 + prob_ign_by_embers[target])`,
   `P_C0 = 0.6`.
5. On success: push event `(time + landing_time, row_to, col_to, ros=0, fli=0)`;
   set `FLAG_SPOT_GEN` on the source cell and `FLAG_SPOT_RECV` on the
   target cell (allocating the target tile if needed). Flags are only
   tracked when spotting tracking is on (`do_spotting`).

## 6. Simulation loop

### 6.1 Kernel: `advance_front_until(end_time)`

Runs **in parallel over realizations** (this is the hot loop). Per
realization, while the heap is non-empty and `heap_min_time <= end_time`:

1. **Headroom check (suspension protocol).** A single pop can push at
   most `FRONT_RESERVE = 8 + 32` events and allocate at most
   `TILE_RESERVE = 1 + 32` tiles (capped by tiles that remain
   unallocated). If `size + FRONT_RESERVE > front_capacity` or
   `tile_count + tile_demand > tile_capacity`, set
   `overflow[r] = 1` and **suspend with the heap intact**. The driver
   grows whichever resource is short (doubling), clears the flags and
   re-enters the kernel; realizations already suspended are skipped on
   re-entry until resumed. If nothing was growable it is a hard bug
   (panic).
2. **Boundary halt (only in Raise mode).** Peek the min event; if it
   targets the boundary ring (`row/col == 0 or == n-1`) and that cell is
   not already burnt, set `out_of_bounds[r] = 1` and suspend **without
   popping** — the caller can checkpoint, expand the domain, and resume
   without losing any spread.
3. Pop min event. If the cell is already burnt, skip (lazy deletion).
4. If the cell is on the boundary ring, set `out_of_bounds[r] = 1`
   (Ignore mode records but continues).
5. Mark the cell burnt: allocate its tile if needed; set `FLAG_FIRE`,
   `arrival = time`, `ros`, `fli`.
6. If the cell is `NO_FUEL`, stop here (it can be ignited — e.g. an
   explicit ignition — but never spreads).
7. Try the 8 neighbours (§5.1), pushing successful ignitions.
8. Spotting (§5.4) if enabled for the source fuel.

After the kernel returns, in Raise mode a nonzero `out_of_bounds` raises
`PropagatorOutOfBoundsError` — with the guarantee (thanks to step 2) that
state is intact and resumable.

### 6.2 Scheduler and boundary conditions

`BoundaryConditions` (external API, validated):

- `time: i64 >= 0` — must not be in the past (`>= current time`).
- `moisture` (% — scalar or grid), `wind_dir` (degrees), `wind_speed`
  (km/h): scalars are broadcast to the grid; converted to internal units.
- `ignitions`: either a list of `(row, col)` (applied to every
  realization) / `(row, col, realization)` tuples, or a boolean raster
  (2D → all realizations, 3D → per-realization planes). Converted to an
  `UpdateBatch` (parallel arrays rows/cols/realizations/ros=0/fli=0).
- `additional_moisture` (%): firefighting action; accumulates.
- `vegetation_changes`: grid of new vegetation codes, NaN = no change.

The `Scheduler` is a time-ordered map `time -> SchedulerEvent` where an
event carries optional weather grids, optional additional-moisture and
vegetation-change grids, and an `UpdateBatch` of ignitions. Adding an
event at an existing time **merges**: weather fields overwrite,
additional moisture adds, vegetation changes overlay (later non-NO_FUEL
values win), ignition batches concatenate.

Applying an event at its time:

1. If sim time advanced, decay action moisture (§6.3).
2. Overwrite `moisture` / `wind_dir` / `wind_speed` if present.
3. Add `additional_moisture` into `actions_moisture` (created zeroed on
   first use; clipped to `[0, 1]`).
4. Vegetation changes: **thaw all frozen tiles** (new fuel can
   invalidate the freeze criterion), apply masked overwrite of `veg`,
   rebuild the fuel index grid.
5. Ignitions: for each update, push a front event at the event time —
   after **thawing every frozen tile the new ignition could reach**
   (§7.2).

Effective moisture used by the kernel: `clip(moisture + actions_moisture, 0, 1)`.

### 6.3 Action-moisture decay

`actions_moisture *= (1 - k)^(Δt_minutes)` with `k = 0.01` per minute,
applied whenever simulation time advances past a boundary event or
propagation window.

### 6.4 Stepping API

- `step()` (legacy): advance to `min(next scheduler time, next front
  event time)`; apply the scheduler event if it is the earliest; then
  propagate up to the new time. No-op when both queues are empty.
- `step(seconds)` / `step(until=...)` (windowed; mutually exclusive
  arguments, must be ≥ 0): propagate to `time + window`, applying every
  scheduler event that falls inside the window at its exact time
  (segment-by-segment: propagate to the event time, apply the event —
  including its ignitions — and continue). Note the window semantics of
  `until` in the current code are `time + until` (it is treated
  identically to `seconds`).
- `next_time() -> Option<i64>`: min of the scheduler's next time and the
  earliest front event across realizations; `None` when fully idle.

### 6.5 Outputs

- `get_output() -> PropagatorOutput`: performs **one** fold pass and
  derives: `fire_probability = count / R` (also spot gen/recv
  probabilities), `min_arrival_time` (0 where never burnt),
  `mean_arrival_time`, `ros_mean`, `fli_mean` (sum/count where count>0,
  NaN elsewhere), `ros_max`, `fli_max`, plus `stats`.
- `PropagatorStats` from a probability map: `n_active` = number of
  realizations with a non-empty heap; `area_mean = Σp * cellsize²`;
  `area_50/75/90` = area of cells with p ≥ 0.5 / 0.75 / 0.9 (m²).
- Individual `compute_*` methods mirror the same quantities (each does
  its own fold in Python; Rust may share).
- Dense getters (analysis/tests, not hot path):
  `get_fire/get_arrival_time/get_ros/get_fireline_int/get_spotting_*`
  materialize `(R, rows, cols)` arrays from tiles, then overlay frozen
  tile contents read from the store.

## 7. Tile freezing

### 7.1 Freeze criterion (`freeze_inactive_tiles -> usize`)

Requires `freeze_dir`. A tile may be frozen (per realization) when
propagation can provably never touch it again:

- **Spotting enabled**: every in-domain cell of the tile is burnt or
  fuel-free (embers can land anywhere, so reachability arguments don't
  hold).
- **Spotting disabled** (reachability criterion): flood-fill (8-connected)
  the *unburnt-fuel* graph from the target cells of every pending front
  event of that realization; a tile is freezable iff it contains **no
  reachable unburnt fuel cell**. Rationale: fire only spreads through
  unburnt fuel, so unreachable components can never ignite.

Freezing must be **behaviour-neutral**: it changes neither dynamics nor
the RNG stream (seeded runs stay bitwise identical modulo threading, see
§9).

Freezing a tile: write its four arrays to the store keyed by
`(realization, world_row, world_col)` of the tile's top-left cell
(world-anchored so keys survive domain growth — growth shifts are
TILE_SIZE multiples); add its aggregates to the fold cache (§7.4); set
`tile_idx = FROZEN_TILE`; then compact the pool (§7.3) and shrink pool
capacity when mostly free (shrink when `2*needed <= capacity`, where
`needed = max(32, min(max_count + TILE_RESERVE, total_tiles))`).

### 7.2 Thawing

- `_thaw_tile(r, tile_row, tile_col)`: read the record, delete the key
  from the live index (records stay in the file — append-only), copy
  into a fresh pool slot, mark the fold-cache position dirty.
- Writing to a frozen tile (`_state_tile_slot` hits `FROZEN_TILE`)
  thaws it transparently.
- **New ignition** at (row, col): thaw the target tile **and every
  frozen tile whose cells are reachable** from the ignition through
  unburnt fuel (flood-fill over the burnt-state union of live + frozen
  tiles). This restores the invariant that live tiles cover everything
  fire can reach.
- `thaw_all()`: bring everything back (used on vegetation changes).
- Re-freezing later **appends a new record**; old offsets remain valid
  for checkpoints (append-only invariant).

### 7.3 Pool compaction

After freezing, per realization: remap live slots contiguously from 0
(order-preserving), **zero the vacated tail slots** (kernel invariant:
newly allocated slots are pristine), update `tile_idx` and `tile_counts`.

### 7.4 Frozen-fold cache

To keep `get_output()` free of disk I/O in steady state, the fold
contribution of frozen tiles is cached in RAM as one `StateFold` block of
shape (32, 32) per **spatial position** `(world_row, world_col)`,
aggregated across realizations:

- On freeze: accumulate the record into its position's block (unless the
  position is already dirty).
- On thaw: mark the position dirty (min/max can't be subtracted); it is
  rebuilt from the remaining frozen records on the next fold.
- On checkpoint restore: invalidate the whole cache (rebuild lazily).
- Folding: ensure the cache is valid, then merge each block into the
  global fold at `(world - origin)`, clipping to the grid (min for
  arrival_min, max for maxes, add for counts/sums).

### 7.5 Tile store (disk format)

Append-only fixed-record file `frozen_tiles.bin` in `freeze_dir`.
Record = the four tile arrays back-to-back, little-endian, row-major:
`flags` (32·32 u8) + `arrival` (i32) + `ros` (f32) + `fli` (f32) =
`RECORD_SIZE = 1024 + 4096·3 = 13312` bytes. In-memory index
`{(realization, world_row, world_col) -> offset}` of *currently frozen*
keys; `snapshot_index()` / `restore_index()` support incremental
checkpointing; `clear()` truncates the file (invalidating any checkpoint
still referencing the store).

## 8. Checkpointing and domain growth

### 8.1 World coordinates

`origin` = world (row, col) of local (0, 0). `world_bounds()` returns
inclusive world bounds. All growth operations require:

- the new grid fully contains the old one;
- the shift `(old_origin - new_origin)` is non-negative (growth only
  north/west or in place) and a **multiple of TILE_SIZE** on both axes,
  so tiles transfer without rewriting.

### 8.2 `checkpoint() -> PropagatorCheckpoint`

Immutable deep snapshot of all dynamic state: time, origin, cellsize, R,
do_spotting, veg, dem, weather + action fields (None-able), front heaps
**trimmed to `max(front_sizes)` slots**, tile pools **trimmed to
`max(tile_counts)`**, `tile_idx`, `tile_counts`, scheduler queue (deep
copy), and the frozen-tile view as `{key -> offset}` **references** into
the session store (no copying — snapshotting a run with a huge frozen
interior is cheap). RNG state is NOT captured (resumed runs are
statistically, not bitwise, equivalent; call `reseed` for determinism).

`fuels`, `p_time_fn`, `p_moist_fn` are not serialized; the caller passes
the same ones when resuming.

### 8.3 `restore(checkpoint)` (in-place rollback)

Requires identical shape and origin. Restores veg (rebuild fuel index),
time, heaps (capacity = next power-of-two-style doubling from 4096 that
fits `stored + FRONT_RESERVE + 1`), tile state, weather (copy),
scheduler queue (deep copy), and the frozen index:

- same session store → just `restore_index` (offsets still valid);
- different store / sidecar file → clear + stream-copy records in;
- no store on this propagator → materialize frozen records directly
  into pools (thaw-at-load).
- Always invalidate the fold cache.

### 8.4 `from_checkpoint(checkpoint, ...)` (resume / grow)

Builds a new propagator from a checkpoint, optionally on a larger grid
(`veg`, `dem`, `origin` all given together; growth rules of §8.1;
overlap vegetation taken from the checkpoint to preserve past
vegetation changes). Re-anchoring by `(row_shift, col_shift)`:

- front rows/cols shifted;
- `tile_idx` re-embedded at the tile offset; pools untouched;
- weather padded by **edge replication**; action moisture padded with
  zeros; (on `expand`, same rules);
- pending scheduler events re-anchored: update coords shifted, weather
  grids edge-padded, additional moisture zero-padded, vegetation changes
  NaN-padded.

Validation on load: version ≤ current (`CHECKPOINT_VERSION = 2`),
matching realizations / do_spotting / cellsize.

### 8.5 `expand(veg, dem, origin)` (cheap in-place growth)

Same growth rules, but no copying of heaps/pools: pad `veg` (overlap
keeps current values), replace dem, rebuild fuel index, shift heap
coordinates in place, re-embed `tile_idx`, pad weather/action fields,
re-anchor pending scheduler events. Frozen tiles need no work
(world-anchored keys).

### 8.6 Persistence format

`save(path)`: compressed archive (`.npz` in Python) holding scalars,
all arrays, the pickled scheduler queue, and — if frozen tiles exist —
`frozen_keys`/`frozen_offsets` arrays plus a **sidecar `path.tiles`
file** with the raw records (streamed, memory-bounded; must be kept next
to the archive). `load` verifies the sidecar exists and is at least
`len(index) * RECORD_SIZE` bytes.

The Rust rewrite may choose its own container (and should document it);
if cross-compat with existing `.npz` checkpoints is desired that is an
explicit extra requirement, not assumed here. Version field must be
honoured (reject newer, accept v1 = no frozen index).

## 9. Randomness and determinism

Random draws in the hot path: per-neighbour Bernoulli, ember count
(Poisson), ember angle (uniform), ember thrust (normal), ember ignition
(uniform).

Python/numba semantics: each worker thread has its own RNG;
`reseed(seed)` seeds thread `i` with `seed + i`. Reproducibility holds
for a fixed machine and thread count only, because realizations are
statically partitioned over threads.

**Rust requirement (improvement)**: give each *realization* its own
counter-based or per-realization-seeded RNG (e.g. seeded
`seed ⊕ realization`), making runs reproducible independently of thread
count. Statistical family: uniform f64, Poisson, Gaussian — exact
sequences need not match Python. Checkpoints do not capture RNG state
(documented behaviour); optionally the Rust version may capture it.

## 10. Public API surface (Rust sketch)

```rust
pub struct Propagator { /* opaque */ }

pub struct PropagatorConfig {
    pub veg: Array2<i32>,
    pub dem: Array2<f64>,
    pub fuels: FuelSystem,            // default: legacy 7-fuel table
    pub cellsize: f64,                // default 20.0
    pub do_spotting: bool,            // default false
    pub realizations: usize,          // default 100
    pub origin: (i64, i64),           // default (0, 0)
    pub seed: Option<u64>,
    pub freeze_dir: Option<PathBuf>,
    pub ros_model: RosModel,          // Wang | Rothermel | Standard
    pub moisture_model: MoistureModel,// Trucchia | Baghino
    pub out_of_bounds_mode: OobMode,  // Ignore | Raise (default Raise)
}

impl Propagator {
    pub fn new(cfg: PropagatorConfig) -> Result<Self>;
    pub fn reseed(&mut self, seed: u64);

    // driving
    pub fn set_boundary_conditions(&mut self, bc: BoundaryConditions) -> Result<()>;
    pub fn step_legacy(&mut self) -> Result<()>;
    pub fn step_window(&mut self, seconds: i64) -> Result<()>;
    pub fn next_time(&self) -> Option<i64>;
    pub fn time(&self) -> i64;

    // outputs
    pub fn get_output(&mut self) -> PropagatorOutput;   // &mut: fold cache
    pub fn compute_stats(&self, prob: &Array2<f32>) -> PropagatorStats;
    // dense getters: get_fire, get_arrival_time, get_ros, get_fireline_int,
    // get_spotting_generation, get_spotting_receiving

    // memory management
    pub fn freeze_inactive_tiles(&mut self) -> Result<usize>;
    pub fn thaw_all(&mut self) -> Result<usize>;

    // checkpointing / growth
    pub fn checkpoint(&self) -> PropagatorCheckpoint;
    pub fn restore(&mut self, cp: &PropagatorCheckpoint) -> Result<()>;
    pub fn from_checkpoint(cp: &PropagatorCheckpoint, opts: ResumeOptions) -> Result<Self>;
    pub fn expand(&mut self, veg: Array2<i32>, dem: Array2<f64>, origin: (i64, i64)) -> Result<()>;
    pub fn world_bounds(&self) -> (i64, i64, i64, i64);
}
```

Errors (Rust: one error enum): out-of-bounds (resumable, see §6.1),
invalid boundary conditions (past time, bad ignition format), checkpoint
mismatches (version/shape/origin/realizations/spotting/cellsize), growth
violations (containment, TILE_SIZE alignment), freeze without
`freeze_dir`, fuel-system construction errors, IO.

## 11. Performance requirements and parallelism

- Hot loop parallel **over realizations** (embarrassingly parallel:
  realizations share only read-only inputs). Rayon `par_iter_mut` over
  per-realization state is the natural mapping; the
  suspend-grow-resume protocol (§6.1) then becomes: run all
  realizations, collect overflow flags, grow shared capacities, rerun
  non-finished ones. Alternatively, per-realization `Vec`s can grow
  independently, eliminating the suspension protocol entirely — allowed,
  as long as boundary-halt semantics are preserved.
- Fold pass parallel over spatial tiles (each output cell has one owner).
- Memory scales with active front + live (unfrozen) burned area:
  front heaps ~O(front), tile pools ~O(burned tiles × 13 KiB), dense
  2D fields O(grid) each.
- Reference throughput to beat: the numba implementation (see
  `benchmarks/`); the Rust port should target ≥ parity single-threaded
  and better scaling.

## 12. Invariants (checklist for tests)

1. A cell burns at most once per realization (lazy heap deletion).
2. Event times are strictly ≥ pop time (transition times clamped ≥ 1 s);
   the heap never yields decreasing times.
3. `NO_FUEL` cells never propagate (but may be marked burnt by explicit
   ignition events).
4. Newly allocated tile slots are all-zero.
5. Frozen tiles read as fully burnt; freezing/thawing never changes
   simulation results (with fixed seeds and single thread, bitwise).
6. `checkpoint` → `restore` round-trips all dynamic state; resumed runs
   from a grown grid equal runs on the large grid from the start
   (statistically; bitwise under a shared seed discipline is a non-goal).
7. Growth preserves world-anchored positions: local + origin is
   invariant for every burnt cell, pending event, and frozen key.
8. `get_output` after freeze equals `get_output` after `thaw_all`
   (fold cache correctness).
9. Boundary halt in Raise mode loses no state: checkpoint, expand,
   resume completes the spread exactly as an unbounded run.
10. Scheduler merge semantics (§6.2) for same-time events.
11. Action moisture decays as `(0.99)^minutes` and clamps to [0, 1];
    effective moisture = `clip(moisture + actions, 0, 1)`.

## 13. Source map (Python → spec)

| Python file | Spec sections |
|---|---|
| `core/propagator.py` | §3, §6, §7, §8, §10 |
| `core/numba/front_tracking.py` | §6.1 |
| `core/numba/propagation.py` | §5.1, §5.4 |
| `core/numba/functions.py` | §5.2, §5.3 |
| `core/numba/tiles.py` | §3.3, §3.4, §7 |
| `core/numba/models.py` | §4 |
| `core/numba/rng.py` | §9 |
| `core/scheduler.py` | §6.2 |
| `core/checkpoint.py` | §8 |
| `core/tile_store.py` | §7.5 |
| `core/constants.py` | §2, §4 |
| `tests/core/` | behavioural reference for §12 |
