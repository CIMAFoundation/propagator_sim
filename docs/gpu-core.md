# GPU Core (WebGPU)

`rust/propagator-gpu` is an experimental execution backend for the Rust core
that runs the spread model on the GPU through [wgpu](https://wgpu.rs)
(Metal, Vulkan, DX12 natively; WebGPU in the browser). It reproduces the
CPU core's ensemble statistics — same expected burned area, burn-probability
maps correlated at 0.999 or better — but with a different algorithm, chosen
to suit a GPU.

## Formulation

The CPU core is a discrete-event simulation: each realization pops the
earliest ignition event from a queue, burns the cell and pushes its
neighbours. That loop is sequential by construction. The GPU core instead
treats each realization as a **shortest-path problem on a random graph**:

- The Bernoulli draw of every edge `(realization, cell, direction)` is a
  hash of those three numbers and the seed, fixed for the whole run. Each
  realization is therefore a deterministic graph with random missing edges,
  and the ignition time of a cell is the length of the shortest path to it
  from the ignitions. Statistically this is the same model as the CPU
  core, which draws once per edge when the source cell burns.
- Edge weights are the transition times of the spread model, evaluated with
  the weather of the window in which the **source** cell ignites, as the
  CPU kernel does at pop time. Nothing is precomputed: the GPU recomputes
  the eight edges of a cell when it pushes them.
- State per `(realization, cell)` is one 32-bit word, `arrival << 4 |
  code`, where `code` is the direction the fire came from (or "forced
  ignition"). Spread is `atomicMin` of `arrival + transition` into the
  neighbour's word. `pushed_at` remembers the value at which a cell was last
  pushed; a cell whose arrival is lowered later is pushed again, so the
  result does not depend on the order in which work happens.
- Work is ordered in **time buckets**. For every `(realization, 16x16
  tile)` the GPU keeps the earliest arrival not yet pushed. At a bucket
  start the tiles whose minimum falls inside the bucket are gathered into a
  worklist; each sweep processes the worklist with one workgroup per tile
  (one thread per cell) and queues the tiles that received arrivals below
  the next bucket end; the dispatch size comes from the GPU-side counter
  (indirect dispatch), so idle tiles cost nothing. Every weather window
  ends with sweeps to convergence (an empty worklist), so the fold observes
  final arrivals.
- The fold over realizations also runs on the GPU: per cell, count, first
  and mean arrival, mean and maximum ROS and fireline intensity. ROS and FLI
  are recomputed from the winning edge (the stored direction, the source's
  arrival and its window), so no per-realization payload is stored and the
  fold is race-free.

The relaxation is exact in the Dijkstra sense: the bucket width and the
number of sweeps per bucket change throughput, not results (a test asserts
bit-identical maps across bucket settings).

## Scope

Supported: contact spread with the Wang, Rothermel and Standard ROS models
and the Trucchia and Baghino moisture models; scalar weather per window
(wind direction, wind speed, fuel moisture); forced point ignitions;
boundary mode "ignore".

Not yet: ember spotting, per-cell weather fields, firefighting actions,
domain growth, checkpoints, the browser (wasm) build of the backend.

## Memory

Two 32-bit words per `(realization, cell)` — `1024² x 100` realizations is
800 MiB — plus small per-tile worklists and one 7-field fold per cell. The
state is dense; unlike the CPU core it does not scale with the burned area.
Initialisation runs on the GPU (no host upload of the state).

## Measured

`gpu_bench` runs the CPU core and the GPU core on the same scenario
(homogeneous grassland, point ignition, 30 km/h wind, hourly windows) and
compares them. On an Apple M4 Pro (14 CPU cores used by the CPU core):

| grid, realizations, hours | CPU core | GPU core | area ratio | map correlation |
|---|---|---|---|---|
| 256², 100, 6 h | 0.03 s | 0.04 s | 1.002 | 0.9994 |
| 1000², 100, 12 h | 0.09 s | 0.12 s | 1.000 | 0.9997 |
| 2000², 100, 24 h | 0.31 s | 0.45 s | 1.002 | 0.9998 |

The GPU core is correct and within a factor of 1.5 of the CPU core on this
machine, not faster: the fire front is sparse in space and time, which is
exactly what the event-driven CPU kernel exploits, while the GPU pays for
sweeps over whole tiles and for synchronisation at window ends. The backend
is expected to pay off on discrete GPUs with far more throughput than the
host CPU, and in the browser, where WebGPU offers more parallelism than
wasm threads. Both remain to be measured.

```bash
cd rust/propagator-gpu
cargo run --release --bin gpu_bench -- 1000 100 12        # grid reals hours
GPU_DEBUG=1 cargo run --release --bin gpu_bench -- 256 10 1  # per-edge model check
cargo test --release                                       # skips without a GPU
```

## Implementation notes

- `src/shader.wgsl` holds the spread models in f32 (the CPU uses f64; the
  per-edge values agree to four digits), the relaxation, the worklist
  passes and the fold.
- WGSL's `atan2` is implementation-defined for a zero argument and Metal
  returned the wrong sign for `-0.0`, swapping east and west; the eight
  neighbour angles are tabulated instead.
- `from`, `target` and `module` are reserved words in WGSL.
