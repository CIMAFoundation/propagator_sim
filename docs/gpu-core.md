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
domain growth, checkpoints.

The backend also builds for the browser: the wasm crate's `gpu` feature
exposes `GpuPropagator` (async `create`, `step`, `snapshot`) on WebGPU, and
the interactive demo runs on it with `?engine=gpu` (see below).

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
MOIST=5 WIND=45 MIXED=1 SLOPE=2 cargo run --release --bin gpu_bench -- 512 100 12
cargo test --release                                       # skips without a GPU
```

### In the browser

The demo page (`docs/demo/app.html?engine=gpu`) loads `docs/demo/pkg-gpu/`
and drives the WebGPU backend instead of the wasm core. Measured in Chrome
on the same M4 Pro, Alexandroupolis landscape (1024², 20 m), 48 steps of
30 min, spotting off, times summed over the run (kernel = stepping, fold =
outputs for the map and the area readout):

| engine | scenario | kernel | fold | total | area |
|---|---|---|---|---|---|
| WebGPU | 45 km/h, 5 %, 50 real. | 0.9 s | 1.0 s | 1.9 s | 4519 ha |
| wasm, 13 workers | same | 1.4 s | 6.4 s | 7.8 s | 4148 ha |
| wasm, 1 thread | same | 6.0 s | 25.1 s | 31.1 s | 4148 ha |
| WebGPU | 30 km/h, 0 %, 50 real. | 0.6 s | 1.0 s | 1.6 s | 4023 ha |
| wasm, 13 workers | same | 1.8 s | 11.1 s | 12.9 s | 3981 ha |
| WebGPU | 45 km/h, 5 %, 100 real. | 1.5 s | 1.5 s | 2.9 s | 4370 ha |
| wasm, 13 workers | same | 0.9 s | 7.7 s | 8.6 s | 4202 ha |

In the browser the GPU wins clearly, 4x to 8x end to end. Half of that is
the fold: the GPU folds all cells in one pass and reads 28 MB back, while
the wasm fold walks every allocated tile of every realization on the main
thread and worker pool. Kernels alone are comparable; the wasm numbers
varied by 2x between runs on this (thermally loaded) machine.

Open item: on this landscape the WebGPU burned area runs 1–9 % above the
CPU core (4519 vs 4148 ha at 50 realizations, 4370 vs 4202 at 100), where
the synthetic native scenarios agree within 2 %. Whether that is ensemble
noise or a systematic difference on real terrain and mixed fuels needs a
native run on the demo data with several seeds.

WebGPU limits matter: Chrome allows 10 storage buffers per shader stage,
so the kernel packs its state into 8 (static inputs in one buffer,
ping-pong pairs as two halves of one buffer selected by a phase flag,
worklist counters together), and a buffer bound as storage cannot also be
the indirect-dispatch source in the same dispatch, so the dispatch
arguments are copied out first. A dispatch dimension is capped at 65535
workgroups; worklists longer than that use two dimensions.

## Implementation notes

- `src/shader.wgsl` holds the spread models in f32 (the CPU uses f64; the
  per-edge values agree to four digits), the relaxation, the worklist
  passes and the fold.
- WGSL's `atan2` is implementation-defined for a zero argument and Metal
  returned the wrong sign for `-0.0`, swapping east and west; the eight
  neighbour angles are tabulated instead.
- `from`, `target` and `module` are reserved words in WGSL.
