# Rust vs numba Core

PROPAGATOR ships two interchangeable simulation cores behind the same public
interface:

- the **numba core** (`propagator.core`) — the reference implementation,
  pure Python with numba-JIT-compiled kernels; always available.
- the **Rust core** (`propagator-core`) — a native implementation exposed to
  Python as the `propagator_rust` extension and driven through the
  `propagator.rust_core` adapter; opt-in, requires building a wheel.

Both drive the identical model and produce the same maps and statistics.
This page covers how they differ and how to choose; for the mechanics they
share (tiling, front heaps, domain growth, freezing) see
[Core Internals](core-internals.md), and for the full engine specification
see the [Rust Core Specification](rust-core-spec.md).

## Same behaviour, different engine

The cores agree on the observable model. Given the same inputs they produce
statistically equivalent outputs — burn probabilities, arrival times, rate
of spread and intensity maps all match to Monte-Carlo noise. They are **not**
bitwise-identical to each other, and are not meant to be: each draws from an
independent random-number stream, so individual realizations differ while
the ensemble statistics converge. Bitwise cross-core parity is an explicit
non-goal.

Fuel definitions are supplied to both cores in the same **config units**
(rate of spread `v0` in m/h, humidity in percent); each core applies the
identical internal conversions (`÷60`, `÷100`), so a given fuel table means
the same thing to both.

| | numba core | Rust core |
| --- | --- | --- |
| Language / runtime | Python + numba JIT | native (Rust), via PyO3 |
| Availability | always installed | build the `propagator_rust` wheel |
| Per-realization state | slices of shared N-D arrays | independent owned structures |
| Front-heap growth | preallocate + *suspend-and-regrow* | grow `Vec`s on demand |
| RNG | one per **thread** | one per **realization** |
| Reproducibility | fixed machine **and** thread count | any thread count |
| Parallelism | numba `prange` / OpenMP | scoped OS threads over realizations |
| First-step latency | includes JIT warm-up | none |

## Reproducibility and threading

This is the sharpest practical difference.

The **numba core** gives each worker *thread* its own RNG (`reseed(seed)`
seeds thread `i` with `seed + i`) and partitions realizations statically
across threads. Two runs with the same seed are bitwise-identical only on the
same machine **and** with the same numba thread count (`NUMBA_NUM_THREADS`);
change the thread count and the per-realization streams shift.

The **Rust core** seeds each *realization* independently (from
`seed ⊕ realization`), so a seeded run is bitwise reproducible **regardless
of thread count** — the same 100 realizations come out identically on 1 or 32
threads. This also makes tile freezing provably behaviour-neutral: it never
perturbs the RNG stream.

Neither core captures RNG state in a checkpoint, so a run resumed from a
checkpoint is statistically — not bitwise — equivalent to the original
continuation, unless you `reseed()` explicitly. See
[Reproducibility](checkpoints.md#reproducibility).

## Performance

Both cores allocate on demand, so memory scales with burned area rather than
grid size in either case. The Rust core additionally trims peak RSS (owned
per-realization pools, no shared over-allocation) and removes JIT warm-up
from the first step.

Representative measurements on the development machine — a homogeneous
1000×1000 grassland grid, 100 realizations, 12 simulated hours, point
ignition, spotting disabled, fixed seed, via `benchmarks/compare_cores.py`
(numba JIT warm-up excluded from the timed loop):

| Mode | Core | Timed loop | Peak RSS | Output fold |
| --- | ---: | ---: | ---: | ---: |
| default threads | numba | 5.846 s | 618.8 MB | 0.534 s |
| default threads | Rust | 0.205 s | 465.3 MB | 0.027 s |
| single thread | numba | 2.154 s | 625.5 MB | 0.498 s |
| single thread | Rust | 1.485 s | 458.0 MB | 0.026 s |

In this scenario the Rust core reduced peak RSS by ~150–170 MB and ran ~1.45×
faster in the controlled single-thread run; the larger default-thread gap
also reflects thread-pool configuration differences between numba/OpenMP and
the Rust core. Treat the numbers as indicative — they are workload- and
machine-dependent — and re-run `benchmarks/compare_cores.py` for your own.

## Selecting a core

**From the CLI** — pass `--core`:

```bash
propagator run ...                 # numba (default)
propagator run ... --core rust     # Rust (requires the wheel)
```

**Programmatically** — the Rust adapter is a keyword-for-keyword drop-in for
the numba `Propagator`; switch the import:

```python
# numba core
from propagator.core import Propagator

# Rust core (identical constructor and methods)
from propagator.rust_core import Propagator
```

The adapter marshals NumPy arrays to and from the native core and re-raises
its out-of-bounds condition as the same `PropagatorOutOfBoundsError` the
numba core uses, so `expand()` / `checkpoint()` growth loops work unchanged.

## Building the Rust core

The native extension is not installed by default. Build and install the
wheel into the project virtualenv with:

```bash
./rust/build.sh
```

The script runs `maturin` against the project's `.venv` (working around the
interpreter-selection conflict when both `CONDA_PREFIX` and `VIRTUAL_ENV` are
set) and installs the resulting `propagator_rust` wheel. Until it is built,
`--core rust` and `from propagator.rust_core import Propagator` raise an
`ImportError` pointing back to this script.

## When to use which

- **numba** — the default. Zero extra setup, always present, and the
  reference for correctness. Best for interactive/exploratory work,
  development, and any environment where building a native wheel is
  inconvenient.
- **Rust** — for throughput and memory-bound runs: large grids, many
  realizations, long horizons, batch/operational pipelines, and
  thread-count-independent reproducibility. Requires building the wheel once.

Both honour the same checkpoints, domain growth and freezing semantics, so
you can develop against numba and deploy on Rust without changing your
driver code.
