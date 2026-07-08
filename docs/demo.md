# Interactive Demo

The simulator's Rust core is compiled to **WebAssembly** and runs entirely in
your browser — no server, no upload. Everything below is the same
`propagator-core` engine used by the CLI and the Python bindings, driven
through a thin [`wasm-bindgen`](https://rustwasm.github.io/wasm-bindgen/)
layer (`propagator-wasm`).

Click or drag on the map to place ignition points, tune the weather and
simulation settings, then press **Run** to watch the stochastic fire-arrival
probability spread across a synthetic 3.6 × 3.6 km landscape.

<iframe
  src="app.html"
  title="PROPAGATOR WebAssembly demo"
  loading="lazy"
  style="width:100%; min-height:760px; border:1px solid var(--md-default-fg-color--lightest); border-radius:10px;">
</iframe>

!!! note "About this demo"
    The landscape (fuel types and terrain) is procedurally generated and fixed
    per page load, so results are reproducible. The run is **single-threaded**
    and always seeded, since `wasm32` has no threads or system entropy — the
    native build parallelises across realizations and can be seeded for
    bit-for-bit reproducibility. See
    [Rust vs numba Core](rust-vs-numba.md) and
    [Core Internals](core-internals.md) for how the engine works.

## Building the WebAssembly bundle

The bundle shipped with these docs lives in `docs/demo/pkg/`. To rebuild it
from the Rust workspace:

```bash
cd rust/propagator-wasm
wasm-pack build --release --target web --out-dir ../../docs/demo/pkg
```

This produces `propagator_wasm.js` (the ES-module glue) and
`propagator_wasm_bg.wasm` (~130 kB), which `docs/demo/app.html` loads directly.
