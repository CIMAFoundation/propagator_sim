# Interactive Demo

The simulator's Rust core is compiled to **WebAssembly** and runs entirely in
your browser — no server, no upload. Everything below is the same
`propagator-core` engine used by the CLI and the Python bindings, driven
through a thin [`wasm-bindgen`](https://rustwasm.github.io/wasm-bindgen/)
layer (`propagator-wasm`).

Click or drag on the map to place ignition points, tune the weather and
simulation settings, then press **Run** to watch the stochastic fire-arrival
probability spread across a **~20 × 20 km window of the Alexandroupolis / Evros
(Greece) landscape** — real terrain and EU 12-class fuel extracted from the
`eu_fuel12` and `eu_dem` cloud-optimized GeoTIFFs. The map opens with the
[Evros 2023 reconstruction](https://github.com/CIMAFoundation/propagator_sim/blob/main/example/alexandroupolis/config.json)
ignition point and its first boundary condition (NNE wind, 30 km/h, 6 % fuel
moisture) already set.

<iframe
  src="app.html"
  title="PROPAGATOR WebAssembly demo"
  loading="lazy"
  style="width:100%; min-height:760px; border:1px solid var(--md-default-fg-color--lightest); border-radius:10px;">
</iframe>

!!! note "About this demo"
    The 1024 × 1024 landscape (20 m cells, ~20 km across) is a fixed extract.
    The run is **single-threaded**, since `wasm32` has no threads — the native
    build parallelises across realizations. The fuel model is the EU 12-class
    system (`example/pedrogao/fuels_eu12.yaml`), defined in the demo's
    JavaScript and handed to the core through a `FuelSystemBuilder`, then
    coloured with the `eu_fuel12` colormap. See
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

The landscape asset `docs/demo/alexandroupolis.bin.gz` (a gzip-compressed,
packed `veg` + `dem` window, inflated in the browser) is extracted once from
the S3 COGs with `AWS_PROFILE=return`; the extraction script lives alongside
the demo and reads `eu_fuel12_utm_35.tif` / `eu_dem_utm_35.tif` around the
ignition point.
