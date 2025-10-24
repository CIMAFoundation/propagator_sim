<div class="hero">
  <img class="hero__logo" src="img/propagator.png" alt="PROPAGATOR wildfire simulator logo" />
  <h1 class="hero__title">PROPAGATOR Sim</h1>
  <p class="hero__lead">
    An operational cellular-automata wildfire simulator developed by
    <a href="https://www.cimafoundation.org" target="_blank" rel="noopener">CIMA Research Foundation</a>.
    PROPAGATOR couples a Numba-powered propagation core with reusable I/O pipelines
    and a configurable CLI for deterministic or ensemble fire forecasting.
  </p>
  <div class="hero__actions">
    <a class="md-button md-button--primary" href="getting-started/">Get started</a>
    <a class="md-button" href="programmatic/">Programmatic workflow</a>
    <a class="md-button" href="reference/propagator/">API reference</a>
    <a class="md-button" href="bibliography/">Bibliography</a>
  </div>
</div>

## What's Included
- **Simulation engine**: the `propagator.core` package evolves ignition grids, applies stochastic spread models, and handles boundary conditions.
- **Data access**: helpers under `propagator.io` prepare GeoTIFFs or tiled rasters and emit GeoTIFF, GeoJSON, and JSON outputs.
- **Command line tools**: the `propagator` CLI orchestrates runs, handles configuration files, and writes time-stepped products to disk.
- **Documentation + API reference**: MkDocs pages provide operator guides, while mkdocstrings renders the public Python API.

## Typical Workflow
1. Prepare a JSON configuration describing ignition geometry, simulation horizon, and model toggles (see `example/config*.json`).
2. Supply static data—either as GeoTIFF DEM/vegetation layers or a tileset directory—and optional YAML fuel definitions.
3. Launch the run via the CLI (`uv run propagator …`), enabling verbose or recording modes for richer logging.
4. Inspect the generated rasters, isochrones, and metadata in the configured output directory.

## Quick Links
- [Getting started](getting-started.md): prerequisites, install, first simulation, and programmatic API tips
- [Programmatic Workflow](programmatic.md): end-to-end scripting with loaders and writers
- [CLI](cli.md): command options, modes, and logging
- [API](reference/index.md): Python package and Numba backend reference
