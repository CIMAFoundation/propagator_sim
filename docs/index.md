<div class="hero">
  <img class="hero__logo" src="img/propagator.png" alt="PROPAGATOR wildfire simulator logo" />
  <h1 class="hero__title">PROPAGATOR Sim</h1>
  <p class="hero__lead">
    An operational cellular-automata wildfire simulator developed by
    <a href="https://www.cimafoundation.org" target="_blank" rel="noopener">CIMA Research Foundation</a>.
    PROPAGATOR couples a Numba-powered propagation core with reusable I/O pipelines
    and a configurable CLI for fire forecasting.
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
- **Local web demo**: an optional debug and demonstration interface for trying scenarios, firefighting actions, and animated outputs in a browser.
- **Documentation + API reference**: MkDocs pages provide operator guides, while mkdocstrings renders the public Python API.

## Typical Workflow
Choose the workflow that fits the task:

- For repeatable or automated runs, prepare a JSON configuration, supply
  GeoTIFFs or tiled data, and run `uv run propagator …`.
- For integration in another application, use the Python API directly and
  control boundary conditions, stepping, and outputs programmatically.
- For debugging or demonstrations, the optional local web interface can set up
  a scenario on a map and visualize its time-stepped outputs.

All three workflows use the same simulation engine. The Web UI presents each
saved time step as a fire-probability heatmap, isochrones, burned-area
statistics, and an area-growth chart; the CLI writes equivalent GIS-ready
products to disk.

## Quick Links
- [Getting started](getting-started.md): prerequisites, install, first simulation, and programmatic API tips
- [Programmatic Workflow](programmatic.md): end-to-end scripting with loaders and writers
- [CLI](cli.md): command options, modes, and logging
- [Local Web Demo](web.md): optional map-based debugging and demonstration interface
- [Simulation Outputs](outputs.md): output fields, units, and aggregation across realizations
- [Fire Spotting](spotting.md): ember-transport model, configuration, and outputs
- [API](reference/index.md): Python package and Numba backend reference
