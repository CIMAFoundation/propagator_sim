# propagator.io

The `propagator.io` package wraps raster loading routines and writer utilities
used by both the CLI and programmatic workflows. The sections below surface the
public classes and protocols that are exported via `propagator.io.__all__`.

## Public Shortcuts

::: propagator.io

## Loaders

::: propagator.io.loader.protocol

::: propagator.io.loader.geotiff

::: propagator.io.loader.tiles

## Boundary Conditions and Firefighting Actions

`TimedInput` turns time-stamped weather, ignition, and action definitions into
the core engine's `BoundaryConditions`. The Web UI and CLI share these action
classes, so intervention geometry and effects are rasterized consistently.

::: propagator.io.boundary_conditions.TimedInput

::: propagator.io.actions.Action

::: propagator.io.actions.CanadairAction

::: propagator.io.actions.HelicopterAction

::: propagator.io.actions.WaterlineAction

::: propagator.io.actions.HeavyAction

## Writers

::: propagator.io.writer.protocol

::: propagator.io.writer.raster_geotiff

::: propagator.io.writer.metadata_json

::: propagator.io.writer.isochrones_geojson
