# Web UI

A local, interactive web interface lets you pick an area on a map, set
weather and simulation parameters, and watch the fire spread animate —
without writing a config file or GeoTIFFs by hand. It is built on top of the
same `Propagator` engine and the public-data pipeline described in
[Getting Started](getting-started.md), not a separate implementation.

It is a **local-only** tool: the server binds to `127.0.0.1` and is meant to
run on your own machine, not to be exposed on a network.

## Install and Run

```bash
uv sync --extra web
uv run propagator-web
```

Then open <http://127.0.0.1:8765> in a browser.

## Using It

1. Click the map to place the simulation area's center, then click again to
   place the ignition point (the two "pick" modes switch automatically after
   the first click).
2. Adjust radius, cellsize, wind, moisture, duration, and number of
   realizations in the sidebar.
3. Click **Avvia simulazione**. The server downloads DEM (Copernicus
   GLO-30) and land-cover (ESA WorldCover 10 m) tiles for the area, builds
   the fuel grid, and runs the simulation, reporting progress as it goes.
4. Once done, scrub the time slider to see the fire-probability heatmap and
   isochrone lines for each hour, alongside area/active-realization stats
   and a growth chart.

If the ignition point falls on a non-burnable fuel class (fuel code 3,
non-vegetated — typically urban areas), a warning is shown but the run still
completes (it will simply show no spread). If the ignition point falls
outside the selected area, the run fails immediately with a clear error.

## How It Works

- `propagator.io.data_prep` (shared with `example/italy/prepare_area_data.py`)
  downloads and reprojects DEM/land-cover tiles into an aligned grid, then
  maps ESA WorldCover classes onto PROPAGATOR's legacy fuel codes. See that
  module's docstring for the limitations of this default mapping.
- `propagator.web.runner` builds a `Propagator` from that grid and steps it
  one `time_resolution_h` at a time, capturing a `fire_probability` frame
  after each step. Every frame is built on the exact same grid, so its map
  bounds never change while scrubbing the time slider.
- `propagator.web.jobs.JobManager` runs one simulation at a time in a
  background thread and keeps its state in memory — there's no database and
  state doesn't survive a server restart.
- Isochrone lines reuse `propagator.io.writer.isochrones_geojson.extract_isochrone`
  (the same code the CLI uses to write `isochrones_*.json`), reprojected to
  WGS84 for the map.

## Guardrails

Request parameters are validated by `propagator.web.schemas.SimulateRequest`:

| Parameter | Range | Default |
| --- | --- | --- |
| `radius_km` | 0–50 | 15 |
| `cellsize` | 20–100 m | 30 m |
| `realizations` | 1–50 | 10 |
| `time_limit_h` | 0–48 | 6 |
| `time_resolution_h` | 0–6, ≤ `time_limit_h` | 1 |

Combinations of radius/cellsize/realizations that would produce an
unreasonably large grid for an interactive local run are rejected outright
(422 response) rather than left to run for a long time — lower the radius,
raise the cellsize, or reduce realizations if you hit this.

Only one simulation runs at a time; starting a new one while another is
still running/preparing data returns a 429 response.

## Downloaded Data Cache

Downloaded DEM/land-cover tiles are cached under
`~/.propagator/cache` (override with the `PROPAGATOR_CACHE_DIR` environment
variable) so repeated runs over the same area don't re-download tiles.

## Pre-caching All of Italy

The first simulation over a given area downloads its DEM/land-cover tiles
on the fly, which adds a network wait during the "preparing_data" phase.
To skip that wait entirely for any point in Italy, pre-download every tile
covering the country once:

```bash
uv run python example/italy/download_national_data.py
```

This is a one-off operation, roughly **5–7 GB** total (mostly WorldCover
land-cover tiles), so run it once on a good connection — bandwidth is the
only limit. It's safe to interrupt and re-run: tiles already downloaded are
skipped, so it resumes where it left off. It writes into the same cache
directory the web app already uses (`~/.propagator/cache` by default, or
`PROPAGATOR_CACHE_DIR`), so no further configuration is needed — the next
time you run a simulation anywhere in Italy, `preparing_data` skips
straight to building the grid, with no download step.

To use a different disk/location for the cache:

```bash
uv run python example/italy/download_national_data.py --cache-dir /path/to/big/disk
```

This only pre-fetches raw source tiles; it does not build or store a
national mosaic — `prepare_area_data` still builds each simulation's small
local grid on demand from the cached tiles, which is fast (a few seconds)
once there's no download involved.
