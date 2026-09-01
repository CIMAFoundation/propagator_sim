# Local Web Demo

A local, interactive debug and demonstration interface lets you pick an area
on a map, set weather and simulation parameters, and watch the fire spread
animate — without writing a config file or GeoTIFFs by hand. It is built on
top of the same `Propagator` engine and the public-data pipeline described in
[Getting Started](getting-started.md), not a separate implementation.

It is an **optional, local-only debug/demo tool**, not the primary operational
interface. The server binds to `127.0.0.1` and is meant to run on your own
machine, not to be exposed on a network.

## Install and Run

```bash
uv sync --extra web
uv run propagator-web
```

Then open <http://127.0.0.1:8765> in a browser.

The server is deliberately local and binds only to `127.0.0.1`. The demo is
not an internet-hosted service and should not be exposed as an operational
multi-user server.

## In-app Manual

Select **Manual** in the app header to open the built-in guide. It
explains wildfire behaviour, every input and output, the simulation algorithm,
and the limits of interpreting model results in plain English. The guide is
served locally at <http://127.0.0.1:8765/manual.html> while the app is running.

## Using It

1. Click the map to place the simulation area's center, then click again to
   place the ignition point (the two "pick" modes switch automatically after
   the first click).
2. Adjust radius, cell size, wind, moisture, duration, output frequency, and
   number of realizations in the sidebar. Enable spotting if the scenario
   should include wind-carried embers.
3. Optionally add one or more firefighting actions. Choose the action and its
   scheduled time, select **Draw line**, click points on the map, and select
   **Finish line**. The queued actions appear below the controls and can be
   removed before starting the run.
4. Click **Run simulation**. The server downloads DEM (Copernicus
   GLO-30) and land-cover (ESA WorldCover 10 m) tiles for the area, builds
   the fuel grid, and runs the simulation, reporting progress as it goes.
5. Once done, scrub the time slider to see the fire-probability heatmap and
   isochrone lines for each hour, alongside area/active-realization stats
   and a growth chart.

If the ignition point falls on a non-burnable fuel class (fuel code 3,
non-vegetated — typically urban areas), a warning is shown but the run still
completes (it will simply show no spread). If the ignition point falls
outside the selected area, the run fails immediately with a clear error.

## Firefighting Actions

Actions are scheduled relative to the start of the simulation and applied
when that simulation time is reached. Each drawn WGS84 line is rasterized on
the scenario grid through the same `TimedInput` boundary-condition machinery
used by the CLI.

| Action | Simulated effect |
| --- | --- |
| Canadair | Adds 25 percentage points of fuel moisture on the line and 22 points in its one-cell buffer. |
| Helicopter | Adds 22 percentage points at deterministic scattered drop points near the line and 20 points in their one-cell buffer. |
| Waterline | Adds 27 percentage points of fuel moisture across the line and its one-cell buffer. |
| Heavy vehicles | Replaces fuel across the line and its one-cell buffer with the non-vegetated fuel class, creating a persistent firebreak. |

Moisture effects are added only inside the affected cells, stack on top of
the scenario's existing moisture, and then decay according to the simulation's
moisture-relief model. Heavy-vehicle actions change the fuel rather than its
moisture.

Actions are scenario inputs, not live controls: add or remove them before
selecting **Run simulation**. To compare intervention strategies, run the
same scenario again with a different action type, line, or scheduled time.

## Reading the Results

- The heatmap shows the fraction of realizations in which each cell burned.
- Isochrones outline the selected probability thresholds at the displayed
  time; the defaults are 50%, 75%, and 90%.
- Expected burned area sums each cell's probability multiplied by its area.
- Threshold areas report how much land reached at least 50%, 75%, or 90%
  probability.
- Active realizations count the simulations whose fire front is still moving.
- The growth chart compares expected area with the area above the 50%
  threshold through time.

These products describe a stochastic scenario, not a precise fire perimeter.
Use them to compare likely trends and intervention assumptions, never as the
sole basis for operational decisions.

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
- Firefighting actions are converted to `CanadairAction`, `HelicopterAction`,
  `WaterlineAction`, or `HeavyAction` objects and scheduled as future
  `BoundaryConditions`. This keeps Web UI and CLI action semantics aligned.

## Guardrails

Request parameters are validated by `propagator.web.schemas.SimulateRequest`:

| Parameter | Range | Default |
| --- | --- | --- |
| `radius_km` | >0–50 | 15 |
| `cellsize` | 20–100 m | 30 m |
| `realizations` | 1–50 | 10 |
| `time_limit_h` | >0–48 | 6 |
| `time_resolution_h` | >0–6, ≤ `time_limit_h` | 1 |
| action time | 0–`time_limit_h` | 1 h in the UI |

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
