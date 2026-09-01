# Web UI

A local, interactive web interface lets you pick an area on a map, set
weather and simulation parameters, and watch the fire spread animate —
without writing a config file or GeoTIFFs by hand. It is built on top of the
same `Propagator` engine and the public-data pipeline described in
[Getting Started](getting-started.md), not a separate implementation.

It is a **single-user, unauthenticated** tool: there is no login and the
whole server runs one simulation job at a time (`JobBusyError`/429 if two
people start a run at once). It therefore binds to `127.0.0.1:8765` by
default, reachable only from the machine running it.

## Install and Run

```bash
uv sync --extra web
uv run propagator-web
```

Then open <http://127.0.0.1:8765> in a browser.

To reach it from another machine on the LAN, opt in explicitly by binding
to all interfaces (and pick a different port with `PROPAGATOR_WEB_PORT` if
needed):

```bash
PROPAGATOR_WEB_HOST=0.0.0.0 uv run propagator-web
```

then open `http://<host-ip>:8765`. Since there is no authentication,
anyone who can reach that address can start, cancel, and view runs — only
do this on a trusted network, and never expose it directly to the
internet.

The UI is available in English and Italian: it auto-detects your browser
language on first visit and can be switched anytime via the EN/IT control
in the header (the choice is remembered for future visits, via
`localStorage`).

## Using It

1. Click the map to place the simulation area's center, then click again to
   place the ignition point (the two "pick" modes switch automatically after
   the first click).
2. Adjust radius, cellsize, wind, moisture, duration, and number of
   realizations in the sidebar.
3. Click **Run simulation**. The server downloads DEM (Copernicus
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

## Points of Interest

If "Points of interest" is checked (default on), `propagator.io.osm_poi`
queries the public [Overpass API](https://overpass-api.de) for hospitals,
schools, fire/police stations, other emergency features, power
infrastructure, major roads (motorway/trunk/primary/secondary), and
buildings — using the exact same area bbox already used for the DEM/fuel
download (`radius_km` around the center), not a separate polygon.

Which of those eight categories (hospitals, fire stations, police,
schools, other emergency, major roads, buildings, power infrastructure)
are actually fetched/reported is configurable via the checkboxes under
"Points of interest" (`poi_categories` in the API, defaulting to every
category). The selection narrows the Overpass query itself, not just the
reported result, so unchecking categories makes the fetch materially
cheaper — `building` alone accounts for the large majority of elements
returned over a built-up area. Each distinct selection gets its own
on-disk cache entry (the cache is keyed by the query), so switching
selection re-queries once and is then served from cache like any other.
The overall POI count is capped by "Max POIs" (`max_pois`, default 1000,
1–5000), applied after category filtering so it only bites into the
categories actually selected.

If Overpass cannot finish the query within its own server-side budget
(too large a radius over a dense area), it answers with a `remark`
instead of results. That is reported as a non-fatal warning and the run
continues without POIs; narrowing the radius or the category selection
is what makes it succeed — retrying the identical query would just hit
the same timeout.

Power infrastructure is reported with a specific subtype category
(`power_line`, `power_tower`, `power_pole`, `power_substation`,
`power_plant`, `power_transformer`, ...) instead of a generic "power"
label, and includes `voltage`/`operator` in the tooltip when OSM has
them tagged. Lines and polygonal elements (substations, plants) keep
their full geometry — fire arrival is sampled along their entire extent,
not just a single centroid point, so a long line is reported as reached
as soon as the fire front hits it anywhere along its path. Roads are
still reduced to a single centroid point (a known limitation, not yet
addressed).

Each POI is sampled against the simulation's arrival-time grid every
frame, so on the map, markers (or, for a line/polygon POI, its actual
path) start grey (not yet reached) and turn red once the fire front
reaches any of their sampled cells, with the arrival time shown on
hover. To bound response size and keep the map responsive, at most
`max_pois` are kept per run (1000 by default), prioritizing critical
categories (hospitals, fire/police stations, substations, plants) over
generic buildings when there would be more.

Overpass responses are cached under `~/.propagator/cache/osm` (same
`PROPAGATOR_CACHE_DIR` override as the DEM/land-cover cache), keyed by the
query, so re-running the same area doesn't repeat the request. While the
fetch is in progress the status panel shows "Fetching OpenStreetMap
points of interest…", distinct from the earlier DEM/land-cover download
phase, so a slow or unreachable Overpass endpoint isn't mistaken for a
stuck DEM download. The connection uses a short (5 s) connect timeout so
an unreachable endpoint is retried and given up on quickly rather than
stalling the run for minutes; the read timeout (30 s) stays generous
enough for the query's own execution budget. If the Overpass request
ultimately fails (e.g. the public endpoint is rate-limiting or
unreachable), the run still completes using the already-downloaded
DEM/fuel data — a non-fatal warning is shown and no POIs are reported for
that run.

If the default endpoint (`overpass-api.de`) isn't reachable from your
network, set `PROPAGATOR_OVERPASS_URL` to a different mirror (e.g.
`https://z.overpass-api.de/api/interpreter`, another IP of the same
official service) — no code change needed. Not every public mirror
accepts unregistered traffic (some require prior whitelisting and
otherwise reject requests), so confirm a candidate mirror actually
answers your queries before relying on it.

## Guardrails

Request parameters are validated by `propagator.web.schemas.SimulateRequest`:

| Parameter | Range | Default |
| --- | --- | --- |
| `radius_km` | 0–50 | 10 |
| `cellsize` | 20–100 m | 30 m |
| `realizations` | 1–50 | 10 |
| `time_limit_h` | 0–48 | 12 |
| `time_resolution_h` | 0–6, ≤ `time_limit_h` | 1 |

Combinations of radius/cellsize/realizations that would produce an
unreasonably large grid for an interactive local run are rejected outright
(422 response) rather than left to run for a long time — lower the radius,
raise the cellsize, or reduce realizations if you hit this. A second check
estimates the actual memory the run would allocate (front-event heap plus
per-cell grid state, including the extra spotting arrays when spotting is
enabled) and rejects the request (422) if it exceeds a 4 GiB budget, even
if it stays under the cell-realizations cap above — lower the radius,
raise the cellsize, reduce realizations, or disable spotting if you hit
this.

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
