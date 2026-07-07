# CLI Usage

The `propagator` command drives simulations from the terminal. It validates
input files, prepares rasters, runs the propagation loop, and writes outputs on
every reporting interval.

```bash
uv run propagator --help
```

## Basic Invocation

```bash
uv run propagator \
  --config example/config.json \
  --mode geotiff \
  --dem example/dem.tif \
  --fuel example/fuel.tif \
  --output results/run-2025-02-19
```

CLI arguments are powered by `pydantic-settings`; required inputs raise clear
validation errors before the simulation starts.

## Operating Modes

- **GeoTIFF mode** (`--mode geotiff`): supply explicit DEM (`--dem`) and fuel
  (`--fuel`) GeoTIFF rasters. Use this for bespoke datasets or the bundled
  quickstart sample.
- **Tiles mode** (`--mode tiles`, default): point to a directory of tiled DEM
  and vegetation rasters with `--tilespath` and choose a tileset via
  `--tileset`. The simulator infers the geographic window from ignition
  coordinates defined in the configuration.
- **COG mode** (`--mode cog`): stream windowed inputs from cloud-optimized
  GeoTIFFs (`s3://`, `https://` or local paths) listed per UTM zone in
  `--cog_dem` / `--cog_fuel`. Only a `--grid_dim`-cell window around the
  ignition is loaded, and when the fire reaches the window boundary the
  domain **grows automatically** by `--grow_margin` cells per side, loading
  the wider window on demand — the simulation continues without losing any
  spread. URLs are paired by their `utm_<zone>` filename hint, so the DEM
  and fuel lists may cover different zone sets. S3 access uses the standard
  AWS credential chain (`AWS_PROFILE`, environment variables, ...).

Switching between modes controls which arguments are required; passing both
`--dem` and `--fuel` automatically activates GeoTIFF mode even if `--mode` is
left at the default.

## Argument Reference

| Flag | Type / Default | Description |
| --- | --- | --- |
| `--config PATH` | required | JSON configuration file parsed into `PropagatorConfigurationLegacy`. |
| `--fuel_config PATH` | optional | YAML file defining a custom fuel system (`fuels` mapping). |
| `--mode {tiles,geotiff,cog}` | `tiles` | Select how static rasters are loaded (see above). |
| `--dem PATH` | required in geotiff mode | DEM GeoTIFF when running in geotiff mode. |
| `--fuel PATH` | required in geotiff mode | Fuel/vegetation GeoTIFF when running in geotiff mode. |
| `--tilespath PATH` | required in tiles mode | Base directory containing tiled rasters. |
| `--tileset NAME` | optional | Tileset to use within `tilespath` (defaults to `default`). |
| `--cog_dem URLS` | required in cog mode | Comma-separated DEM COG URLs, one per UTM zone. |
| `--cog_fuel URLS` | required in cog mode | Comma-separated fuel COG URLs matching the DEM zones. |
| `--grid_dim N` | `3072` | Initial window size in cells around the ignition (cog mode). |
| `--grow_margin N` | `512` | Cells added per side on automatic domain growth (cog mode); multiple of 32. |
| `--freeze_dir PATH` | optional | Freeze burned-out tiles to this directory each output interval (see [Checkpoints & Domain Growth](checkpoints.md)). |
| `--seed N` | optional | Seed the simulation RNGs for reproducible runs (fixed machine and thread count). |
| `--output PATH` | required | Destination directory; created if missing. Stores GeoTIFF, GeoJSON, and JSON outputs. |
| `--isochrones FLOAT …` | `0.5 0.75 0.9` | Probability thresholds for GeoJSON isochrone export. Repeat the flag to set multiple values. |
| `--record` | flag, default off | When enabled, saves a Rich console log in the output directory. |
| `--ignore-out-of-bounds` | flag, default off | Continue the simulation when the fire reaches the DEM boundary. |
| `--verbose` | flag, default off | Print status tables, boundary conditions, and timing information. |

Boolean switches use implicit flags: including `--verbose`, `--record`, or
`--ignore-out-of-bounds` turns each behaviour on.

## Example: Pedrogao Grande on EU-wide COGs

`example/pedrogao/` contains a demo reconstruction of the June 2017
Pedrogao Grande fire (Portugal) running on EU-wide 20 m DEM and 12-class
fuel COGs hosted on S3, with spotting enabled and hourly weather
boundary conditions:

```bash
export AWS_PROFILE=<profile>
export EU_DEM_COGS="<comma-separated DEM COG URLs>"
export EU_FUEL_COGS="<comma-separated fuel COG URLs>"

uv run propagator \
  --config example/pedrogao/config.json \
  --fuel_config example/pedrogao/fuels_eu12.yaml \
  --mode cog \
  --cog_dem "$EU_DEM_COGS" \
  --cog_fuel "$EU_FUEL_COGS" \
  --grid_dim 1024 \
  --grow_margin 512 \
  --seed 2017 \
  --output example/pedrogao/output --verbose
```

The loader picks the UTM-29 pair covering the ignition, reads a
1024-cell (~20 km) window around it, and grows the domain automatically
if the fire reaches the boundary. Note the bundled weather boundary
conditions are illustrative, not a validated reconstruction of the
event's extreme fire weather.

If a COG object is private, use an AWS profile or environment variables that
can read the object. For public COGs, set the GDAL/AWS anonymous-access
environment expected by your deployment instead. A failure such as
`Access Denied` or `not recognized as being in a supported file format` during
growth usually means GDAL could not read the remote object, not that the
simulation state itself is invalid.

## Example: Alexandroupolis / Evros on EU-wide COGs

`example/alexandroupolis/` contains an illustrative Alexandroupolis / Evros
configuration using the same EU COG set and fuel system. It is configured for a
17-day horizon, so it is a much larger run than the Pedrogao demo and should be
run with tile freezing enabled:

```bash
export AWS_PROFILE=<profile>
export EU_DEM_COGS="<comma-separated DEM COG URLs>"
export EU_FUEL_COGS="<comma-separated fuel COG URLs>"

uv run propagator \
  --config example/alexandroupolis/config.json \
  --fuel_config example/alexandroupolis/fuels_eu12.yaml \
  --mode cog \
  --cog_dem "$EU_DEM_COGS" \
  --cog_fuel "$EU_FUEL_COGS" \
  --grid_dim 2048 \
  --grow_margin 2048 \
  --freeze_dir example/alexandroupolis/freeze \
  --seed 2023 \
  --output example/alexandroupolis/output \
  --record --verbose
```

The larger initial window and growth margin reduce how often the run has to
re-open and expand remote COG windows. The tradeoff is a higher baseline memory
footprint and larger per-timestep output rasters.

## Large COG Runs and Tile Freezing

For multi-day or continental-scale COG runs, use `--freeze_dir` to page
burned-out inactive simulation tiles to disk. The CLI calls
`freeze_inactive_tiles()` after each output interval when this option is set.
Frozen tiles are restored automatically if later output statistics need them.

Practical defaults:

- Use a fast local SSD path for `--freeze_dir`; avoid network filesystems.
- Use a unique, empty freeze directory per run, for example
  `results/<run-name>-freeze`.
- Keep `--output` separate from `--freeze_dir` so final artefacts and transient
  tile pages are easy to inspect or remove independently.
- Start with `--grid_dim 2048` and `--grow_margin 1024` or `2048` for large
  fires. Larger values reduce COG growth events but increase memory and output
  raster sizes.
- Keep `time_resolution` as coarse as the analysis allows. Every output
  interval writes all configured rasters, so hourly 17-day runs can produce
  thousands of GeoTIFFs.
- Monitor both output and freeze storage:

```bash
du -sh results/<run-name> results/<run-name>-freeze
df -h .
```

Tile freezing reduces memory used by inactive burned tiles; it does not reduce
the size of GeoTIFF/JSON outputs. If a run is dominated by the active front or
by large output rasters, freezing may help less than increasing RAM, reducing
`realizations`, using a coarser `time_resolution`, or writing to a faster disk.

## Output Products

During the run, the CLI periodically writes:
- GeoTIFF rasters for fire probability, arrival time (min/mean), fireline
  intensity (mean/max), and rate of spread (mean/max). When `do_spotting` is
  enabled in the configuration, spotting generation and receiving probability
  rasters are written as well (see [Fire Spotting](spotting.md)).
- GeoJSON isochrones for configured probability thresholds.
- Metadata JSON capturing CLI arguments, execution time, and summary statistics.

Set `--record` to capture the Rich console log alongside these artefacts, which
is useful for post-run audits.

## Troubleshooting

- Missing GeoTIFFs or tiles raise validation errors before the simulation
  boots; check path spelling if you hit them.
- If dependency wheels complain about PROJ/GDAL, ensure the native libraries
  are installed (see [Getting Started](getting-started.md#prerequisites)).
- For reproducible runs across multiple ignitions or meteorological scenarios,
  adjust `realizations`, `time_limit`, and `boundary_conditions` inside the
  JSON configuration file.
