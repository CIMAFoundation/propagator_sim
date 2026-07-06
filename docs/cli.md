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
| `--fuel-config PATH` | optional | YAML file defining a custom fuel system (`fuels` mapping). |
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
AWS_PROFILE=<profile> propagator \
  --config example/pedrogao/config.json \
  --fuel_config example/pedrogao/fuels_eu12.yaml \
  --mode cog \
  --cog_dem "s3://cima-propagator-return/cogs/eu/eu_dem_utm_26.tif,...,s3://cima-propagator-return/cogs/eu/eu_dem_utm_39.tif" \
  --cog_fuel "s3://cima-propagator-return/cogs/eu/eu_fuel12_utm_26.tif,...,s3://cima-propagator-return/cogs/eu/eu_fuel12_utm_37.tif" \
  --grid_dim 1024 --grow_margin 512 --seed 2017 \
  --output example/pedrogao/output --verbose
```

The loader picks the UTM-29 pair covering the ignition, reads a
1024-cell (~20 km) window around it, and grows the domain automatically
if the fire reaches the boundary. Note the bundled weather boundary
conditions are illustrative, not a validated reconstruction of the
event's extreme fire weather.

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
