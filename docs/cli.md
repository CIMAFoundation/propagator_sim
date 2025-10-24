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

Switching between modes controls which arguments are required; passing both
`--dem` and `--fuel` automatically activates GeoTIFF mode even if `--mode` is
left at the default.

## Argument Reference

| Flag | Type / Default | Description |
| --- | --- | --- |
| `--config PATH` | required | JSON configuration file parsed into `PropagatorConfigurationLegacy`. |
| `--fuel-config PATH` | optional | YAML file defining a custom fuel system (`fuels` mapping). |
| `--mode {tiles,geotiff}` | `tiles` | Select how static rasters are loaded (see above). |
| `--dem PATH` | required in geotiff mode | DEM GeoTIFF when running in geotiff mode. |
| `--fuel PATH` | required in geotiff mode | Fuel/vegetation GeoTIFF when running in geotiff mode. |
| `--tilespath PATH` | required in tiles mode | Base directory containing tiled rasters. |
| `--tileset NAME` | optional | Tileset to use within `tilespath` (defaults to `default`). |
| `--output PATH` | required | Destination directory; created if missing. Stores GeoTIFF, GeoJSON, and JSON outputs. |
| `--isochrones FLOAT …` | `0.5 0.75 0.9` | Probability thresholds for GeoJSON isochrone export. Repeat the flag to set multiple values. |
| `--record` | flag, default off | When enabled, saves a Rich console log in the output directory. |
| `--ignore-out-of-bounds` | flag, default off | Continue the simulation when the fire reaches the DEM boundary. |
| `--verbose` | flag, default off | Print status tables, boundary conditions, and timing information. |

Boolean switches use implicit flags: including `--verbose`, `--record`, or
`--ignore-out-of-bounds` turns each behaviour on.

## Output Products

During the run, the CLI periodically writes:
- GeoTIFF rasters for fire probability, fireline intensity (mean/max), and rate
  of spread (mean/max).
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
