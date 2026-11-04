# Simulation Outputs

Every reporting interval the simulator exposes a `PropagatorOutput` snapshot
via `Propagator.get_output()`. The CLI persists these snapshots as GeoTIFF
rasters, GeoJSON isochrones, and metadata JSON; programmatic users can consume
them directly (see the [Programmatic Workflow](programmatic.md)).

All per-cell fields are 2D arrays with the same shape as the input rasters,
aggregated across the stochastic realizations of the run.

## Fields

| Field | Units | Description |
| --- | --- | --- |
| `time` | seconds | Simulation time of the snapshot, from simulation start. |
| `fire_probability` | [0, 1] | Fraction of realizations in which the cell has burned. |
| `min_arrival_time` | seconds | Earliest time the fire reached the cell across all realizations in which it burned. Cells never reached hold `0`. |
| `mean_arrival_time` | seconds | Mean arrival time over the realizations in which the cell burned. |
| `ros_mean` | m/min | Mean rate of spread over realizations where the cell burned. |
| `ros_max` | m/min | Maximum rate of spread across realizations. |
| `fli_mean` | kW/m | Mean fireline intensity over realizations where the cell burned. |
| `fli_max` | kW/m | Maximum fireline intensity across realizations. |
| `flame_height_mean` | m | Mean flame height over realizations where the cell burned (NaN elsewhere). See [Flame Height](#flame-height). |
| `flame_height_max` | m | Maximum flame height across realizations (0 where never burnt). |
| `radiant_reach_mean` | m | Mean radiant reach over realizations where the cell burned (NaN elsewhere). See [Radiant Reach](#radiant-reach). |
| `radiant_reach_max` | m | Maximum radiant reach across realizations (0 where never burnt). |
| `spotting_generation_probability` | [0, 1] | Fraction of realizations in which the cell launched at least one ember (all zeros when spotting is disabled). See [Fire Spotting](spotting.md). |
| `spotting_receiving_probability` | [0, 1] | Fraction of realizations in which the cell was hit by an ember (all zeros when spotting is disabled). |
| `stats` | — | Aggregate statistics: number of active cells and burned-area mean and 50/75/90th percentiles. |

## Arrival Times

Arrival times record, per realization, the simulation time at which each cell
ignited. They are useful to derive isochrone maps from a single realization
ensemble, or to quantify the spread of arrival uncertainty:

```python
output = sim.get_output()

# Earliest possible arrival (best-case warning time)
earliest = output.min_arrival_time

# Expected arrival where the fire is likely to reach
expected = output.mean_arrival_time
```

Both arrays are expressed in seconds from simulation start. `min_arrival_time`
is the minimum over the realizations in which the cell burned, so cells with a
low `fire_probability` can still carry a meaningful (if unlikely) early
arrival time — combine it with `fire_probability` when communicating risk.

## Flame Height

Flame height is computed in the core from each realization's fireline
intensity `I` (kW/m) with Byram's flame-length relation (Byram 1959; SI form
after Alexander 1982), taken as the height of an upright flame:

```text
H = 0.0775 * I^0.46    [m]
```

The core applies it per cell and per realization while folding the state,
so `flame_height_mean` is the mean of the per-realization flame heights,
not the flame height of `fli_mean`. Since the relation is concave, it is
never larger than the latter. Wind and slope tilt are not modelled, and
cells with zero intensity have zero flame height.

## Radiant Reach

Radiant reach is the distance from a flaming cell at which radiant heat has
fallen to a level people and structures can tolerate. It is four flame
heights, the safe-separation convention of Butler & Cohen (1998), clamped
so that a single weak or very hot cell stays plausible:

```text
R = clamp(4 * H, 10 m, 200 m)    for I > 0
R = 0                            for I <= 0 (no flame)
```

Like flame height, it is computed in the core for each realization, so
`radiant_reach_mean` is the mean of the per-realization reaches. A
realization that burned the cell with zero intensity counts as 0 in the
mean. The map gives the reach *of* each burnt cell; it is not a buffer
around the fire.

## CLI Raster Products

When running through the CLI, each reporting interval writes one GeoTIFF per
variable: `fire_probability`, `min_arrival_time`, `mean_arrival_time`,
`ros_mean`, `ros_max`, `fireline_intensity_mean`, `fireline_intensity_max`,
`flame_height_mean`, `flame_height_max`, `radiant_reach_mean`,
`radiant_reach_max`, and — when spotting is enabled in the configuration —
`spotting_generation_probability` and `spotting_receiving_probability`.
See [CLI Usage](cli.md) for invocation details.
