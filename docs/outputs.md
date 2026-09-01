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
| `flame_length_mean` | m | Mean Byram flame length over realizations where the cell burned, derived from fireline intensity via Byram's (1959) empirical relation. Not a temperature: a proxy for fire severity/containment difficulty, historically correlated with observed flame length. |
| `flame_length_max` | m | Maximum Byram flame length across realizations. |
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

## CLI Raster Products

When running through the CLI, each reporting interval writes one GeoTIFF per
variable: `fire_probability`, `min_arrival_time`, `mean_arrival_time`,
`ros_mean`, `ros_max`, `fireline_intensity_mean`, `fireline_intensity_max`,
`flame_length_mean`, `flame_length_max`, and — when spotting is enabled in
the configuration — `spotting_generation_probability` and
`spotting_receiving_probability`.
See [CLI Usage](cli.md) for invocation details.
