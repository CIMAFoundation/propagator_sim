# Programmatic Workflow

This guide walks through a richer example that mirrors the CLI pipeline but is
implemented entirely in Python. It combines pieces from `propagator.core` and
`propagator.io` to load rasters, configure simulations, and write artefacts on
each reporting interval.

## Scenario Overview

We will:

1. Parse a JSON configuration for ignition geometry and model options.
2. Load DEM and fuel rasters from GeoTIFF files using the high-level loader.
3. Instantiate the simulator with a custom fuel system.
4. Generate raster, GeoJSON, and metadata outputs with the writer utilities.

The example uses the datasets under `example/` so you can run it without
additional downloads.

## Complete Script

```python
from pathlib import Path

from pyproj import CRS

from propagator.core import FUEL_SYSTEM_LEGACY, Propagator
from propagator.io.configuration import PropagatorConfigurationLegacy
from propagator.io.loader.geotiff import PropagatorDataFromGeotiffs
from propagator.io.writer import (
    GeoTiffWriter,
    IsochronesGeoJSONWriter,
    MetadataJSONWriter,
    OutputWriter,
)

ROOT = Path(__file__).resolve().parent
config_path = ROOT / "config.json"
dem_path = ROOT / "dem.tif"
fuel_path = ROOT / "fuel.tif"
output_dir = ROOT / "output-programmatic"

cfg = PropagatorConfigurationLegacy.model_validate_json(config_path.read_text())

loader = PropagatorDataFromGeotiffs(
    dem_file=str(dem_path),
    veg_file=str(fuel_path),
)
dem = loader.get_dem()
veg = loader.get_veg()
geo_info = loader.get_geo_info()

sim = Propagator(
    dem=dem,
    veg=veg,
    realizations=cfg.realizations,
    fuels=FUEL_SYSTEM_LEGACY,
    do_spotting=cfg.do_spotting,
    out_of_bounds_mode="ignore",
    p_time_fn=cfg.p_time_fn,
    p_moist_fn=cfg.p_moist_fn,
)

writers = OutputWriter(
    raster_writer=GeoTiffWriter(
        start_date=cfg.init_date,
        output_folder=output_dir,
        geo_info=geo_info,
        dst_crs=CRS.from_epsg(4326),
        raster_variables_mapping={
            "fire_probability": lambda out: out.fire_probability,
            "ros_mean": lambda out: out.ros_mean,
            "ros_max": lambda out: out.ros_max,
        },
    ),
    metadata_writer=MetadataJSONWriter(
        start_date=cfg.init_date,
        output_folder=output_dir,
        prefix="metadata",
    ),
    isochrones_writer=IsochronesGeoJSONWriter(
        start_date=cfg.init_date,
        output_folder=output_dir,
        prefix="isochrones",
        thresholds=[0.3, 0.6, 0.9],
        geo_info=geo_info,
        dst_crs=CRS.from_epsg(4326),
    ),
)

non_vegetated = sim.fuels.get_non_vegetated()
for bc in cfg.get_boundary_conditions(geo_info, non_vegetated):
    sim.set_boundary_conditions(bc)

while True:
    next_time = sim.next_time()
    if next_time is None or next_time > cfg.time_limit:
        break

    sim.step()
    if sim.time % cfg.time_resolution == 0:
        output = sim.get_output()
        writers.write_output(output)

final_output = sim.get_output()
print(f"Final simulated time: {final_output.time} seconds")
```

### Key Elements

- **Configuration parsing**: `PropagatorConfigurationLegacy.model_validate_json`
  applies the same schema enforced by the CLI, ensuring you reuse existing
  validation.
- **Raster loading**: `PropagatorDataFromGeotiffs` handles `rasterio` opening and
  returns NumPy arrays along with spatial metadata in `geo_info`.
- **Output orchestration**: `OutputWriter` coordinates the specialized writers.
  You can add or remove raster variables by editing the `raster_variables_mapping`
  dictionary.
- **Simulation loop**: keep calling `next_time()` until it returns `None` or you
  exceed `time_limit`, then advance with `step()`. Every reporting interval,
  `get_output()` captures derived stats and raw fields ready for persistence.

### Going Further

- Swap `PropagatorDataFromGeotiffs` for `PropagatorDataFromTiles` when working
  with tiled rasters and dynamic midpoints.
- Load a custom fuel system using `fuels_from_yaml` or programmatically (see `cli.main`) if the
  legacy fuel system does not match your fuel types.
- Instead of the bundled writers, feed `PropagatorOutput` into your own
  analytics pipeline—store arrays in cloud buckets, stream summaries to a
  dashboard, or trigger scheduling logic for subsequent model runs.
