# Getting Started

Follow these steps to install PROPAGATOR, launch your first simulation, and
inspect the generated outputs.

## Prerequisites
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) 0.4+ (recommended) or another virtualenv
  manager
- PROJ/GDAL libraries available on your system (needed by `rasterio`,
  `fiona`, and other I/O extras used by the CLI)

Clone the repository and switch into the project root before running the
commands below.

## Install Dependencies

Using uv (installs the core library, CLI, I/O extras, and development
tooling):

```bash
uv sync --dev --all-extras
```

Using pip (creates a virtual environment and installs the package in editable
mode with CLI + I/O extras):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[cli,io]'
```

If you only need the simulation engine (without CLI/I/O helpers) drop the
extras from the install command.

## First Simulation

The repository ships with small GeoTIFF datasets and configuration files under
`example/`. Run a deterministic simulation in GeoTIFF mode with:

```bash
uv run propagator \
  --config example/config.json \
  --mode geotiff \
  --dem example/dem.tif \
  --fuel example/fuel.tif \
  --output results/quickstart
```

The command will create the `results/quickstart` directory if it does not
already exist. Time-related settings in the configuration file (for example
`time_limit` and `time_resolution`) are expressed in seconds.

### Optional inputs
- Provide a custom fuel model with `--fuel-config example/fuel_config.yaml`.
- Switch to tiles mode by supplying `--mode tiles --tilespath <tiles_dir>
  [--tileset <name>]` when using staged tiled rasters instead of GeoTIFFs.
- Add `--verbose` to print progress tables and boundary-condition details.
- Add `--record` to save terminal logs alongside the simulation outputs.

## Inspect the Outputs

After the run completes, the output directory contains GeoTIFF rasters
(`fire_probability`, `fireline_intensity_*`, `ros_*`), GeoJSON isochrones, and
metadata JSON summarising the run. Visualise the rasters with any GIS tool or
load them back into Python using `rasterio` or `geopandas`.

## Programmatic Usage

You can embed PROPAGATOR directly in your Python workflows when you need custom
post-processing or bespoke integration logic. The snippet below mirrors
`example/example.py` and shows the essential steps:

```python
import numpy as np
from propagator.core import BoundaryConditions, FUEL_SYSTEM_LEGACY, Propagator

dem = np.zeros((2000, 2000), dtype=np.float32)
veg = np.full(dem.shape, 5, dtype=np.int32)

sim = Propagator(
    dem=dem,
    veg=veg,
    realizations=10,
    fuels=FUEL_SYSTEM_LEGACY,
    do_spotting=False,
    out_of_bounds_mode="raise",
)

ignition_mask = np.zeros_like(dem, dtype=np.uint8)
ignition_mask[dem.shape[0] // 2, dem.shape[1] // 2] = 1

sim.set_boundary_conditions(
    BoundaryConditions(
        time=0,
        ignition_mask=ignition_mask,
        wind_speed=np.ones_like(dem) * 40,
        wind_dir=np.ones_like(dem) * 90,
        moisture=np.zeros_like(dem),
    )
)

while (next_time := sim.next_time()) is not None and sim.time <= 3600:
    sim.step()
    if sim.time % 600 == 0:
        fire_prob = sim.compute_fire_probability()
        # use fire_prob in your analytics stack (save to disk, visualise, etc.)
```

Key points:
- Provide DEM/fuel rasters as NumPy arrays; no disk I/O is required unless you
  need it.
- Boundary conditions can be updated over time—compute time-dependent wind or
  moisture fields before calling `set_boundary_conditions`.
- The main loop alternates between `next_time()` to schedule time steps and
  `step()` to advance the simulation. At chosen intervals, derive statistics
  with methods like `compute_fire_probability()` or retrieve the full
  `PropagatorOutput` via `get_output()`.

For a complete runnable notebook-style walkthrough (including Matplotlib plots),
open `example/example.py`. For a production-style pipeline that loads rasters
and writes outputs via `propagator.io`, see the [Programmatic Workflow](programmatic.md)
guide.

## Validate the Environment

To confirm your setup, run the automated tests and a docs build:

```bash
uv run pytest -q
uv run ruff check src tests
uv run mkdocs build
```

All commands should finish without errors: if GDAL-related wheels fail to
install, verify that your system libraries are available to `pip`/`uv`.
