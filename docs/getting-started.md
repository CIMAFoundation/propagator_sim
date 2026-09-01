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

To use the interactive browser interface, install the `web` extra instead:

```bash
pip install -e '.[web]'
```

`uv sync --dev --all-extras` already includes it.

## Optional Local Web Demo

For debugging and demonstrations, run the local web server and open the
address it prints:

```bash
uv run propagator-web
```

The demo is available at <http://127.0.0.1:8765>. Pick the simulation center
and ignition on the map, configure the weather and duration, and select
**Run simulation**. You can also draw intervention lines and schedule
Canadair, helicopter, waterline, or heavy-vehicle actions before starting.

The first run in an area downloads terrain and land-cover data. See the
[Local Web Demo](web.md) guide for the complete workflow, result
interpretation, cache details, and the link to the built-in plain-English
manual.

## First CLI Simulation

The repository ships with small GeoTIFF datasets and configuration files under
`example/`. Run a simulation in GeoTIFF mode with:

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

The configuration must include boundary conditions with an entry at time `0`
and at least one ignition in that initial state. If ignitions are provided at
the top level, the CLI merges them into the first boundary-condition block.

### Optional inputs
- Provide a custom fuel model with `--fuel-config example/fuel_config.yaml`.
- Switch to tiles mode by supplying `--mode tiles --tilespath <tiles_dir>
  [--tileset <name>]` when using staged tiled rasters instead of GeoTIFFs.
- Add `--verbose` to print progress tables and boundary-condition details.
- Add `--record` to save terminal logs alongside the simulation outputs.
- Add `--ignore-out-of-bounds` to keep running after the fire reaches the
  domain edge.

## Inspect the Outputs

After the run completes, the output directory contains GeoTIFF rasters
(`fire_probability`, `fireline_intensity_*`, `ros_*`), GeoJSON isochrones, and
metadata JSON summarising the run. Visualise the rasters with any GIS tool or
load them back into Python using `rasterio` or `geopandas`.

Each artefact name includes the simulation time, for example
`fire_probability_3600.tiff` or `metadata_3600.json`. When `--record` is
enabled, the terminal export is written as `run.html` and `run.log` in the
output directory.

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

ignitions = [(dem.shape[0] // 2, dem.shape[1] // 2)]

sim.set_boundary_conditions(
    BoundaryConditions(
        time=0,
        ignitions=ignitions,
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
- `ignitions` accepts boolean rasters or `(row, col[, realization])` tuples.
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
