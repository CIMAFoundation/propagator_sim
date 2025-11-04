# PROPAGATOR: An Operational Cellular-Automata Wildfire Simulator

PROPAGATOR is an operational wildfire spread model developed by
[CIMA Research Foundation](https://www.cimafoundation.org). The project couples
a Numba-accelerated cellular automata core (`propagator.core`), reusable I/O
pipelines (`propagator.io`), and a configurable CLI for stochastic
fire propagation modeling. Comprehensive documentation lives under `docs/`, covering quick starts, API reference, and
programmatic guides.

## Quick Start

Clone the repository and create an environment with the CLI and I/O extras:

```bash
uv sync --dev --all-extras
```

or, using plain `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[cli,io]'
```

This installs the PROPAGATOR package in editable mode together with the optional
extras required for raster handling, the CLI, and documentation tooling.

## Running Simulations

Launch the CLI over the bundled GeoTIFF sample:

```bash
uv run propagator \
  --config example/config.json \
  --mode geotiff \
  --dem example/dem.tif \
  --fuel example/veg.tif \
  --output results/quickstart
```

See `uv run propagator --help` or `docs/cli.md` for the full argument table.

### Programmatic API

You can embed PROPAGATOR directly into Python workflows:

```python
import numpy as np
from propagator.core import BoundaryConditions, FUEL_SYSTEM_LEGACY, Propagator

dem = np.zeros((2000, 2000), dtype=np.float32)
veg = np.full_like(dem, 5, dtype=np.int32)

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
        # Persist or visualise probability grids here.
```

- `ignitions` accepts either boolean rasters or `(row, col[, realization])`
  tuples, so you can mix masks from remote sensing products with ad-hoc points.

For an end-to-end script that mirrors the CLI pipeline (including loaders and
writers), see `docs/programmatic.md` or the `example/example.py` file.

## Documentation

The MkDocs site covers:

- **Getting Started** (`docs/getting-started.md`): prerequisites, environment
  setup, quick run instructions, and programmatic usage tips.
- **CLI Usage** (`docs/cli.md`): operating modes, flag reference, output
  products, and troubleshooting.
- **Programmatic Workflow** (`docs/programmatic.md`): loader/writer pipeline
  example with `propagator.io`.
- **API Reference** (`docs/reference/`): mkdocstrings pages for the core,
  I/O, and Numba packages.
- **Bibliography** (`docs/bibliography.md`): peer-reviewed work describing
  PROPAGATOR and its operational deployments.

Serve the docs locally:

```bash
uv run mkdocs serve
```

Build the static site:

```bash
uv run mkdocs build
```

## How to Contribute

We welcome issues and pull requests! To contribute:

1. Fork the repository and create a feature branch (`git checkout -b feat/xyz`).
2. Set up the development environment with `uv sync --dev --all-extras`.
3. Make your changes, keeping module structure and style guidelines in mind.
4. Run the quality gates before submitting:
   ```bash
   uv run ruff check src tests
   uv run pytest -q
   uv run mkdocs build
   ```
5. Commit using Conventional Commit messages (e.g., `feat(core): add wind bias`).
6. Open a pull request describing the change, verification steps, and any
   relevant screenshots or artefacts.

For major features or architectural changes, please open an issue first to
discuss the proposal. Contributors are encouraged to reference the documentation
pages above when adding or modifying features.
