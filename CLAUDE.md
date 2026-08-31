# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PROPAGATOR is an operational wildfire spread simulator built by CIMA Research
Foundation. It models fire propagation as a stochastic cellular automaton over
a DEM/vegetation grid, driven by wind, slope, moisture, and fuel-type
probabilities, with an optional ember-spotting model.

## Commands

```bash
uv sync --dev --all-extras          # create venv with CLI/IO extras + dev tooling
uv run propagator --help            # CLI usage
uv run propagator --config example/config.json --mode geotiff \
  --dem example/dem.tif --fuel example/fuel.tif --output results/quickstart

uv run pytest -q                    # run tests
uv run pytest tests/core/test_scheduler.py -q      # single file
uv run pytest -k test_name -q                      # single test by pattern
uv run pytest --maxfail=1 -q

uv run ruff check src tests         # lint (79-char lines; only add --fix after reviewing diffs)
uv run ruff format src tests        # format

uv run mkdocs serve                 # docs with live reload
uv run mkdocs build                 # static docs site
```

Tests are configured in `pyproject.toml` ([tool.pytest.ini_options]) with
`pythonpath = ["src"]`, so `propagator` imports directly without an editable
install for test runs.

## Architecture

Three packages under `src/propagator/`:

- **`propagator.core`** — the simulation engine.
  - `propagator.py` — the `Propagator` dataclass: the main simulation object.
    Owns grid state (`fire`, `arrival_time`, `ros`, `fireline_int`,
    `moisture`, spotting arrays) and a binary-heap-based fire "front" per
    realization (`_front_times/_rows/_cols/_ros/_fli`, pushed/popped via
    `_front_push`) used to drive propagation. `step()` advances either to the
    next scheduled event (`_step_legacy`) or across a fixed time window
    (`_step_window`/`until=`). Actual cell-to-cell spread math happens inside
    the Numba-jitted `advance_front_until` (imported from `core.numba`),
    called from `_propagate_until`.
  - `scheduler.py` — `Scheduler`/`SchedulerEvent`: a time-ordered event queue
    (custom `SortedDict` using `bisect.insort`) for boundary-condition changes
    (wind, moisture, vegetation edits, firefighting actions) and ignitions,
    decoupled from the fire-front heap. `push_updates`/`pop`/`next_time` are
    hot-path methods — see inline comments before changing their complexity.
  - `models.py` — shared dataclasses: `BoundaryConditions`, `UpdateBatch`,
    `UpdateBatchWithTime`, `PropagatorOutput`, `PropagatorStats`.
  - `constants.py` — physical/model constants (cell size, defaults, fuel
    codes).
  - `core/numba/` — Numba-jitted kernels: `propagation.py`
    (`advance_front_until`, the actual CA update loop),
    `front_tracking.py`, `functions.py` (spread-probability / travel-time /
    moisture functions, `get_p_time_fn`, `get_p_moisture_fn`), `models.py`
    (`FuelSystem`, `FUEL_SYSTEM_LEGACY`, `fuelsystem_from_dict`). Because
    these are `@njit`-compiled, function signatures must stay Numba-typeable
    (no arbitrary Python objects across the jit boundary) — see the
    `p_time_fn`/`p_moist_fn` docstrings in `propagator.py` for the expected
    signatures.

- **`propagator.io`** — I/O layer, isolated from the core engine so the
  simulator can be driven with plain NumPy arrays independent of any file
  format.
  - `loader/` — `PropagatorDataFromGeotiffs` (single GeoTIFF DEM/fuel pair)
    and `PropagatorDataFromTiles` (tiled rasters with dynamic midpoints),
    both implementing `loader/protocol.py`.
  - `writer/` — `GeoTiffWriter`, `IsochronesGeoJSONWriter`,
    `MetadataJSONWriter`, orchestrated together by `OutputWriter`
    (`writer/__init__.py`) which fans a `PropagatorOutput` snapshot out to
    each configured writer per reporting interval.
  - `configuration.py` — `PropagatorConfigurationLegacy` (Pydantic model)
    parses the JSON config format used by both the CLI and programmatic
    scripts, and builds `BoundaryConditions` from it.
  - `actions.py`, `boundary_conditions.py`, `geo.py`, `geometry.py` — action
    (firefighting) parsing, boundary-condition assembly, and geospatial
    helpers (CRS reprojection, geometry-to-grid rasterization).

- **`propagator.cli`** — `main.py` backs the `propagator` console script
  (`[project.scripts]` in `pyproject.toml`); wires config parsing, loaders,
  the `Propagator` simulation loop, and writers together. `console.py`/
  `logging_config.py` handle Rich-based terminal output and logging setup.

### Simulation loop shape

The pattern used throughout examples and the CLI:

```python
sim.set_boundary_conditions(bc)   # schedule ignitions/wind/moisture changes
while (t := sim.next_time()) is not None and t <= time_limit:
    sim.step()
    if sim.time % time_resolution == 0:
        output = sim.get_output()   # PropagatorOutput: probabilities, ROS, stats
```

`next_time()` returns the minimum of the next scheduled boundary-condition
event and the next fire-front pop time; `None` means the simulation is
finished. Per-cell outputs (`compute_fire_probability`,
`compute_ros_mean/max`, `compute_arrival_time_min/mean`,
`compute_fireline_int_mean/max`) aggregate across `realizations` (the third
array axis) and are bundled into `PropagatorOutput` via `get_output()`.

## Notes

- Wind direction inputs are degrees clockwise from north; internally
  converted to radians counter-clockwise from east. Moisture/wind-speed
  units and conversions happen in `Propagator.set_boundary_conditions`.
- `out_of_bounds_mode` ("raise" default, or "ignore") controls whether the
  fire front reaching the grid edge raises `PropagatorOutOfBoundsError`.
- Simulation outputs written locally should go under `results/`
  (git-ignored).
- `docs/spotting.md` is regenerated at MkDocs build time by
  `docs_hooks/spotting.py`, which imports the package and matplotlib to
  redraw figures from the live constants — keep spotting constants and that
  hook in sync.
- Commit messages follow Conventional Commits (`feat`, `fix`, `refactor`,
  `chore`, with scopes like `feat(spotting): ...`).
