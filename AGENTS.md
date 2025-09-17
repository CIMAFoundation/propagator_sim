# Repository Guidelines

## Project Structure & Module Organization
- `propagator/`: Core fire spread logic including `propagator.py`, `scheduler.py`, and helper functions.
- `propagator_io/`: Configuration models and IO utilities such as `configuration.py`, `input.py`, and `output.py`.
- `propagator_cli/`: CLI entrypoints (`cli.py`, `args_parser.py`, `console.py`) and supporting console helpers.
- `example/`: Lightweight DEM, vegetation, and parameter samples for smoke tests.
- `tests/`: Pytest suite (e.g., `tests/test_propagator.py`) covering propagation rules and IO flows.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create and activate a local virtual environment.
- `pip install -e .[dev]` or `uv sync`: Install project and dev dependencies using pip or uv.
- `python main.py --help`: Inspect CLI options; use sample params via `python main.py -f ./example/params.json -of ./example/output -tl 24`.
- `pytest -q`: Run the full automated test suite.
- `ruff check .` / `ruff format .`: Lint and format codebase; run lint before formatting when iterating.

## Coding Style & Naming Conventions
- Target Python ≥ 3.13, 4-space indentation, explicit type hints for public APIs.
- Modules use `snake_case.py`; classes `CamelCase`; functions and variables `snake_case`.
- Organize imports per Ruff defaults; drop unused code. Keep dataclasses and Pydantic models documented with short docstrings.

## Testing Guidelines
- Primary framework: Pytest with deterministic seeding when randomness is involved.
- Place tests under `tests/` with filenames `test_*.py` and functions `test_*`.
- Run `pytest -q` locally before submitting changes; add coverage for new logic and edge cases.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) as in existing history. Squash before merging when possible.
- PRs should describe behavior changes, link tracked issues, and note any CLI flag updates.
- Include relevant test output or CLI snippets when touching runtime behavior; update `example/` assets or docs if inputs change.
