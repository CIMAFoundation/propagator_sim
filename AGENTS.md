# Repository Guidelines

## Project Structure & Module Organization
- `src/propagator/core` implements the wildfire CA engine (grid updates, spread models); `src/propagator/io` manages raster/GeoPackage I/O; `src/propagator/cli` backs the `propagator` console entrypoint.
- Keep shared helpers near their domain modules and expose new public APIs in `src/propagator/__init__.py` when relevant.
- Authoritative docs live in `docs/` and build through MkDocs; runnable walk-throughs sit in `example/`. Simulation outputs created locally should stay under `results/` (git-ignored by default).
- Tests mirror the package layout inside `tests/` (e.g., `tests/core` exercises `src/propagator/core`). Add new suites by replicating the source structure.

## Build, Test, and Development Commands
- `uv sync --dev --all-extras` creates the virtualenv with optional CLI/I/O extras and dev tooling.
- `uv run propagator --help` lists runtime switches; use `uv run propagator <config>` to launch scenarios.
- `uv run pytest -q` executes the unit tests; append `-k pattern` or `--maxfail=1` for targeted debugging.
- `uv run ruff check src tests` enforces formatting and lint rules; add `--fix` only after reviewing diffs.
- `uv run mkdocs serve` develops docs with live reload; `uv run mkdocs build` generates the static site.

## Coding Style & Naming Conventions
- Ruff enforces PEP 8–aligned style (79-char lines, E/F/W/Q/I rules with E203/E501 ignored). Run it before pushing.
- Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; `UPPER_CASE` for constants.
- Favor explicit type hints on public interfaces and keep docstrings concise but informative.
- Add brief inline comments only when an algorithm or model tweak is non-obvious.

## Testing Guidelines
- Pytest is configured via `pyproject.toml`; place shared fixtures in a module-level `conftest.py` when needed.
- Name tests after the behavior under test (e.g., `test_scheduler_respects_firebreak`). Prefer parametrization for scenario coverage.
- Seed randomness inside tests to keep runs reproducible, especially for stochastic spread models.
- Update or add integration-style tests whenever core spread logic or I/O contracts change.

## Commit & Pull Request Guidelines
- Follow the Conventional Commit style seen in history (`feat`, `fix`, `refactor`, `chore`, etc.) and keep scopes focused.
- Reference tracking issues in commit bodies or PR descriptions when applicable.
- PRs should explain behavior changes, list verification steps (`uv run pytest`, `uv run propagator` summaries), and attach artifacts/screens when useful.
- Request review before merging; rerun lint and tests after addressing comments to keep CI green.
