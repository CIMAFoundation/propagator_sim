# Getting Started

## Install

Using uv (recommended):

```bash
uv sync --all-extras
```

Using pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## Quick Run

Run the CLI entrypoint:

```bash
uv run propagator --help
```

Example with sample data:

```bash
uv run propagator -f ./example/params.json -of ./example/output -tl 24 -dem ./example/dem.tif -veg ./example/veg.tif
```
