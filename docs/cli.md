# CLI Usage

The CLI entrypoint is `propagator` (defined in `project.scripts`).

Show help:

```bash
uv run propagator --help
```

Typical usage:

```bash
uv run propagator \
  -f ./example/params.json \
  -of ./example/output \
  -tl 24 \
  -dem ./example/dem.tif \
  -veg ./example/veg.tif
```

See `propagator_cli/args_parser.py` for all flags and their meanings.
