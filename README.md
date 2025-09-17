# PROPAGATOR: An Operational Cellular-Automata Based Wildfire Simulator

This repository contains the python implementation of the PROPAGATOR wildfire simulation algorithm, developed by CIMA Research Foundation.
The package contains the core simulation engine in the `core` module, I/O utilities in the `io` module, and a command line interface (CLI) in the `cli` module.

Link to the research paper: [PROPAGATOR: An Operational Cellular-Automata Based Wildfire Simulator](https://www.mdpi.com/2571-6255/3/3/26)

## How to use it as a library

Install the package using pip, poetry, uv or any other tool that can install from a git repository.

```bash
pip install git+https://github.com/CIMAFOUNDATION/propagator_sim.git
```

Then, you can use the `propagator` package in your python code.
This command will install the latest version from the `main` branch. It will resolve to the minimal dependencies for the core simulation engine.
If you want to use the I/O utilities or the CLI, you need to install the extra dependencies as well.

```bash
pip install git+https://github.com/CIMAFOUNDATION/propagator_sim.git[io,cli]
```

You can find an example of how to use the package in the `examples/example.py` file.

## How to develop

Clone this repository. Use `uv sync --dev --all-extras` to create a virtual environment and install the required dependencies.

```bash
uv sync
```

## Launch a simulation


```bash
uv run propagator
```

See `uv run propagator --help` for command line args.

## Documentation

This repo uses MkDocs with the Material theme and mkdocstrings for API reference.

- Serve locally: `uv run mkdocs serve`
- Build static site: `uv run mkdocs build`

Docs live under `docs/` and are configured by `mkdocs.yml`.
