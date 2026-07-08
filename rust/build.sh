#!/usr/bin/env bash
# Build the Rust core Python extension (propagator_rust) and install it into
# the project's virtualenv (.venv). Works around maturin picking the wrong
# interpreter when both CONDA_PREFIX and VIRTUAL_ENV are set.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$root/.venv"

if [[ ! -x "$venv/bin/maturin" ]]; then
    echo "maturin not found in $venv; installing..." >&2
    "$venv/bin/uv" pip install maturin
fi

env -u CONDA_PREFIX VIRTUAL_ENV="$venv" \
    "$venv/bin/maturin" build --release \
    -m "$root/rust/propagator-py/Cargo.toml" \
    -i "$venv/bin/python"

wheel="$(ls -t "$root"/rust/propagator-py/target/wheels/propagator_rust-*.whl | head -1)"
echo "Installing $wheel" >&2
"$venv/bin/uv" pip install --reinstall "$wheel"
