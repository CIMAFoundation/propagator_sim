#!/usr/bin/env bash
# Build the two WebAssembly bundles the docs demo (docs/demo/app.html) loads:
#
#   docs/demo/pkg/          single-threaded, runs on any page
#   docs/demo/pkg-threads/  multi-threaded (rayon on web workers); picked by
#                           the page when it is cross-origin isolated
#
# The threaded bundle needs wasm atomics, which the prebuilt wasm32 std lacks,
# so it is built on nightly with `-Z build-std`:
#   rustup toolchain install nightly --component rust-src --target wasm32-unknown-unknown
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/../../docs/demo"

echo "== single-threaded bundle -> docs/demo/pkg" >&2
wasm-pack build "$here" --release --target web --out-dir "$out/pkg"

echo "== threaded bundle -> docs/demo/pkg-threads" >&2
# rustc no longer adds the threading link flags itself when `atomics` is
# enabled, so pass what it used to: a shared, imported memory with a maximum
# (shared memories must declare one), plus the TLS / heap-base exports
# wasm-bindgen's threads transform looks for.
link_flags=(
  --shared-memory
  --import-memory
  --max-memory=2147483648
  --export=__wasm_init_tls
  --export=__tls_size
  --export=__tls_align
  --export=__tls_base
  --export=__heap_base
  --export=__data_end
)
rustflags='-C target-feature=+atomics,+bulk-memory,+mutable-globals'
for flag in "${link_flags[@]}"; do
  rustflags+=" -C link-arg=$flag"
done
RUSTUP_TOOLCHAIN=nightly \
  CARGO_UNSTABLE_BUILD_STD=std,panic_abort \
  RUSTFLAGS="$rustflags" \
  wasm-pack build "$here" --release --target web \
  --out-dir "$out/pkg-threads" --features threads

# wasm-pack writes a `*` .gitignore into each out-dir; the bundles are
# committed, so drop them
rm -f "$out/pkg/.gitignore" "$out/pkg-threads/.gitignore"
