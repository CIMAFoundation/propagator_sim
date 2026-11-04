"""One-off pre-download of DEM/land-cover tiles covering all of Italy.

After running this once, `propagator.io.data_prep.prepare_area_data` (used
both by `example/italy/prepare_area_data.py` and by the interactive web
app, see `docs/web.md`) finds every tile it needs already cached, for any
point/radius within Italy — no network wait during a simulation run.

Downloads roughly 5-7 GB total (measured tile sizes: ~35-45 MB per DEM
tile, ~60-90 MB per WorldCover tile). Safe to interrupt and re-run: tiles
already present in the cache directory are not re-downloaded.

Usage
-----
uv run python example/italy/download_national_data.py
uv run python example/italy/download_national_data.py --cache-dir /path/to/big/disk
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from propagator.io.data_prep import download_italy_tiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to store downloaded tiles (default: $PROPAGATOR_CACHE_DIR "
        "or ~/.propagator/cache, the same default the web app uses).",
    )
    args = ap.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is None:
        env_dir = os.environ.get("PROPAGATOR_CACHE_DIR")
        cache_dir = (
            Path(env_dir) if env_dir else Path.home() / ".propagator" / "cache"
        )

    print(
        f"Downloading DEM + land-cover tiles for all of Italy into {cache_dir}"
    )
    print("This is a one-off, resumable operation (~5-7 GB total).\n")

    downloaded_bytes = 0

    def on_progress(kind: str, name: str, size: int) -> None:
        nonlocal downloaded_bytes
        if kind in ("dem_ok", "wc_ok"):
            downloaded_bytes += size
            print(
                f"  ok   {name} ({size / 1e6:.1f} MB, {downloaded_bytes / 1e9:.2f} GB so far)"
            )
        else:
            print(f"  skip {name} (no coverage)")

    summary = download_italy_tiles(
        cache_dir=cache_dir, progress_cb=on_progress
    )

    print("\nDone.")
    print(
        f"  DEM tiles:        {summary['n_dem_downloaded']} downloaded, {summary['n_dem_missing']} without coverage"
    )
    print(
        f"  WorldCover tiles: {summary['n_worldcover_downloaded']} downloaded, {summary['n_worldcover_missing']} without coverage"
    )
    print(f"  Total downloaded: {summary['total_bytes'] / 1e9:.2f} GB")
    print(
        "\nThe web app (uv run propagator-web) will now use this cache automatically."
    )


if __name__ == "__main__":
    main()
