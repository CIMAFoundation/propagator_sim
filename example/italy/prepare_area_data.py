"""Prepare DEM and fuel GeoTIFFs for a circular area around an Italian
(or any) location, ready to feed into `propagator --mode geotiff`.

Thin CLI wrapper around `propagator.io.data_prep.prepare_area_data`, which
downloads Copernicus DEM GLO-30 and ESA WorldCover 10m tiles (both public,
no API key required) and builds an aligned dem.tif/fuel.tif pair. See that
module's docstring for details on the fuel mapping and its limitations.

Usage
-----
uv run python example/italy/prepare_area_data.py \
    --lat 42.4207 --lon 12.1077 --radius-km 30 \
    --output-dir example/italy/viterbo

Then run the simulation with:
uv run propagator --config example/italy/viterbo/config.json \
    --mode geotiff \
    --dem example/italy/viterbo/dem.tif \
    --fuel example/italy/viterbo/fuel.tif \
    --output results/viterbo
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from propagator.io.data_prep import AreaDataError, prepare_area_data


def make_config(
    lat: float,
    lon: float,
    ignition_lat: float,
    ignition_lon: float,
    time_limit: int,
    realizations: int,
    wind_dir: float,
    wind_speed: float,
    moisture: float,
) -> dict:
    return {
        "name": f"Simulation around ({lat}, {lon})",
        "init_date": datetime.now(timezone.utc).strftime("%Y%m%d%H%M"),
        "ignitions": [f"POINT: [{ignition_lat};{ignition_lon}]"],
        "realizations": realizations,
        "time_limit": time_limit,
        "do_spotting": False,
        "time_resolution": 3600,
        "boundary_conditions": [
            {
                "time": 0,
                "w_dir": wind_dir,
                "w_speed": wind_speed,
                "moisture": moisture,
            }
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius-km", type=float, default=30.0)
    ap.add_argument(
        "--cellsize",
        type=float,
        default=30.0,
        help="Output grid resolution in meters (default: 30, matching the "
        "native DEM resolution). Smaller values give finer detail but a "
        "much larger grid (cells scale with radius^2 / cellsize^2).",
    )
    ap.add_argument(
        "--ignition-lat",
        type=float,
        default=None,
        help="Ignition point latitude (default: same as --lat). Note the "
        "ignition must fall on a burnable fuel class; a town/city center "
        "is usually 'non-vegetated' (class 3) and won't ignite.",
    )
    ap.add_argument("--ignition-lon", type=float, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to cache downloaded source tiles (default: "
        "<output-dir>/_cache).",
    )
    ap.add_argument("--time-limit", type=int, default=6 * 3600)
    ap.add_argument("--realizations", type=int, default=10)
    ap.add_argument("--wind-dir", type=float, default=0.0)
    ap.add_argument("--wind-speed", type=float, default=20.0)
    ap.add_argument("--moisture", type=float, default=10.0)
    args = ap.parse_args()

    ignition_lat = args.ignition_lat if args.ignition_lat is not None else args.lat
    ignition_lon = args.ignition_lon if args.ignition_lon is not None else args.lon

    print(f"Preparing data for ({args.lat}, {args.lon}), radius {args.radius_km} km ...")
    try:
        result = prepare_area_data(
            args.lat,
            args.lon,
            args.radius_km,
            cellsize=args.cellsize,
            ignition_lat=ignition_lat,
            ignition_lon=ignition_lon,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
    except AreaDataError as e:
        raise SystemExit(str(e))

    if result.ignition_warning:
        print(f"\nWARNING: {result.ignition_warning}")

    config = make_config(
        args.lat,
        args.lon,
        ignition_lat,
        ignition_lon,
        args.time_limit,
        args.realizations,
        args.wind_dir,
        args.wind_speed,
        args.moisture,
    )
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=4))

    print(f"\nDone. Files written to {args.output_dir}:")
    print(f"  {result.dem_path}")
    print(f"  {result.fuel_path}")
    print(f"  {config_path}")
    print(
        "\nRun the simulation with:\n"
        f"  uv run propagator --config {config_path} --mode geotiff "
        f"--dem {result.dem_path} --fuel {result.fuel_path} --output "
        f"results/{args.output_dir.name}"
    )


if __name__ == "__main__":
    main()
