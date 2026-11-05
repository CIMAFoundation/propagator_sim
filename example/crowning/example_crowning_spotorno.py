"""Crown fire simulation on the Spotorno (Liguria) 1 m rasters.

Ported from the `feat/crownfire` marimo notebook (Perello-nico) to a plain
script: the notebook sliders are command-line options. The canopy base
height raster is used as the fuel strata gap, as in the original; the
canopy bulk density is a uniform value (as the notebook's slider was).

    uv run python example/crowning/example_crowning_spotorno.py
    uv run python example/crowning/example_crowning_spotorno.py \\
        --model perrakis --cbd 0.2 --core rust
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio as rio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from propagator.core import (  # noqa: E402
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
)
from propagator.core.constants import FUEL_SYSTEM_LEGACY_DICT  # noqa: E402
from propagator.core.numba import get_p_crowning_initiation_fn  # noqa: E402

HERE = Path(__file__).parent


def read(name: str, dtype: str) -> np.ndarray:
    """Read a Spotorno raster; nodata becomes NaN (float) or 0 (int)."""
    with rio.open(HERE / f"spotorno_{name}_1m.tif") as src:
        values = src.read(1, masked=True)
    fill = 0 if np.dtype(dtype).kind == "i" else np.nan
    return values.astype(dtype).filled(fill)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", choices=["numba", "rust"], default="numba")
    parser.add_argument(
        "--model",
        choices=["cruz", "perrakis"],
        default="cruz",
        help="crowning initiation model",
    )
    parser.add_argument("--realizations", type=int, default=10)
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--wind-speed", type=float, default=13.0)  # km/h
    parser.add_argument("--wind-dir", type=float, default=180.0)  # deg
    parser.add_argument("--moisture", type=float, default=6.0)  # %
    parser.add_argument(
        "--cbd", type=float, default=0.01, help="canopy bulk density kg/m3"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dem = read("dem", "float32")
    veg = read("veg", "int32")
    fsg = read("cbh", "float32")  # used as the fuel strata gap
    cbd = np.full(dem.shape, args.cbd, dtype=np.float32)

    common = dict(
        dem=dem,
        veg=veg,
        realizations=args.realizations,
        cellsize=1.0,  # 1 m rasters
        seed=args.seed,
        out_of_bounds_mode="raise",
        do_crowning=True,
        fsg=fsg,
        cbd=cbd,
    )
    if args.core == "rust":
        from propagator.rust_core import Propagator as RustPropagator

        simulator = RustPropagator(
            fuels_dict=FUEL_SYSTEM_LEGACY_DICT,
            crowning_initiation_model=args.model,
            **common,
        )
    else:
        simulator = Propagator(
            p_crowning_init_fn=get_p_crowning_initiation_fn(args.model),
            **common,
        )

    ignition = (dem.shape[0] // 2, dem.shape[1] // 2)
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            ignitions=[ignition],
            wind_dir=args.wind_dir,
            wind_speed=args.wind_speed,
            moisture=args.moisture,
        )
    )

    time_limit = args.hours * 3600
    while True:
        next_time = simulator.next_time()
        if next_time is None or next_time > time_limit:
            break
        try:
            simulator.step(seconds=min(600, time_limit - simulator.time))
        except PropagatorOutOfBoundsError:
            print("Fire reached the domain boundary, stopping.")
            break
        if simulator.time >= time_limit:
            break

    output = simulator.get_output()
    print(
        f"t={simulator.time}s burnt cells (p>0): "
        f"{int((output.fire_probability > 0).sum())}, crowning cells: "
        f"{int((output.crowning_probability > 0).sum())}, active: "
        f"{int((output.active_crowning_probability > 0).sum())}"
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes[0, 0]
    ax.set_title("DEM")
    image = ax.imshow(np.where(dem > 0, dem, np.nan), cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    ax.set_title("Vegetation")
    cmap = ListedColormap(["#FFD700", "#808080", "#ADFF2F", "#8B0000"])
    bounds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    image = ax.imshow(
        np.where(veg > 0, veg, np.nan),
        cmap=cmap,
        norm=BoundaryNorm(bounds, cmap.N),
    )
    colorbar = fig.colorbar(
        image, ax=ax, boundaries=bounds, ticks=[2, 3, 4, 5]
    )
    colorbar.ax.set_yticklabels(
        ["shrubs", "not vegetated", "grassland", "conifers"]
    )

    ax = axes[0, 2]
    ax.set_title("Fuel strata gap [m]")
    image = ax.imshow(np.where(fsg >= 0, fsg, np.nan), cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax, title, values in (
        (axes[1, 0], "Fire probability", output.fire_probability),
        (axes[1, 1], "Crowning probability", output.crowning_probability),
        (
            axes[1, 2],
            "Active crowning probability",
            output.active_crowning_probability,
        ),
    ):
        ax.set_title(title)
        image = ax.imshow(values, cmap="hot", vmin=0, vmax=1)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.subplots_adjust(hspace=0.3, wspace=0.4)
    path = args.output / f"spotorno_crowning_{args.core}_{args.model}.png"
    fig.savefig(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
