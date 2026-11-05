"""Crown fire on a flat, homogeneous conifer stand.

Ported from the `feat/crownfire` branch (Perello-nico). Every cell is a
crowning-prone conifer (legacy fuel 5) under a uniform canopy; the script
saves hourly maps of fire, crowning and active crowning probability.

    uv run python example/crowning/simple_example_crowning.py
    uv run python example/crowning/simple_example_crowning.py --core rust
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from time import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from propagator.core import (  # noqa: E402
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
)
from propagator.core.constants import (  # noqa: E402
    CELLSIZE,
    FUEL_SYSTEM_LEGACY_DICT,
)
from propagator.core.numba import get_p_crowning_initiation_fn  # noqa: E402

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", choices=["numba", "rust"], default="numba")
    parser.add_argument(
        "--model", choices=["cruz", "perrakis"], default="cruz"
    )
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--realizations", type=int, default=10)
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    shape = (args.size, args.size)
    veg = np.full(shape, 5, dtype=np.int32)  # conifers
    dem = np.zeros(shape, dtype=np.float32)
    fsg = np.full(shape, 2.0, dtype=np.float32)  # fuel strata gap [m]
    cbd = np.full(shape, 0.3, dtype=np.float32)  # canopy bulk density

    common = dict(
        dem=dem,
        veg=veg,
        realizations=args.realizations,
        do_spotting=False,
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

    center = (args.size // 2, args.size // 2)
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            ignitions=[center],
            wind_speed=20.0,  # km/h
            wind_dir=90.0,  # degrees from north
            moisture=3.0,  # percentage
        )
    )

    start = time()
    for hour in range(1, args.hours + 1):
        if simulator.next_time() is None:
            break
        try:
            simulator.step(seconds=3600)
        except PropagatorOutOfBoundsError:
            print("Fire reached the domain boundary, stopping.")
            break
        output = simulator.get_output()
        print(
            f"Time: {timedelta(seconds=int(simulator.time))} | "
            f"burnt area: {output.stats.area_mean / 1e4:.1f} ha | "
            f"crown fire area: "
            f"{output.crowning_probability.sum() * CELLSIZE**2 / 1e4:.1f} ha"
        )
        for name, values in (
            ("fire_probability", output.fire_probability),
            ("crowning_probability", output.crowning_probability),
            (
                "active_crowning_probability",
                output.active_crowning_probability,
            ),
        ):
            plt.figure(figsize=(8, 6))
            plt.imshow(values, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(label=name.replace("_", " "))
            plt.title(f"{name} at {timedelta(hours=hour)}")
            plt.savefig(args.output / f"{name}_{hour:02d}h.png")
            plt.close()
    print(f"Simulation completed in {time() - start:.1f} s")


if __name__ == "__main__":
    main()
