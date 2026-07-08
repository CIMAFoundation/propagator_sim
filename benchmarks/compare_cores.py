"""Compare numba and Rust cores on the same synthetic scenario.

The harness runs one core per process so peak RSS is attributable to that
core. Numba JIT warmup is excluded from the timed loop.

Usage:
    uv run python benchmarks/compare_cores.py --core numba
    uv run python benchmarks/compare_cores.py --core rust
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from typing import Any

import numpy as np

from propagator.core import (
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
)
from propagator.core import (
    Propagator as NumbaPropagator,
)
from propagator.core.constants import FUEL_SYSTEM_LEGACY_DICT


def _maxrss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def _rss_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(out.strip()) / 1024


def _make_simulator(
    core: str,
    *,
    dem: np.ndarray,
    veg: np.ndarray,
    realizations: int,
    seed: int,
    n_threads: int | None,
) -> Any:
    if core == "numba":
        np.random.seed(seed)
        return NumbaPropagator(
            dem=dem,
            veg=veg,
            realizations=realizations,
            fuels=FUEL_SYSTEM_LEGACY,
            do_spotting=False,
            out_of_bounds_mode="ignore",
        )

    from propagator.rust_core import Propagator as RustPropagator

    return RustPropagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels_dict=FUEL_SYSTEM_LEGACY_DICT,
        do_spotting=False,
        seed=seed,
        out_of_bounds_mode="ignore",
        n_threads=n_threads,
    )


def run(
    core: str,
    *,
    grid: int,
    realizations: int,
    hours: int,
    step_seconds: int,
    seed: int,
    n_threads: int | None,
) -> dict[str, Any]:
    veg = np.full((grid, grid), 4, dtype=np.int32)
    dem = np.zeros((grid, grid), dtype=np.float32)

    setup_start = time.perf_counter()
    sim = _make_simulator(
        core,
        dem=dem,
        veg=veg,
        realizations=realizations,
        seed=seed,
        n_threads=n_threads,
    )
    center = grid // 2
    sim.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            ignitions=[(center, center)],
            wind_speed=30.0,
            wind_dir=90.0,
            moisture=0.0,
        )
    )
    setup_seconds = time.perf_counter() - setup_start

    warmup_start = time.perf_counter()
    sim.step(seconds=1)
    warmup_seconds = time.perf_counter() - warmup_start
    rss_after_warmup = _rss_mb()
    maxrss_after_warmup = _maxrss_mb()

    target = hours * 3600
    timed_start = time.perf_counter()
    steps = 0
    while sim.time < target:
        if sim.next_time() is None:
            break
        sim.step(seconds=step_seconds)
        steps += 1
    timed_seconds = time.perf_counter() - timed_start

    output_start = time.perf_counter()
    out = sim.get_output()
    output_seconds = time.perf_counter() - output_start
    stats = out.stats

    return {
        "core": core,
        "grid": grid,
        "realizations": realizations,
        "hours": hours,
        "step_seconds": step_seconds,
        "seed": seed,
        "n_threads": n_threads,
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "timed_seconds": timed_seconds,
        "output_seconds": output_seconds,
        "steps": steps,
        "final_sim_time": sim.time,
        "rss_after_warmup_mb": rss_after_warmup,
        "maxrss_after_warmup_mb": maxrss_after_warmup,
        "maxrss_final_mb": _maxrss_mb(),
        "area_mean_ha": stats.area_mean / 1e4,
        "area_50_ha": stats.area_50 / 1e4,
        "n_active": stats.n_active,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", choices=("numba", "rust"), required=True)
    parser.add_argument("--grid", type=int, default=1000)
    parser.add_argument("--realizations", type=int, default=100)
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--step-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-threads", type=int, default=None)
    args = parser.parse_args()

    result = run(
        args.core,
        grid=args.grid,
        realizations=args.realizations,
        hours=args.hours,
        step_seconds=args.step_seconds,
        seed=args.seed,
        n_threads=args.n_threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
