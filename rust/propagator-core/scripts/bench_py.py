"""Python/numba twin of the Rust `bench` binary.

Same scenario (homogeneous grassland, point ignition, wind 30 km/h @ 90 deg,
moisture 0%), so timings and burned-area aggregates line up with the Rust
core. JIT compilation is excluded from the timed region via a warmup step.

Usage: bench_py.py <grid> <realizations> <hours> [seed] [dump.f32]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from propagator.core import FUEL_SYSTEM_LEGACY, BoundaryConditions, Propagator


def main() -> None:
    argv = sys.argv
    n = int(argv[1]) if len(argv) > 1 else 1000
    reals = int(argv[2]) if len(argv) > 2 else 100
    hours = int(argv[3]) if len(argv) > 3 else 12
    seed = int(argv[4]) if len(argv) > 4 else 12345
    dump = argv[5] if len(argv) > 5 else None

    np.random.seed(seed)
    veg = np.full((n, n), 4, dtype=np.int32)  # grassland (fuel code 4)
    dem = np.zeros((n, n), dtype=np.float32)

    sim = Propagator(
        dem=dem,
        veg=veg,
        realizations=reals,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )
    c = n // 2
    sim.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            ignitions=[(c, c)],
            wind_speed=30.0,
            wind_dir=90.0,
            moisture=0.0,
        )
    )

    # Warmup: trigger numba JIT so it isn't counted in the timed region.
    sim.step(seconds=1)

    target = hours * 3600
    t0 = time.perf_counter()
    steps = 0
    while sim.time < target:
        if sim.next_time() is None:
            break
        sim.step(seconds=3600)
        steps += 1
    elapsed = time.perf_counter() - t0

    out = sim.get_output()
    s = out.stats
    area_mean_ha = s.area_mean / 1e4
    area_50_ha = s.area_50 / 1e4

    print(
        f"PY    grid={n}x{n} reals={reals} steps={steps} sim_time={sim.time}s",
        file=sys.stderr,
    )
    print(
        f"      wall={elapsed:.3f}s  area_mean={area_mean_ha:.1f}ha  "
        f"area_50={area_50_ha:.1f}ha  n_active={s.n_active}",
        file=sys.stderr,
    )
    print(
        f"PY\t{n}\t{reals}\t{sim.time}\t{elapsed:.4f}\t{area_mean_ha:.2f}\t{area_50_ha:.2f}"
    )

    if dump is not None:
        fp = np.asarray(out.fire_probability, dtype="<f4")
        fp.tofile(dump)
        print(
            f"      dumped fire_probability [{n}x{n} f32] -> {dump}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
