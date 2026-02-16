from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np

from propagator.core import (  # type: ignore
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
    fuelsystem_from_dict,
    get_spotting_fn,
)

FUEL_SYSTEM_LEGACY_DICT = {
    # key_fuel: dict(
    #     name="name_fuel",
    #     v0=140,  # nominal rate of spread - in m/h
    #     d0=1.5,  # dead fuel load - in kg/m2
    #     hhv=20000,  # higher heating value - in kJ/kg
    #     d1=3,  # live fuel load - in kg/m2 (optional)
    #     humidity=60, # live fuel moisture - in percentage (optional)
    #     spread_probability={  # spread probability to other fuel types
    #         1: 0.3,
    #         2: 0.375,
    #         3: 0.005,
    #         4: 0.45,
    #         5: 0.225,
    #         6: 0.25,
    #         7: 0.075,
    #     },
    #     spotting=False,  # if the fuel type is prone to spotting (optional)
    #     prob_ign_by_embers=0.0,  # prob. of ignition by embers (optional)
    #     burn=True,  # if the fuel type is combustible (optional)
    # ),
    1: dict(
        name="broadleaves",
        v0=140,
        d0=1.5,
        d1=3,
        hhv=20000,
        humidity=60,
        spread_probability={
            1: 0.3,
            2: 0.375,
            3: 0.00,
            4: 0.45,
            5: 0.225,
            6: 0.25,
            7: 0.075,
        },
    ),
    2: dict(
        name="shrubs",
        v0=140,
        d0=1,
        d1=3,
        hhv=21000,
        humidity=45,
        spread_probability={
            1: 0.375,
            2: 0.375,
            3: 0.00,
            4: 0.475,
            5: 0.325,
            6: 0.25,
            7: 0.1,
        },
    ),
    3: dict(
        name="non-vegetated",
        v0=20,
        d0=0.1,
        hhv=100,
        spread_probability={
            1: 0.00,
            2: 0.00,
            3: 0.00,
            4: 0.00,
            5: 0.00,
            6: 0.00,
            7: 0.00,
        },
        burn=False,
    ),
    4: dict(
        name="grassland",
        v0=120,
        d0=0.5,
        hhv=17000,
        spread_probability={
            1: 0.25,
            2: 0.35,
            3: 0.00,
            4: 0.475,
            5: 0.1,
            6: 0.3,
            7: 0.075,
        },
    ),
    5: dict(
        name="conifers",
        v0=200,
        d0=1,
        d1=4,
        hhv=21000,
        humidity=55,
        spread_probability={
            1: 0.275,
            2: 0.4,
            3: 0.00,
            4: 0.475,
            5: 0.35,
            6: 0.475,
            7: 0.275,
        },
        spotting=True,
        prob_ign_by_embers=0.4,
    ),
    6: dict(
        name="agro-forestry areas",
        v0=120,
        d0=0.5,
        d1=2,
        hhv=19000,
        humidity=60,
        spread_probability={
            1: 0.25,
            2: 0.3,
            3: 0.00,
            4: 0.375,
            5: 0.2,
            6: 0.35,
            7: 0.075,
        },
    ),
    7: dict(
        name="non-fire prone forests",
        v0=60,
        d0=1,
        d1=2,
        hhv=18000,
        humidity=65,
        spread_probability={
            1: 0.25,
            2: 0.375,
            3: 0.00,
            4: 0.475,
            5: 0.35,
            6: 0.25,
            7: 0.075,
        },
    ),
}


def build_synthetic_landscape(
    n_rows: int, n_cols: int, veg_switch_col: int, gap_size: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic DEM and vegetation map.

    Left half: conifers (fuel id 5, spotting-prone in legacy fuel system).
    Right half: grassland (fuel id 4, not spotting-prone).
    """
    dem = np.zeros((n_rows, n_cols), dtype=np.float32)
    veg = np.full((n_rows, n_cols), 2, dtype=np.int32)
    veg[:, :veg_switch_col] = 5
    if gap_size > 0:
        veg[:, veg_switch_col : veg_switch_col + gap_size] = (
            3  # add a line of transition fuel to amplify spotting at the interface
        )
    return dem, veg


def main() -> None:
    np.random.seed(7)

    n_rows, n_cols = 50, 200
    veg_switch_col = 100
    realizations = 100
    dem, veg = build_synthetic_landscape(
        n_rows=n_rows, n_cols=n_cols, veg_switch_col=veg_switch_col, gap_size=3
    )
    fuels = fuelsystem_from_dict(FUEL_SYSTEM_LEGACY_DICT)

    # Available spotting models: "alexandridis" (default), "trucchia", "pereira", "koo"
    # To use a different model, import and pass spotting_fn:
    # from propagator.core.numba import get_spotting_fn
    spotting_fn = get_spotting_fn("trucchia")  # or "pereira", "koo"

    sim = Propagator(
        dem=dem,
        veg=veg,
        fuels=fuels,
        realizations=realizations,
        cellsize=20.0,
        do_spotting=True,
        out_of_bounds_mode="ignore",
        spotting_fn=spotting_fn,  # Optional: uncomment to use a different model
    )

    # Strong, uniform wind to amplify long-range spotting transport.
    wind_speed = 50.0
    wind_dir = 270
    moisture = 0

    # Ignition inside the spotting-prone side, near the interface.
    ign_col = n_cols // 2 - 90
    ignitions = [(r, ign_col) for r in range(n_rows // 2 - 8, n_rows // 2 + 9)]
    sim.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            ignitions=ignitions,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            moisture=moisture,
        )
    )

    max_time = 8 * 3600
    dt = 60
    step = 0
    t0 = time()
    outdir = Path("example/output")
    outdir.mkdir(parents=True, exist_ok=True)

    def save_frame(
        c_time: int,
        fire_prob: np.ndarray,
        spot_gen: np.ndarray,
        spot_rec: np.ndarray,
        mean_intensity: np.ndarray,
        mean_arrival: np.ndarray,
        min_arrival: np.ndarray,
    ) -> None:
        fig, axes = plt.subplots(
            3, 2, figsize=(12, 14), constrained_layout=True
        )
        ax00, ax01 = axes[0]
        ax10, ax11 = axes[1]
        ax20, ax21 = axes[2]

        m0 = ax00.imshow(fire_prob, cmap="hot", vmin=0.0, vmax=1.0)
        # add a line to indicate the fuel type interface
        ax00.axvline(
            x=veg_switch_col - 0.5, color="cyan", linestyle="--", linewidth=1
        )
        fig.colorbar(m0, ax=ax00, shrink=0.8)
        ax00.set_title("Fire Probability")
        ax00.set_axis_off()

        m1 = ax01.imshow(spot_gen, cmap="YlOrRd", vmin=0.0, vmax=1.0)
        ax01.axvline(
            x=veg_switch_col - 0.5, color="cyan", linestyle="--", linewidth=1
        )
        fig.colorbar(m1, ax=ax01, shrink=0.8)
        ax01.set_title("Spotting Generation Probability")
        ax01.set_axis_off()

        m2 = ax10.imshow(spot_rec, cmap="PuRd", vmin=0.0, vmax=1.0)
        ax10.axvline(
            x=veg_switch_col - 0.5, color="cyan", linestyle="--", linewidth=1
        )
        fig.colorbar(m2, ax=ax10, shrink=0.8)
        ax10.set_title("Spotting Receiving Probability")
        ax10.set_axis_off()

        max_intensity = float(np.nanpercentile(mean_intensity, 99))
        m3 = ax11.imshow(
            mean_intensity,
            cmap="inferno",
            vmin=0.0,
            vmax=max_intensity if max_intensity > 0 else None,
        )
        ax11.axvline(
            x=veg_switch_col - 0.5, color="cyan", linestyle="--", linewidth=1
        )
        fig.colorbar(m3, ax=ax11, shrink=0.8)
        ax11.set_title("Mean Fireline Intensity")
        ax11.set_axis_off()

        mean_arrival_vmax = float(np.nanpercentile(mean_arrival, 99))
        m4 = ax20.imshow(
            mean_arrival,
            cmap="viridis",
            vmin=0.0,
            vmax=mean_arrival_vmax if mean_arrival_vmax > 0 else None,
        )
        ax20.axvline(
            x=veg_switch_col - 0.5, color="cyan", linestyle="--", linewidth=1
        )
        fig.colorbar(m4, ax=ax20, shrink=0.8)
        ax20.set_title("Mean Arrival Time (s)")
        ax20.set_axis_off()

        min_arrival_masked = np.where(min_arrival > 0, min_arrival, np.nan)
        min_arrival_vmax = float(np.nanpercentile(min_arrival_masked, 99))
        m5 = ax21.imshow(
            min_arrival_masked,
            cmap="cividis",
            vmin=0.0,
            vmax=min_arrival_vmax if min_arrival_vmax > 0 else None,
        )
        ax21.axvline(
            x=veg_switch_col - 0.5, color="cyan", linestyle="--", linewidth=1
        )
        fig.colorbar(m5, ax=ax21, shrink=0.8)
        ax21.set_title("Min Arrival Time (s)")
        ax21.set_axis_off()

        fig.suptitle(f"Simulation time: {timedelta(seconds=int(c_time))}")
        fig.savefig(outdir / f"spotting_composite_{c_time:06d}.png", dpi=140)
        plt.close(fig)

    while sim.time < max_time:
        current_time = int(sim.time)
        next_time = sim.next_time()
        if next_time is None:
            break
        scheduled_delta = int(next_time) - current_time
        try:
            sim.step(seconds=dt)
        except PropagatorOutOfBoundsError:
            break
        step += 1
        actual_delta = int(sim.time) - current_time
        output = sim.get_output()
        save_frame(
            int(sim.time),
            output.fire_probability,
            output.spotting_generation_probability,
            output.spotting_receiving_probability,
            output.fli_mean,
            output.mean_arrival_time,
            output.min_arrival_time,
        )
        print(
            f"step={step:04d} "
            f"curr={timedelta(seconds=current_time)} "
            f"next={timedelta(seconds=int(next_time))} "
            f"scheduled_dt={timedelta(seconds=scheduled_delta)} "
            f"actual_dt={timedelta(seconds=actual_delta)}"
        )
        if step % 10 == 0:
            print(
                f"time={timedelta(seconds=int(sim.time))} "
                f"mean_area={output.stats.area_mean / 10000:.1f} ha "
                f"active={output.stats.n_active}"
            )

    output = sim.get_output()
    runtime = time() - t0

    fire_prob = output.fire_probability
    spot_gen = output.spotting_generation_probability
    spot_rec = output.spotting_receiving_probability
    mean_intensity = output.fli_mean
    mean_arrival_time = output.mean_arrival_time
    min_arrival_time = output.min_arrival_time

    right_half = slice(n_cols // 2, n_cols)
    received_right = float(np.mean(spot_rec[:, right_half]))

    print("\nFinal summary")
    print(f"simulation_time={timedelta(seconds=int(sim.time))}")
    print(f"runtime_seconds={runtime:.2f}")
    print(f"fire_probability_max={float(np.max(fire_prob)):.3f}")
    print(f"spotting_generation_max={float(np.max(spot_gen)):.3f}")
    print(f"spotting_receiving_max={float(np.max(spot_rec)):.3f}")
    print(f"spotting_receiving_mean_right_half={received_right:.5f}")

    np.save(outdir / "spotting_fire_probability.npy", fire_prob)
    np.save(outdir / "spotting_generation_probability.npy", spot_gen)
    np.save(outdir / "spotting_receiving_probability.npy", spot_rec)
    np.save(outdir / "spotting_mean_intensity.npy", mean_intensity)
    np.save(outdir / "spotting_mean_arrival_time.npy", mean_arrival_time)
    np.save(outdir / "spotting_min_arrival_time.npy", min_arrival_time)
    print(f"saved_numpy_outputs={outdir}")
    print(f"saved_composite_frames={outdir}")


if __name__ == "__main__":
    main()
