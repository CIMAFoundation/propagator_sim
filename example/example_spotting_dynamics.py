from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from time import time
from typing import Any

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
    n_rows, n_cols = 50, 200
    veg_switch_col = 40
    realizations = 100
    wind_speed = 70.0
    wind_dir = 270
    moisture = 0
    max_time = 8 * 3600
    dt = 60

    scenarios = [
        ("no_spotting", None, "No spotting"),
        ("alexandris", "alexandridis", "Alexandridis"),
        ("trucchia", "trucchia", "Trucchia"),
        ("koo", "koo", "Koo"),
        ("pereira", "pereira", "Pereira"),
    ]

    outdir = Path("example/output")
    outdir.mkdir(parents=True, exist_ok=True)

    run_states: list[dict[str, Any]] = []
    for model_key, spotting_model, model_label in scenarios:
        np.random.seed(7)
        dem, veg = build_synthetic_landscape(
            n_rows=n_rows,
            n_cols=n_cols,
            veg_switch_col=veg_switch_col,
            gap_size=3,
        )
        fuels = fuelsystem_from_dict(FUEL_SYSTEM_LEGACY_DICT)

        sim_kwargs: dict[str, Any] = dict(
            dem=dem,
            veg=veg,
            fuels=fuels,
            realizations=realizations,
            cellsize=20.0,
            do_spotting=spotting_model is not None,
            out_of_bounds_mode="ignore",
        )
        if spotting_model is not None:
            sim_kwargs["spotting_fn"] = get_spotting_fn(spotting_model)

        sim = Propagator(**sim_kwargs)

        ign_point = n_cols // 2 - 90
        ignitions = [
            (n_rows // 2, ign_point)
            # for r in range(n_rows // 2 - 8, n_rows // 2 + 9)
        ]
        sim.set_boundary_conditions(
            BoundaryConditions(
                time=0,
                ignitions=ignitions,
                wind_speed=wind_speed,
                wind_dir=wind_dir,
                moisture=moisture,
            )
        )
        run_states.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "sim": sim,
                "done": False,
            }
        )

    model_labels = [state["model_label"] for state in run_states]
    n_models = len(run_states)
    y_ticks = [i * n_rows + (n_rows - 1) / 2 for i in range(n_models)]

    def collect_ordered_runs() -> list[dict[str, Any]]:
        ordered_runs = []
        right_half = slice(n_cols // 2, n_cols)
        for state in run_states:
            sim = state["sim"]
            output = sim.get_output()
            spot_rec = output.spotting_receiving_probability
            ordered_runs.append(
                {
                    "model_key": state["model_key"],
                    "model_label": state["model_label"],
                    "simulation_time_seconds": int(sim.time),
                    "fire_prob": output.fire_probability,
                    "spot_gen": output.spotting_generation_probability,
                    "spot_rec": spot_rec,
                    "mean_intensity": output.fli_mean,
                    "mean_arrival_time": output.mean_arrival_time,
                    "min_arrival_time": output.min_arrival_time,
                    "received_right": float(np.mean(spot_rec[:, right_half])),
                }
            )
        return ordered_runs

    def save_comparison_plot(
        ordered_runs: list[dict[str, Any]], c_time: int, output_path: Path
    ) -> None:
        def stack_metric(metric_key: str) -> np.ndarray:
            return np.vstack([run[metric_key] for run in ordered_runs])

        stacked_fire_prob = stack_metric("fire_prob")
        stacked_spot_gen = stack_metric("spot_gen")
        stacked_spot_rec = stack_metric("spot_rec")
        stacked_mean_intensity = stack_metric("mean_intensity")
        stacked_mean_arrival = stack_metric("mean_arrival_time")
        stacked_min_arrival = np.vstack(
            [
                np.where(
                    run["min_arrival_time"] > 0,
                    run["min_arrival_time"],
                    np.nan,
                )
                for run in ordered_runs
            ]
        )

        def decorate_axis(ax: Any) -> None:
            ax.axvline(
                x=veg_switch_col - 0.5,
                color="cyan",
                linestyle="--",
                linewidth=1,
            )
            for i in range(1, n_models):
                ax.axhline(y=i * n_rows - 0.5, color="white", linewidth=0.6)
            ax.set_yticks(y_ticks, model_labels)
            ax.set_ylabel("Model")

        fig, axes = plt.subplots(
            3, 2, figsize=(12, 16), constrained_layout=True
        )
        ax00, ax01 = axes[0]
        ax10, ax11 = axes[1]
        ax20, ax21 = axes[2]

        m0 = ax00.imshow(stacked_fire_prob, cmap="hot", vmin=0.0, vmax=1.0)
        decorate_axis(ax00)
        fig.colorbar(m0, ax=ax00, shrink=0.8)
        ax00.set_title("Fire Probability")

        m1 = ax01.imshow(stacked_spot_gen, cmap="YlOrRd", vmin=0.0, vmax=1.0)
        decorate_axis(ax01)
        fig.colorbar(m1, ax=ax01, shrink=0.8)
        ax01.set_title("Spotting Generation Probability")

        m2 = ax10.imshow(stacked_spot_rec, cmap="PuRd", vmin=0.0, vmax=1.0)
        decorate_axis(ax10)
        fig.colorbar(m2, ax=ax10, shrink=0.8)
        ax10.set_title("Spotting Receiving Probability")

        max_intensity = float(np.nanpercentile(stacked_mean_intensity, 99))
        m3 = ax11.imshow(
            stacked_mean_intensity,
            cmap="inferno",
            vmin=0.0,
            vmax=max_intensity if max_intensity > 0 else None,
        )
        decorate_axis(ax11)
        fig.colorbar(m3, ax=ax11, shrink=0.8)
        ax11.set_title("Mean Fireline Intensity")

        mean_arrival_vmax = float(np.nanpercentile(stacked_mean_arrival, 99))
        m4 = ax20.imshow(
            stacked_mean_arrival,
            cmap="viridis",
            vmin=0.0,
            vmax=mean_arrival_vmax if mean_arrival_vmax > 0 else None,
        )
        decorate_axis(ax20)
        fig.colorbar(m4, ax=ax20, shrink=0.8)
        ax20.set_title("Mean Arrival Time (s)")

        min_arrival_vmax = float(np.nanpercentile(stacked_min_arrival, 99))
        m5 = ax21.imshow(
            stacked_min_arrival,
            cmap="cividis",
            vmin=0.0,
            vmax=min_arrival_vmax if min_arrival_vmax > 0 else None,
        )
        decorate_axis(ax21)
        fig.colorbar(m5, ax=ax21, shrink=0.8)
        ax21.set_title("Min Arrival Time (s)")

        fig.suptitle(
            "Spotting model comparison\n"
            f"Simulation time: {timedelta(seconds=int(c_time))}"
        )
        fig.savefig(output_path, dpi=140)
        plt.close(fig)

    t0_all = time()
    while True:
        progressed = False
        for state in run_states:
            if state["done"]:
                continue

            sim = state["sim"]
            if sim.time >= max_time:
                state["done"] = True
                continue

            next_time = sim.next_time()
            if next_time is None:
                state["done"] = True
                continue

            try:
                sim.step(seconds=dt)
                progressed = True
            except PropagatorOutOfBoundsError:
                state["done"] = True

        if not progressed:
            break

        ordered_runs = collect_ordered_runs()
        frame_time = max(
            run["simulation_time_seconds"] for run in ordered_runs
        )
        save_comparison_plot(
            ordered_runs,
            frame_time,
            outdir / f"spotting_composite_{frame_time:06d}.png",
        )
    total_runtime = time() - t0_all

    ordered_runs = collect_ordered_runs()
    max_time_seconds = max(
        run["simulation_time_seconds"] for run in ordered_runs
    )
    comparison_path = outdir / "spotting_models_comparison.png"
    save_comparison_plot(ordered_runs, max_time_seconds, comparison_path)

    print("\nFinal summary")
    for run in ordered_runs:
        print(f"[{run['model_label']}]")
        print(
            f"simulation_time={timedelta(seconds=run['simulation_time_seconds'])}"
        )
        print(f"fire_probability_max={float(np.max(run['fire_prob'])):.3f}")
        print(f"spotting_generation_max={float(np.max(run['spot_gen'])):.3f}")
        print(f"spotting_receiving_max={float(np.max(run['spot_rec'])):.3f}")
        print(
            f"spotting_receiving_mean_right_half={run['received_right']:.5f}"
        )

    print(f"\nsequential_runtime_seconds={total_runtime:.2f}")
    for run in ordered_runs:
        model_key = run["model_key"]
        np.save(
            outdir / f"spotting_{model_key}_fire_probability.npy",
            run["fire_prob"],
        )
        np.save(
            outdir / f"spotting_{model_key}_generation_probability.npy",
            run["spot_gen"],
        )
        np.save(
            outdir / f"spotting_{model_key}_receiving_probability.npy",
            run["spot_rec"],
        )
        np.save(
            outdir / f"spotting_{model_key}_mean_intensity.npy",
            run["mean_intensity"],
        )
        np.save(
            outdir / f"spotting_{model_key}_mean_arrival_time.npy",
            run["mean_arrival_time"],
        )
        np.save(
            outdir / f"spotting_{model_key}_min_arrival_time.npy",
            run["min_arrival_time"],
        )
    print(f"saved_numpy_outputs={outdir}")
    print(f"saved_comparison_plot={comparison_path}")


if __name__ == "__main__":
    main()
