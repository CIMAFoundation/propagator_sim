# NOTEBOOK FOR CROWN FIRE - EXAMPLE

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from datetime import timedelta
    from matplotlib import pyplot as plt
    # get propagator from the python file directly
    import sys
    sys.path.append("../src")
    from propagator.core import (
        Propagator,
        BoundaryConditions,
        PropagatorOutOfBoundsError
    )
    return (
        BoundaryConditions,
        Propagator,
        PropagatorOutOfBoundsError,
        mo,
    )


@app.cell
def _(mo):
    mo.md("""
    #Crown Fire simulation
    ###Synthetic example
    """)
    return


@app.cell
def _(mo):
    realizations_slider = mo.ui.slider(
        label="Number of simulations",
        start=1, stop=100,
        step=1, value=10,
        show_value=True
    )
    timesteps_slider = mo.ui.slider(
        label="Simulation time [hours]",
        start=1, stop=24,
        step=1, value=3,
        show_value=True
    )
    wind_speed_slider = mo.ui.slider(
        label="Wind Speed [km/h]",
        start=0, stop=50,
        step=1, value=13,
        show_value=True
    )
    wind_dir_slider = mo.ui.slider(
        label="Wind Dir [°]",
        start=0, stop=360,
        step=5, value=180,
        show_value=True
    )
    fuel_moisture_slider = mo.ui.slider(
        label="Fuel Moisture [%]",
        start=0, stop=50,
        step=1, value=6,
        show_value=True
    )
    cbh_slider = mo.ui.slider(
        label="Canopy Base Height [m]",
        start=0, stop=8,
        step=0.5, value=1,
        show_value=True
    )
    cbd_slider = mo.ui.slider(
        label="Canopy Bulk Density [kg/m3]",
        start=0, stop=0.4,
        step=0.1, value=0,
        show_value=True
    )
    return (
        fuel_moisture_slider,
        realizations_slider,
        timesteps_slider,
        wind_dir_slider,
        wind_speed_slider,
        cbh_slider,
        cbd_slider,
    )


@app.cell
def _(
    fuel_moisture_slider,
    mo,
    realizations_slider,
    timesteps_slider,
    wind_dir_slider,
    wind_speed_slider,
    cbh_slider,
    cbd_slider,
):
    mo.vstack([
        realizations_slider,
        timesteps_slider,
        wind_speed_slider,
        wind_dir_slider,
        fuel_moisture_slider,
        cbh_slider,
        cbd_slider,
    ],  justify='center')
    return


@app.cell
def _(
    BoundaryConditions,
    Propagator,
    fuel_moisture_slider,
    realizations_slider,
    timesteps_slider,
    wind_dir_slider,
    wind_speed_slider,
    cbh_slider,
    cbd_slider,
    np
):
    # static inputs
    SIZE_X = 500
    SIZE_Y = 500
    veg = np.full((SIZE_X, SIZE_Y), 2, dtype=np.int32)
    dem = np.zeros((SIZE_X, SIZE_Y), dtype=np.float32)

    # add some conifers
    veg[100:200, 100:400] = 5
    cbh = np.zeros((SIZE_X, SIZE_Y), dtype=np.float32)
    cbd = np.zeros((SIZE_X, SIZE_Y), dtype=np.float32)
    cbh[veg == 5] = cbh_slider.value
    cbd[veg == 5] = cbd_slider.value
    # set central pixel as ignition point
    ign_x, ign_y = dem.shape[0] // 2, dem.shape[1] // 2
    simulator = Propagator(
        dem=dem,
        veg=veg,
        cbh=cbh,
        cbd=cbd,
        realizations=realizations_slider.value
    )
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            ignitions=[(ign_x, ign_y)],
            wind_dir=wind_dir_slider.value,
            wind_speed=wind_speed_slider.value,
            moisture=fuel_moisture_slider.value
        )
    )
    time_limit = timesteps_slider.value * 3600
    return simulator, time_limit


@app.cell
def _(
    PropagatorOutOfBoundsError,
    mo,
    simulator,
    time_limit,
    plt,
    np
):
    with mo.status.spinner(title="Running ...") as _spinner:
        while True:
            next_time = simulator.next_time()
            if next_time is None:
                break
            if next_time > time_limit:
                break

            try:
                simulator.step()
                _spinner.update(subtitle=f"{simulator.time}/{time_limit}")   
            except PropagatorOutOfBoundsError:
                _spinner.update(
                    subtitle="Simulation stopped: fire reached out of bounds area.")
                break

        # getting output
        output = simulator.get_output()
        fire_probability = output.fire_probability
        crowning_probability = output.crowning_probability
        active_crowning_probability = output.active_crowning_probability
        # plot results
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from propagator.core.constants import FUEL_SYSTEM_LEGACY_DICT # type: ignore
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig)
        # --- Vegetation Map (first row, center cell) ---
        ax_veg = fig.add_subplot(gs[0, 1])
        ax_veg.set_title("Vegetation Map")
        veg_colors = [
            "#228B22",  # 1: broadleaves - forest green
            "#7CFC00",  # 2: shrubs - lawn green
            "#D3D3D3",  # 3: non-vegetated - light gray
            "#ADFF2F",  # 4: grassland - green-yellow
            "#006400",  # 5: conifers - dark green
            "#FFD700",  # 6: agro-forestry - gold
            "#8B4513",  # 7: non-fire prone forests - brown
        ]
        cmap = ListedColormap(veg_colors)
        bounds = np.arange(0.5, len(veg_colors) + 1.5)
        norm = BoundaryNorm(bounds, cmap.N)
        tick_labels = [f"{i}: {FUEL_SYSTEM_LEGACY_DICT[i]['name']}" for i in range(1, len(veg_colors) + 1)]
        im0 = ax_veg.imshow(simulator.veg, cmap=cmap, norm=norm)
        cbar = fig.colorbar(im0, ax=ax_veg, ticks=range(1, len(veg_colors) + 1))
        cbar.ax.set_yticklabels(tick_labels)
        cbar.set_label("Vegetation Type", rotation=270, labelpad=15)
        # --- Fire Probability (2nd row, col 0) ---
        ax_fp = fig.add_subplot(gs[1, 0])
        ax_fp.set_title("Fire Probability")
        im1 = ax_fp.imshow(fire_probability, cmap="hot", vmin=0, vmax=1)
        fig.colorbar(im1, ax=ax_fp, fraction=0.046, pad=0.04)
        # --- Crowning Probability (2nd row, col 1) ---
        ax_cp = fig.add_subplot(gs[1, 1])
        ax_cp.set_title("Crowning Probability")
        im2 = ax_cp.imshow(crowning_probability, cmap="hot", vmin=0, vmax=1)
        fig.colorbar(im2, ax=ax_cp, fraction=0.046, pad=0.04)
        # --- Active Crowning Probability (2nd row, col 2) ---
        ax_acp = fig.add_subplot(gs[1, 2])
        ax_acp.set_title("Active Crowning Probability")
        im3 = ax_acp.imshow(active_crowning_probability, cmap="hot", vmin=0, vmax=1)
        fig.colorbar(im3, ax=ax_acp, fraction=0.046, pad=0.04)
        fig.subplots_adjust(hspace=0.3, wspace=0.2)
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
