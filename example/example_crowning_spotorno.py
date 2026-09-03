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
    import rasterio as rio
    # get propagator from the python file directly
    import sys
    sys.path.append("../src")
    from propagator.core import (
        Propagator,
        BoundaryConditions,
        PropagatorOutOfBoundsError
    )
    from propagator.core.numba import get_crowning_initiation_fn
    return (
        BoundaryConditions,
        Propagator,
        PropagatorOutOfBoundsError,
        get_crowning_initiation_fn,
        mo,
    )


@app.cell
def _(mo):
    mo.md("""
    #Crown Fire simulation
    ###Spotorno example
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
        start=1, stop=4,
        step=1, value=2,
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
    cbd_slider = mo.ui.slider(
        label="Canopy Bulk Density [kg/m3]",
        start=0.01, stop=0.5,
        step=0.05, value=0.01,
        show_value=True
    )
    crowning_initiation_model = mo.ui.dropdown(
        options={"Cruz": "cruz", "Perrakis": "perrakis"},
        value="Cruz",
        label="Crowning initiation model",
    )
    return (
        crowning_initiation_model,
        fuel_moisture_slider,
        realizations_slider,
        timesteps_slider,
        wind_dir_slider,
        wind_speed_slider,
        cbd_slider,
    )


@app.cell
def _(
    crowning_initiation_model,
    fuel_moisture_slider,
    mo,
    realizations_slider,
    timesteps_slider,
    wind_dir_slider,
    wind_speed_slider,
    cbd_slider,
):
    mo.vstack([
        realizations_slider,
        timesteps_slider,
        wind_speed_slider,
        wind_dir_slider,
        fuel_moisture_slider,
        cbd_slider,
        crowning_initiation_model,
    ],  justify='center')
    return


@app.cell
def _(rio):
    with rio.open('spotorno_dem_1m.tif') as f:
        dem = f.read(1)
    with rio.open('spotorno_veg_1m.tif') as f:
        veg = f.read(1)
    with rio.open('spotorno_cbh_1m.tif') as f:
        fsg = f.read(1)
    return dem, fsg, veg


@app.cell
def _(
    BoundaryConditions,
    Propagator,
    dem,
    veg,
    fsg,
    fuel_moisture_slider,
    realizations_slider,
    timesteps_slider,
    wind_dir_slider,
    wind_speed_slider,
    cbd_slider,
    crowning_initiation_model,
    get_crowning_initiation_fn,
    np,
):

    # fake cbd
    cbd = np.full(dem.shape, cbd_slider.value, dtype=np.float32)

    # IGNITION
    ign_x, ign_y = dem.shape[0] // 2, dem.shape[1] // 2

    simulator = Propagator(
        dem=dem,
        veg=veg,
        fsg=fsg,
        cbd=cbd,
        realizations=realizations_slider.value,
        cellsize=1,  # 1m
        crowning_init_fn=get_crowning_initiation_fn(
            crowning_initiation_model.value
        ),
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
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig)
        # --- DEM plot
        ax_dem = fig.add_subplot(gs[0, 0])
        ax_dem.set_title('DEM')
        alpha_mask = (simulator.dem > 0).astype(float)
        im_dem = ax_dem.imshow(simulator.dem, cmap="viridis", vmin=0, vmax=365, alpha=alpha_mask)
        fig.colorbar(im_dem, ax=ax_dem, fraction=0.046, pad=0.04)
        # --- Vegetation Map (first row, center cell) ---

        ax_veg = fig.add_subplot(gs[0, 1])
        ax_veg.set_title("Vegetation Map")
        veg_colors = [
            "#FFD700",  # shrubs
            "#808080",  # not vegetated
            "#ADFF2F",  # grassland
            "#8B0000",  # conifers
        ]
        cmap = ListedColormap(veg_colors)
        # 4 classes: 2, 3, 4, 5 -> need 5 boundaries
        bounds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        norm = BoundaryNorm(bounds, cmap.N)
        tick_labels = ["shrubs", "not vegetated", "grassland", "conifers"]
        alpha_mask = (simulator.veg > 0).astype(float)
        im0 = ax_veg.imshow(
            simulator.veg,
            cmap=cmap,
            norm=norm,
            alpha=alpha_mask,
        )
        cbar = fig.colorbar(
            im0,
            ax=ax_veg,
            boundaries=bounds,     # important for discrete classes
            ticks=[2, 3, 4, 5],    # tick positions at class codes
        )
        cbar.ax.set_yticklabels(tick_labels)

        # --- FSG plot
        ax_fsg = fig.add_subplot(gs[0, 2])
        ax_fsg.set_title('FSG')
        alpha_mask = (simulator.fsg >= 0).astype(float)
        im_fsg = ax_fsg.imshow(simulator.fsg, cmap="viridis", vmin=0, vmax=13, alpha=alpha_mask)
        fig.colorbar(im_fsg, ax=ax_fsg, fraction=0.046, pad=0.04)
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
        fig.subplots_adjust(hspace=0.3, wspace=0.4)
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
