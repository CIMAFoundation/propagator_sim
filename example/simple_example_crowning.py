from datetime import timedelta
from time import time

import matplotlib.pyplot as plt
import numpy as np

from propagator.core import (  # type: ignore
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
)

veg = np.full((2000, 2000), 2, dtype=np.int32)
dem = np.zeros((2000, 2000), dtype=np.float32)

cbh = np.full((2000, 2000), 2.0, dtype=np.float32)  # canopy base height
cbd = np.full((2000, 2000), 0.3, dtype=np.float32)  # canopy bulk density


simulator = Propagator(
    dem=dem,
    veg=veg,
    cbd=cbd,
    cbh=cbh,
    realizations=10,
    fuels=FUEL_SYSTEM_LEGACY,
    do_spotting=False,
    out_of_bounds_mode="raise",
)

# set central pixel as ignition point
center_x, center_y = dem.shape[0] // 2, dem.shape[1] // 2


boundary_condition = BoundaryConditions(
    time=0,
    ignitions=[(center_x, center_y)],
    wind_speed=20.0,  # km/h
    wind_dir=90.0,  # degrees from north
    moisture=3.0,  # percentage
)
simulator.set_boundary_conditions(boundary_condition)

start_time = time()
step_time_init = time()
while simulator.time < 3600 * 24:
    next_time = simulator.next_time()
    if next_time is None:
        break
    try:
        simulator.step()
    except PropagatorOutOfBoundsError:
        print("Fire reached out of bounds area, stopping simulation.")
        break
    finally:
        step_time_end = time()
        if simulator.time % 3600 == 0:
            print(
                f"Time: {timedelta(seconds=int(simulator.time))} | elapsed: {step_time_end - step_time_init} seconds"
            )

            # create a plot of the fire probability
            output = simulator.get_output()
            fire_prob = output.fire_probability
            crowning_prob = output.crowning_probability
            active_crowning_prob = output.active_crowning_probability

            plt.figure(figsize=(8, 6))
            plt.imshow(fire_prob, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(label="Fire Probability")
            plt.title(
                f"Fire Probability at time {timedelta(seconds=int(simulator.time))}"
            )
            plt.savefig(
                f"example/output/fire_probability_{simulator.time}.png"
            )
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.imshow(crowning_prob, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(label="Fire Probability")
            plt.title(
                f"Crowning Probability at time {timedelta(seconds=int(simulator.time))}"
            )
            plt.savefig(
                f"example/output/crowning_probability_{simulator.time}.png"
            )
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.imshow(active_crowning_prob, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(label="Fire Probability")
            plt.title(
                f"Active Crowning Probability at time {timedelta(seconds=int(simulator.time))}"
            )
            plt.savefig(
                f"example/output/active_crowning_probability_{simulator.time}.png"
            )
            plt.close()

            step_time_init = time()

end_time = time()
print(f"Simulation completed in {end_time - start_time} seconds.")
