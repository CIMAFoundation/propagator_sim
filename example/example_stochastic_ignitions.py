from datetime import timedelta
from random import random
from time import time

import matplotlib.pyplot as plt
import numpy as np

from propagator.core import (  # type: ignore
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
)

veg = np.full((2000, 2000), 5, dtype=np.int32)
dem = np.zeros((2000, 2000), dtype=np.float32)
N_REALIZATIONS = 30

simulator = Propagator(
    dem=dem,
    veg=veg,
    realizations=N_REALIZATIONS,
    fuels=FUEL_SYSTEM_LEGACY,
    do_spotting=False,
    out_of_bounds_mode="ignore",
)

ignition_array = np.zeros(dem.shape + (N_REALIZATIONS,), dtype=np.uint8)
center_x, center_y = dem.shape[0] // 2, dem.shape[1] // 2
for r in range(N_REALIZATIONS):
    # set central pixel as ignition point
    x, y = (
        center_x + int(random() * 400) - 200,
        center_y + int(random() * 400) - 200,
    )
    ignition_array[x, y, r] = 1


boundary_condition = BoundaryConditions(
    time=0,
    ignitions=ignition_array,  # type: ignore
    wind_speed=np.ones(dem.shape) * 40,  # km/h
    wind_dir=np.ones(dem.shape) * 90,  # degrees from north
    moisture=np.ones(dem.shape) * 0,  # percentage
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
            ros_mean = output.ros_mean

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
            plt.imshow(ros_mean, cmap="hot")
            plt.colorbar(label="Rate of Spread (mean)")
            plt.title(
                f"Rate of Spread (mean) at time {timedelta(seconds=int(simulator.time))}"
            )
            plt.savefig(f"example/output/rate_of_spread_{simulator.time}.png")
            plt.close()

            step_time_init = time()

end_time = time()
print(f"Simulation completed in {end_time - start_time} seconds.")
