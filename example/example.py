from time import time

import numpy as np

from propagator.core import (  # type: ignore
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
)

veg = np.full((200, 200), 5, dtype=np.int32)
dem = np.zeros((200, 200), dtype=np.float32)

simulator = Propagator(
    dem=dem,
    veg=veg,
    realizations=10,
    fuels=FUEL_SYSTEM_LEGACY,
    do_spotting=False,
    out_of_bounds_mode="raise",
)

ignition_array = np.zeros(dem.shape, dtype=np.uint8)
ignition_array[100:101, 100:101] = 1


boundary_conditions_list: list[BoundaryConditions] = [
    BoundaryConditions(
        time=0,
        ignition_mask=ignition_array,  # type: ignore
        wind_speed=np.ones(dem.shape) * 40,  # km/h
        wind_dir=np.ones(dem.shape) * 90,  # degrees from north
        moisture=np.ones(dem.shape) * 0,  # percentage
    ),
]
for boundary_condition in boundary_conditions_list:
    simulator.set_boundary_conditions(boundary_condition)

start_time = time()
while simulator.time < 3600 * 60:
    next_time = simulator.next_time()
    if next_time is None:
        break

    step_time_init = time()
    simulator.step()
    step_time_end = time()
    if simulator.time % 3600 == 0:
        print(
            f"Time: {simulator.time} | elapsed: {step_time_end - step_time_init} seconds"
        )

        # create a plot of the fire probability
        fire_prob = simulator.compute_fire_probability()
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.imshow(fire_prob, cmap="hot", vmin=0, vmax=1)
        plt.colorbar(label="Fire Probability")
        plt.title(f"Fire Probability at time {simulator.time} seconds")
        plt.savefig(f"example/output/fire_probability_{simulator.time}.png")
        plt.close()

end_time = time()
print(f"Simulation completed in {end_time - start_time} seconds.")
