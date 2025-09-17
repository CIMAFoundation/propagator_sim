from time import time

import numpy as np

from propagator.core import FUEL_SYSTEM_LEGACY, BoundaryConditions, Propagator
from propagator.io import PropagatorDataFromGeotiffs

loader = PropagatorDataFromGeotiffs(
    dem_file="example/dem.tif",
    veg_file="example/fuel.tif",
)

# Load the data
dem = loader.get_dem()
veg = loader.get_veg()

geo_info = loader.get_geo_info()

simulator = Propagator(
    dem=dem,
    veg=veg,
    realizations=100,
    fuels=FUEL_SYSTEM_LEGACY,
    do_spotting=False,
)

ignition_array = np.zeros(dem.shape, dtype=np.uint8)
ignition_array[100:101, 100:101] = 1


boundary_conditions_list: list[BoundaryConditions] = [
    BoundaryConditions(
        time=0,
        ignition_mask=ignition_array,  # type: ignore
        wind_speed=np.ones(dem.shape) * 10,  # km/h
        wind_dir=np.ones(dem.shape) * 180,  # degrees from north
        moisture=np.ones(dem.shape) * 5,  # percentage
    ),
]
for boundary_condition in boundary_conditions_list:
    simulator.set_boundary_conditions(boundary_condition)

start_time = time()
while simulator.time < 3600:
    next_time = simulator.next_time()
    if next_time is None:
        break

    step_time_init = time()
    simulator.step()
    step_time_end = time()
    if simulator.time % 60 == 0:
        print(
            f"Time: {simulator.time} | elapsed: {step_time_end - step_time_init} seconds"
        )

end_time = time()
print(f"Simulation completed in {end_time - start_time} seconds.")
