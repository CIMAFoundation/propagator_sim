import logging
import numpy as np

# from propagator.geo import GeographicInfo
# from propagator.loader.tiles import PropagatorDataFromTiles
from propagator.functions import moist_proba_correction_1, p_time_wang
from propagator.loader.geotiff import PropagatorDataFromGeotiffs
from propagator.propagator import (
    Propagator,
    Actions,
    Event,
    Ignitions,
    BoundaryConditions,
)
# from propagator.settings import PropagatorSettings

from propagator.logging_config import configure_logger

configure_logger()

v0 = np.loadtxt("v0_table.txt")
prob_table = np.loadtxt("prob_table.txt")
p_veg = np.loadtxt("p_vegetation.txt")


# settings_dict = {}
# settings = PropagatorSettings.from_dict(settings_dict)

# Load the input data
# settings.


# if settings.run_from_tiles:
#     ...
#     loader = PropagatorDataFromTiles(...)
# else:
#     ...
#     loader = PropagatorDataFromGeotiffs(...)

loader = PropagatorDataFromGeotiffs(
    dem_file="example/dem.tif",
    veg_file="example/veg.tif",
)

# Load the data
dem = loader.get_dem()
veg = loader.get_veg()
geo_info = loader.get_geo_info()

simulator = Propagator(
    dem=dem,
    veg=veg,
    realizations=1,
    ros_0=v0,
    probability_table=prob_table,
    veg_parameters=p_veg,
    do_spotting=False,
    p_time_fn=p_time_wang,
    p_moist_fn=moist_proba_correction_1,
)

ignition_array = np.zeros(dem.shape, dtype=np.uint8)
ignition_array[100:101, 100:101] = 1

events: list[Event] = [
    Event(
        time=0,
        ignitions=Ignitions(ignitions=ignition_array),
        boundary_conditions=BoundaryConditions(
            wind_speed=np.ones(dem.shape) * 10,
            wind_dir=np.ones(dem.shape) * 180,
            moisture=np.ones(dem.shape) * 0.05,
        )
    ),
    Event(
        time=180,
        boundary_conditions=BoundaryConditions(
            wind_speed=np.ones(dem.shape) * 10,
            wind_dir=np.ones(dem.shape) * 0,
            moisture=np.ones(dem.shape) * 0.05,
        )

    )
]

time_resolution = 60
time_limit = 3600

while True:
    next_time = simulator.next_time()
    if next_time is None:
        break

    logging.info(f"Supposed Next time: {next_time}")

    if len(events) > 0:
        event = events[0]
        if event.time <= next_time:
            if event.ignitions is not None:
                simulator.set_ignitions(event.ignitions)

            if event.boundary_conditions is not None:
                simulator.set_boundary_conditions(event.boundary_conditions)

            if event.actions is not None:
                simulator.apply_actions(event.actions)
                
            events.pop(0)


    logging.info(f"Current time: {simulator.time}")
    simulator.step()
    logging.info(f"New time: {simulator.time}")

    if simulator.time % time_resolution == 0:
        output = simulator.get_output()
        # Save the output to the specified folder
        ...

    if simulator.time > time_limit:
        break
