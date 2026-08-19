import os

from pyproj import datadir as _pyproj_datadir

_proj_data_dir = _pyproj_datadir.get_data_dir()
os.environ["PROJ_LIB"] = _proj_data_dir
os.environ["PROJ_DATA"] = _proj_data_dir

from .core import (
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
    PropagatorStats,
    fuelsystem_from_dict,
    get_p_moisture_fn,
    get_p_time_fn,
)

__all__ = [
    "BoundaryConditions",
    "Propagator",
    "PropagatorOutOfBoundsError",
    "PropagatorStats",
    "FUEL_SYSTEM_LEGACY",
    "fuelsystem_from_dict",
    "get_p_moisture_fn",
    "get_p_time_fn",
]

try:
    from .version import __version__
except Exception:
    __version__ = "0.0.0"
