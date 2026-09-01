import os

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


def _configure_proj_data() -> None:
    """Prefer pyproj's bundled data without making pyproj mandatory."""
    try:
        from pyproj import datadir
    except ModuleNotFoundError as exc:
        if exc.name != "pyproj":
            raise
        return

    proj_data_dir = datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_data_dir
    os.environ["PROJ_DATA"] = proj_data_dir


_configure_proj_data()
del _configure_proj_data

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
