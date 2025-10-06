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
