"""Package init for the wildfire propagator core."""

from .checkpoint import PropagatorCheckpoint
from .constants import TILE_SIZE
from .models import (
    BoundaryConditions,
    PropagatorStats,
)
from .numba import (
    FUEL_SYSTEM_LEGACY,
    fuelsystem_from_dict,
    get_p_moisture_fn,
    get_p_time_fn,
)
from .propagator import (
    Propagator,
    PropagatorOutOfBoundsError,
)

try:
    from ..version import __version__
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "BoundaryConditions",
    "Propagator",
    "PropagatorCheckpoint",
    "PropagatorOutOfBoundsError",
    "PropagatorStats",
    "TILE_SIZE",
    "FUEL_SYSTEM_LEGACY",
    "fuelsystem_from_dict",
    "get_p_moisture_fn",
    "get_p_time_fn",
    "__version__",
]
