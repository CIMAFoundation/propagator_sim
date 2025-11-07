"""Package init for the wildfire propagator core."""

from .models import (
    BoundaryConditions,
    PropagatorStats,
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
    "PropagatorOutOfBoundsError",
    "PropagatorStats",
    "__version__",
]
