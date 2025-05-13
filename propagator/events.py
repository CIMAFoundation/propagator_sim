from dataclasses import dataclass
from typing import Optional
import numpy as np
import geopandas as gpd

@dataclass(frozen=True)
class EventData:
    """
    Base class for event data
    """
    pass

@dataclass(frozen=True)
class Ignitions(EventData):
    ignitions: list[np.ndarray] # array of ignitions as list of numpy in the form of Nx2 [row, col]. The list corresponds to the different realizations of the simulation

@dataclass(frozen=True)
class BoundaryConditions(EventData):
    moisture: np.ndarray | None
    wind_dir: np.ndarray | None
    wind_speed: np.ndarray | None


@dataclass(frozen=True)
class Actions(EventData):
    additional_moisture: np.ndarray | None
    vegetation_changes: np.ndarray | None
