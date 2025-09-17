from typing import Protocol

import numpy as np

from propagator.io.geo import GeographicInfo


class PropagatorDataLoaderException(Exception): ...


class PropagatorInputDataProtocol(Protocol):
    def get_dem(self) -> np.ndarray: ...
    def get_veg(self) -> np.ndarray: ...
    def get_geo_info(self) -> GeographicInfo: ...
