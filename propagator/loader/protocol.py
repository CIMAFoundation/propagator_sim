from typing import Protocol

import numpy as np

from propagator.geo import GeographicInfo


class PropagatorInputDataProtocol(Protocol):
    def get_dem(self) -> np.ndarray: ...
    def get_veg(self) -> np.ndarray: ...
    def get_geo_info(self) -> GeographicInfo: ...
