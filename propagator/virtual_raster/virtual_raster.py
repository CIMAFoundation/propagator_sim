from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


class VirtualRasterSource(ABC):
    """Abstract base class for VirtualRaster sources."""
    @abstractmethod
    def from_offsets_and_shape(self, offset, shape) -> VirtualRaster:
        """Create a new VirtualRaster from an offset and shape.
        :param offsets: Offsets of the new VirtualRaster.
        :param shape: Shape of the new VirtualRaster.
        """
        ...


@dataclass(frozen=True)
class VirtualRaster(object):
    """
    Defines a virtual raster class. 
    The virtual raster is a 2D array that can be extended indefinitely.
    The initial domain is defined by a file path or with a fill value.
    The working domain is defined by an offset and a bounding box.
    Only the part of the raster that is needed is loaded, and it is presented as a normal np.array.
    """

    source: VirtualRasterSource = field(default=None, init=True)
    data: np.ndarray = field(default=None, init=True)
    offsets: Tuple[int, int] = field(default=None, init=True)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    def transform_indexes(self, old, indexes):
        """Transform the indexes to the new offset."""
        left, bottom = old.offsets
        new_left, new_bottom = self.offsets
        new_indexes = indexes[0] +  bottom-new_bottom, indexes[1] + left-new_left
        return new_indexes
    
    def extend(self, *, left=0, up=0, right=0, down=0) -> VirtualRaster:
        """Extend the domain adding pixels in the four directions specified by args.
        The new domain is returned.
        left, down, right, up: positive int
        return: VirtualRaster
        """
        # check if the new domain is valid
        if left < 0  or up < 0 or right < 0 or down < 0:
            raise ValueError("The new domain must be positive.")
        
        _data = self.data
        _shape = _data.shape
        _offsets = self.offsets
        
        offsets = _offsets[0] - up, _offsets[1] - left
        shape = _shape[0] + up + down, _shape[1] + left + right

        # check if the new domain is outside the source domain  
        if offsets[0] < 0 or offsets[1] < 0 or offsets[0] + shape[0] > self.source._rio_object.height or offsets[1] + shape[1] > self.source._rio_object.width:
            raise ValueError("The new domain is outside the source domain.")

        # extract data from the raster source
        data = self.source.from_offsets_and_shape(offsets, shape)
        
        # overwrite new data with the old one
        if len(data.shape) == 3:
            data[up:up+_shape[0], left:left+_shape[1], :] = _data 
        else:
            data[up:up+_shape[0], left:left+_shape[1]] = _data
        
        return VirtualRaster(self.source, data, offsets)
