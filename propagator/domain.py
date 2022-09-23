from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import rasterio as rio
from rasterio import windows


class VirtualRasterSource(ABC):
    transform: rio.transform.Affine = field(default=None, init=False)
    crs: rio.crs.CRS = field(default=None, init=False)
    bounds: rio.coords.BoundingBox = field(default=None, init=False)

    @abstractmethod
    def get_from_bbox(self, left=None, bottom=None, right=None, top=None) -> Tuple[np.ndarray, Tuple[int, int]]:
        pass

    def load(self, left=None, bottom=None, right=None, top=None) -> VirtualRaster:
        """Load the domain from a bounding box."""
        data, offset = self.get_from_bbox(left=left, bottom=bottom, right=right, top=top)
        return VirtualRaster(self, data, offset)

@dataclass
class RIOVirtualRasterSource(VirtualRasterSource):
    file_path: str = field(default=None, init=True)

    _rio_object: rio.DatasetReader = field(default=None, init=False)

    def get_from_bbox(self, left=None, bottom=None, right=None, top=None) -> Tuple[np.ndarray, Tuple[int, int]]:
        window = windows.from_bounds(left=left, bottom=bottom, right=right, top=top, transform=self.transform)
        """Get the domain from a bounding box."""
        data = self._rio_object.read(1, window=window)
        
        offsets = round(window.row_off), round(window.col_off)
        return data, offsets

    def get_from_offset_and_shape(self, offset, shape):
        """Get the domain from an offset and shape."""
        window = windows.Window.from_slices(rows=(offset[0], offset[0]+shape[0]), cols=(offset[1], offset[1]+shape[1]))
        data = self._rio_object.read(1, window=window)
        return data

    def __post_init__(self):
        """Check if the domain is valid."""
        if self.file_path is None:
            raise ValueError("File must be defined.")
        
        self._rio_object = rio.open(self.file_path)

        self.transform = self._rio_object.transform
        self.crs = self._rio_object.crs
        self.bounds = self._rio_object.bounds



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

    
    def transform_indexes(self, old, indexes):
        """Transform the indexes to the new offset."""
        new = [old[i] + indexes[i] for i in range(len(indexes))]
        return new
    
    def extend(self, *, left, bottom, right, top) -> VirtualRaster:
        """Extend the domain adding pixels in the four directions specified by args.
        The new domain is returned.
        left, bottom, right, top: positive int
        """
        _data = self.data
        _shape = _data.shape
        _offsets = self.offsets
        
        offsets = _offsets[0] - bottom, _offsets[1] - left
        shape = _shape[0] + bottom + top, _shape[1] + left + right

        # extract data from the raster source
        data = self.source.get_from_offset_and_shape(offsets, shape)
        # overwrite new data with the old one
        data[bottom:bottom+_shape[0], left:left+_shape[1]] = _data
        
        return VirtualRaster(self.source, data, offsets)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    r = RIOVirtualRasterSource('/Users/mirko/Downloads/ireland_dem_veg/dem_utm29.tif')
    left, bottom, right, top =  602375.0000,5931745.4032,602475.0034,5931845.4992
    raster = r.load(left=left, top=top, bottom=bottom, right=right)
    print(raster.offsets, raster.data.shape)
    plt.figure();plt.imshow(raster.data);plt.title(raster.offsets);
    raster = raster.extend(left=1, bottom=1, right=2, top=2)
    print(raster.offsets, raster.data.shape)
    plt.figure();plt.imshow(raster.data);plt.title(raster.offsets);
    plt.show()
