from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import List, Tuple, Type

import numpy as np
import rasterio as rio
from rasterio import windows

from .virtual_raster import VirtualRasterSource, VirtualRaster




@dataclass
class NPVirtualRasterSource(VirtualRasterSource):
    """VirtualRaster source that uses numpy arrays."""
    max_shape: Tuple[int, int] = field(init=True, repr=True)
    z_shape: int = field(init=True, repr=True, default=1)
    dtype: Type = field(init=True, repr=True, default=bool)
    fill: float = field(default=np.nan)

    def __post_init__(self):
        pass        

    def from_offsets_and_shape(self, offsets, shape) -> VirtualRaster:
        """Create a new VirtualRaster from an offset and shape.
        """
        if self.z_shape == 1:
            data = np.full(shape, self.fill, dtype=self.dtype)
        else:
            data = np.full(shape + (self.z_shape, ), self.fill, dtype=self.dtype)

        raster = VirtualRaster(self, data, offsets)
        return raster


    def new(self, offsets, shape):
        return self.from_offsets_and_shape(offsets, shape)
@dataclass
class RIOVirtualRasterSource(VirtualRasterSource):
    transform: rio.transform.Affine = field(default=None, init=False)
    crs: rio.crs.CRS = field(default=None, init=False)
    bounds: rio.coords.BoundingBox = field(default=None, init=False)

    file_path: str = field(default=None, init=True)

    _rio_object: rio.DatasetReader = field(default=None, init=False)

    def _from_bbox(self, west=None, south=None, east=None, north=None) -> VirtualRaster:
        window = windows.from_bounds(left=west, bottom=south, right=east, top=north, transform=self.transform)
        """Get the domain from a bounding box."""
        data = self._rio_object.read(1, window=window)
        
        offsets = round(window.row_off), round(window.col_off)

        raster = VirtualRaster(self, data, offsets)
        return raster


    def from_offsets_and_shape(self, offsets, shape) -> VirtualRaster:
        """Get the domain from an offset and shape."""
        window = windows.Window.from_slices(rows=(offsets[0], offsets[0]+shape[0]), cols=(offsets[1], offsets[1]+shape[1]))
        data = self._rio_object.read(1, window=window)
        raster = VirtualRaster(self, data, offsets)
        return raster

    def __post_init__(self):
        """
        Initialize the domain from a rasterio object.
        """
        if self.file_path is None:
            raise ValueError("File must be defined.")
        
        self._rio_object = rio.open(self.file_path)

        self.transform = self._rio_object.transform
        self.crs = self._rio_object.crs
        self.bounds = self._rio_object.bounds

    def load(self, west=None, south=None, east=None, north=None) -> VirtualRaster:
        """Load the domain from a bounding box."""
        raster = self._from_bbox(west=west, south=south, east=east, north=north)
        return raster
