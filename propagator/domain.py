from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import List, Tuple, Type

import numpy as np
import rasterio as rio
from rasterio import windows




@dataclass(frozen=True)
class PropagatorState():
    dem: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    veg: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    f_global: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    ros: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    fireline_intensity: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
#    scheduler: Scheduler = field(repr=False, compare=False, hash=False, default=None, init=True)

@dataclass(frozen=True)
class PropagatorStateManager():
    """"""
    dem: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    veg: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    f_global: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    ros: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    fireline_int: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
#    scheduler: Scheduler = field(repr=False, compare=False, hash=False, init=False)

    @staticmethod
    def from_geotiffs(
            dem_path: str, 
            veg_path: str, 
            bounding_box: List[float, float, float, float],
            n_threads: int = 1
            ) -> PropagatorStateManager:
        """Create a PropagatorState from geotiffs.
        :param dem_path: Path to the DEM geotiff.
        :param veg_path: Path to the vegetation geotiff.
        :param bounding_box: Bounding box of the area of interest.
        :param n_threads: Number of threads to use for the simulation.
        """
        dem_source = RIOVirtualRasterSource(file_path=dem_path)
        dem = dem_source.load(*bounding_box)
        veg_source = RIOVirtualRasterSource(file_path=veg_path)
        veg = veg_source.load(*bounding_box)
        f_global = NPVirtualRasterSource(
            max_shape=(10000, 10000), 
            z_shape=n_threads, 
            dtype=bool
        ).new(dem.offsets, dem.shape)

        ros = NPVirtualRasterSource(
            max_shape=(10000, 10000),
            z_shape=n_threads, 
            dtype=np.float32
        ).new(dem.offsets, dem.shape)

        fireline_intensity = NPVirtualRasterSource(
            max_shape=(10000, 10000),
            z_shape=n_threads, 
            dtype=np.float32
        ).new(dem.offsets, dem.shape)

        return PropagatorStateManager(
            dem=dem,
            veg=veg,   
            f_global=f_global,
            ros=ros,
            fireline_int=fireline_intensity
        )


    def extend(self, *, left=0, down=0, up=0, right=0) -> PropagatorStateManager:
        """Extend the state by the given number of pixels in each direction.
        :param left: Number of pixels to extend to the left.
        :param down: Number of pixels to extend down.
        :param up: Number of pixels to extend up.
        :param right: Number of pixels to extend to the right.
        """
        #extend scheduler indexes
        f_global = self.f_global.extend(left=left, down=down, up=up, right=right)
 #       self.scheduler.list = f_global.transform_indexes(self.f_global, self.scheduler.list)

        return PropagatorStateManager(
            dem=self.dem.extend(left=left, down=down, up=up, right=right),
            veg=self.veg.extend(left=left, down=down, up=up, right=right),
            f_global=f_global,
            ros=self.ros.extend(left=left, down=down, up=up, right=right),
            fireline_int=self.fireline_int.extend(left=left, down=down, up=up, right=right),
            
        )

    def __enter__(self) -> PropagatorState:
        """Enter the context manager.
        returns the raw numpy arrays of the state.
        """
        return PropagatorState(
            dem=self.dem.data,
            veg=self.veg.data,
            f_global=self.f_global.data,
            ros=self.ros.data,
            fireline_intensity=self.fireline_int.data,
            #scheduler=self.scheduler
        )
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager.
        """
        pass
        
class VirtualRasterSource(ABC):
    """Abstract base class for VirtualRaster sources."""
    @abstractmethod
    def from_offsets_and_shape(self, offset, shape) -> VirtualRaster:
        """Create a new VirtualRaster from an offset and shape.
        :param offsets: Offsets of the new VirtualRaster.
        :param shape: Shape of the new VirtualRaster.
        """
        ...

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dem_source = '/Users/mirko/Downloads/ireland_dem_veg/dem_utm29_cog.tif'
    veg_source = 'https://propagator-cog-test.s3.eu-west-1.amazonaws.com/veg_utm29_cog2.tif' #'/Users/mirko/Downloads/ireland_dem_veg/veg_utm29.tif'
    west, south, east, north =  411596.1422,5988532.8732,453919.5216,6020290.1967
    state_manager = PropagatorStateManager.from_geotiffs(
        dem_path=dem_source, 
        veg_path=veg_source, 
        n_threads=10, 
        bounding_box=[west, south, east, north])

    with state_manager as state:
        #print(state.dem)
        plt.imshow(state.veg)
        plt.show()

