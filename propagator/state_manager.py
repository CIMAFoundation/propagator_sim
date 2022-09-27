from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import List, Tuple, Type

import numpy as np
import rasterio as rio
from rasterio import windows

from .utils.scheduler import Scheduler
from .virtual_raster.sources import NPVirtualRasterSource, RIOVirtualRasterSource
from .virtual_raster.virtual_raster import VirtualRaster




@dataclass(frozen=True)
class PropagatorStateSnapshot():
    dem: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    veg: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    f_global: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    ros: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    fireline_intensity: np.ndarray = field(repr=False, compare=False, hash=False, default=None, init=True)
    
    active_pixels: List[Tuple[int, int, int]] = field(repr=False, compare=False, hash=False, default=None, init=True)
    time: float = field(repr=True, compare=True, hash=True, default=0.0, init=True)

@dataclass(frozen=True)
class PropagatorState():
    """"""
    dem: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    veg: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    f_global: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    ros: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    fireline_int: VirtualRaster = field(repr=False, compare=False, hash=False, init=True)
    scheduler: Scheduler = field(repr=False, compare=False, hash=False, init=False)

    _snapshot: PropagatorStateSnapshot = field(repr=False, compare=False, hash=False, default=None, init=False)

    @staticmethod
    def from_geotiffs(
            dem_path: str, 
            veg_path: str, 
            bounding_box: List[float, float, float, float],
            n_threads: int = 1
            ) -> PropagatorState:
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

        return PropagatorState(
            dem=dem,
            veg=veg,   
            f_global=f_global,
            ros=ros,
            fireline_int=fireline_intensity,
            scheduler=Scheduler()
        )


    def extend(self, *, left=0, down=0, up=0, right=0):
        """Extend the state by the given number of pixels in each direction.
        :param left: Number of pixels to extend to the left.
        :param down: Number of pixels to extend down.
        :param up: Number of pixels to extend up.
        :param right: Number of pixels to extend to the right.
        """
        #extend scheduler indexes
        self.f_global = self.f_global.extend(left=left, down=down, up=up, right=right)
        self.dem=self.dem.extend(left=left, down=down, up=up, right=right)
        self.veg=self.veg.extend(left=left, down=down, up=up, right=right)
        self.ros=self.ros.extend(left=left, down=down, up=up, right=right)
        self.fireline_int = self.fireline_int.extend(left=left, down=down, up=up, right=right)

        # add offsets to active pixels
        self.scheduler.list = [(x + left, y + down, z) for x, y, z in self.scheduler.list]
          

    def _check_boundaries(self):
        """Check if boundaries will be crossed and extend the domain if needed.
        returns None if no extension is needed, otherwise returns the number of pixels to extend in each direction.
        """
        pass

    def __iter__(self) -> PropagatorStateSnapshot:
        """Enter the context manager.
        returns the raw numpy arrays of the state.
        """
        #check if boundaries will be crossed
        if self._check_boundaries():
            self.extend()

        
        time, active_pixels = self.scheduler.pop()
        self._snapshot = PropagatorStateSnapshot(
            dem=self.dem.data,
            veg=self.veg.data,
            f_global=self.f_global.data,
            ros=self.ros.data,
            fireline_intensity=self.fireline_int.data,
            active_pixels=active_pixels,
            time=time
        )
        return self._snapshot
    
    def push_updates(self, coords: List[Tuple[int, int, int]], time: float):
        """Add active pixels to the scheduler.
        :param active_pixels: List of active pixels.
        """
        self.scheduler.push(coords, time)






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dem_source = '/Users/mirko/Downloads/ireland_dem_veg/dem_utm29_cog.tif'
    veg_source = 'https://propagator-cog-test.s3.eu-west-1.amazonaws.com/veg_utm29_cog2.tif' #'/Users/mirko/Downloads/ireland_dem_veg/veg_utm29.tif'
    west, south, east, north =  411596.1422,5988532.8732,453919.5216,6020290.1967

    state_manager = PropagatorState.from_geotiffs(
        dem_path=dem_source, 
        veg_path=veg_source, 
        n_threads=10,
        bounding_box=[west, south, east, north]
    )

    for state in state_manager:
        pass
    # while True:# c'è propagazione
    #     with state_manager as state:
    #         #print(state.dem)
    #         plt.imshow(state.dem)
    #         plt.show()
            
    #         #propago
    #         #check if boundaries reached
    #     # if boundary_reached extend
    #     state_manager.extend(left=100, right=100, up=100, down=100)

