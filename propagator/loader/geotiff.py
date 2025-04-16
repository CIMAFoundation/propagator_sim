from attr import dataclass
import numpy as np
import rasterio as rio
from propagator.geo import GeographicInfo
from propagator.input import PropagatorDataLoaderException

import logging

from propagator.loader.protocol import PropagatorInputDataProtocol


def check_input_files_consistency(dem_file, veg_file):
    if dem_file.crs != veg_file.crs:
        raise PropagatorDataLoaderException("CRS of input files are inconsistent")

    err_res = abs(dem_file.res[0] - veg_file.res[0]) / veg_file.res[0]
    if err_res > 0.01:
        raise PropagatorDataLoaderException(
            "Resolution of input files are not consistent"
        )

    bounds_err = np.array(
        [
            dem_file.bounds.left - veg_file.bounds.left,
            dem_file.bounds.right - veg_file.bounds.right,
            dem_file.bounds.top - veg_file.bounds.top,
            dem_file.bounds.bottom - veg_file.bounds.bottom,
        ]
    )
    if np.linalg.norm(bounds_err, 1) > veg_file.res[0] * 2:
        raise PropagatorDataLoaderException(
            "Bounding box of input files are not consistent"
        )


def load_data_from_files(veg_filename, dem_filename):
    with rio.open(veg_filename) as veg_file, rio.open(dem_filename) as dem_file:
        check_input_files_consistency(dem_file, veg_file)
        try:
            dem = dem_file.read(1).astype("int16")
            veg = veg_file.read(1).astype("int8")

            veg[:, (0, 1, 2, -3, -2, -1)] = 0
            veg[(0, 1, 2, -3, -2, -1), :] = 0
            veg[(veg < 0) | (veg > 6)] = 0

            geo_info = GeographicInfo.from_file(dem_file)
        except IOError:
            logging.error("Error reading input files")
            raise

    return dem, veg, geo_info


@dataclass
class PropagatorDataFromGeotiffs(PropagatorInputDataProtocol):
    dem_file: str
    veg_file: str

    def __post_init__(self):
        try:
            with (
                rio.open(self.dem_file) as dem_file,
                rio.open(self.veg_file) as veg_file,
            ):
                check_input_files_consistency(dem_file, veg_file)
        except PropagatorDataLoaderException as e:
            raise PropagatorDataLoaderException(f"Error in input files: {e}") from e

    def get_dem(self) -> np.ndarray:
        dem, _, _ = load_data_from_files(self.veg_file, self.dem_file)
        return dem

    def get_veg(self) -> np.ndarray:
        _, veg, _ = load_data_from_files(self.veg_file, self.dem_file)
        return veg

    def get_geo_info(self) -> GeographicInfo:
        _, _, geo_info = load_data_from_files(self.veg_file, self.dem_file)
        return geo_info
