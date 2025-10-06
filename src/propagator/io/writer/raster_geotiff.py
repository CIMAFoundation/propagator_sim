from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import rasterio as rio  # type: ignore
from pyproj import CRS
from rasterio.transform import Affine  # type: ignore

from propagator.core.models import PropagatorOutput
from propagator.io.geo import GeographicInfo, reproject, trim_values

from .protocol import RasterWriterProtocol


def write_geotiff(
    filename: str | Path,
    values: npt.NDArray[np.floating] | npt.NDArray[np.integer],
    dst_trans: Affine,
    dst_crs: CRS,
    dtype: npt.DTypeLike = np.uint8,
    compression: str = "deflate",
) -> None:
    """Write a single-band GeoTIFF with provided transform and CRS.

    Parameters
    ----------
    filename : str or Path
        Output GeoTIFF file path.
    values : ndarray
        2D array of raster values.
    dst_trans : Affine
        Affine transform for the raster.
    dst_crs : CRS
        Coordinate reference system.
    dtype : DTypeLike, optional
        Data type for output raster, by default np.uint8.
    compression : str, optional
        Compression type for GeoTIFF (e.g., "deflate", "lzw"), by default "deflate".
    """
    with rio.Env():
        with rio.open(
            filename,
            "w",
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype=dtype,
            nodata=0,
            transform=dst_trans,
            crs=dst_crs,
            compress=compression,
        ) as f:
            f.write(values.astype(dtype), indexes=1)


@dataclass
class GeoTiffWriter(RasterWriterProtocol):
    start_date: datetime
    output_folder: Path
    raster_variables_mapping: dict[
        str,
        Callable[[PropagatorOutput], npt.NDArray[np.floating]],
    ]
    geo_info: GeographicInfo
    dst_crs: CRS

    trim: bool = True

    def write_rasters(self, output: PropagatorOutput) -> None:
        for key, fun in self.raster_variables_mapping.items():
            values = fun(output)
            dst_trans = self.geo_info.trans

            if self.geo_info.crs != self.dst_crs:
                values, dst_trans = reproject(
                    values,
                    self.geo_info.trans,
                    self.geo_info.crs,
                    self.dst_crs,
                    trim=self.trim,
                )

            elif self.trim:
                values, dst_trans = trim_values(values, dst_trans)

            tiff_file = self.output_folder / f"{key}_{output.time}.tiff"
            # now it returns the RoS in m/h
            write_geotiff(
                tiff_file, values, dst_trans, self.dst_crs, values.dtype
            )
