from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

# ignore missing stubs
import rasterio as rio  # type: ignore
from pyproj import CRS, Proj
from rasterio import enums, transform, warp
from rasterio.transform import Affine  # type: ignore


def reproject(
    values: npt.NDArray[np.floating],
    src_trans: Affine,
    src_crs: CRS,
    dst_crs: CRS,
    trim: bool = True,
) -> Tuple[npt.NDArray[np.floating], Affine]:
    """Reproject a raster (optionally trimmed) to a different CRS.

    Returns `(dst, dst_trans)` with the new raster array and affine transform.
    """
    trimmed_values = values
    if trim:
        trimmed_values, src_trans = trim_values(values.copy(), src_trans)

    rows, cols = trimmed_values.shape
    (west, east), (north, south) = transform.xy(
        src_trans, [0, rows], [0, cols], offset="ul"
    )

    with rio.Env():
        dst_trans, dw, dh = warp.calculate_default_transform(
            src_crs=src_crs,
            dst_crs=dst_crs,
            width=cols,
            height=rows,
            left=west,
            bottom=south,
            right=east,
            top=north,
            resolution=None,
        )
        dst = np.empty(shape=(dh, dw))  # type: ignore # warp calculate_default_transform returns inconsistent types

        warp.reproject(
            source=np.ascontiguousarray(trimmed_values),
            destination=dst,
            src_crs=src_crs,
            dst_crs=dst_crs,
            dst_transform=dst_trans,
            src_transform=src_trans,
            resampling=enums.Resampling.nearest,
            num_threads=1,
        )

    return dst, dst_trans


def trim_values(
    values: npt.NDArray[np.floating],
    src_trans,
):
    """Trim a values raster around non-zero area and return new transform."""
    rows, cols = values.shape
    min_row, max_row = int(rows / 2 - 1), int(rows / 2 + 1)
    min_col, max_col = int(cols / 2 - 1), int(cols / 2 + 1)

    mask = values > 0
    v_rows = np.where(mask.sum(axis=1) > 0)[0]
    if len(v_rows) > 0:
        min_row = max(v_rows[0] - 1, 0)
        max_row = min(v_rows[-1] + 2, rows)

    v_cols = np.where(mask.sum(axis=0) > 0)[0]
    if len(v_cols) > 0:
        min_col = max(v_cols[0] - 1, 0)
        max_col = min(v_cols[-1] + 2, cols)

    trim_values = values[min_row:max_row, min_col:max_col]
    rows, cols = trim_values.shape

    (west, east), (north, south) = transform.xy(
        src_trans, [min_row, max_row], [min_col, max_col], offset="ul"
    )
    trim_trans = transform.from_bounds(west, south, east, north, cols, rows)
    return trim_values, trim_trans


@dataclass(frozen=True)
class GeographicInfo:
    crs: CRS
    trans: transform.Affine
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]

    def get_stepx_stepy(self) -> tuple[float, float]:
        step_x = (self.bounds[2] - self.bounds[0]) / self.shape[1]
        step_y = (self.bounds[3] - self.bounds[1]) / self.shape[0]
        return step_x, step_y

    @staticmethod
    def from_bounds(
        west: float,
        south: float,
        east: float,
        north: float,
        rows: int,
        cols: int,
        zone: int,
        proj: str = "utm",
        datum: str = "WGS84",
    ) -> "GeographicInfo":
        """
        Create a GeographicInfo object from bounds and projection parameters.
        :param west: West bound
        :param south: South bound
        :param east: East bound
        :param north: North bound
        :param rows: Number of rows
        :param cols: Number of columns
        :param zone: UTM zone number
        :param proj: Projection type (default is UTM)
        :param datum: Datum (default is WGS84)
        :return: GeographicInfo object
        """
        prj = Proj(proj=proj, zone=zone, datum=datum)
        crs = CRS.from_proj4(prj.srs)
        trans = transform.from_bounds(west, south, east, north, cols, rows)
        bounds = (west, south, east, north)
        shape = (rows, cols)

        return GeographicInfo(crs=crs, trans=trans, bounds=bounds, shape=shape)

    @staticmethod
    def from_file(rio_file: rio.DatasetReader) -> "GeographicInfo":
        """
        Create a GeographicInfo object from a raster file.
        :param file: Path to the raster file
        :return: GeographicInfo object
        """
        bounds = rio_file.bounds
        cols, rows = rio_file.width, rio_file.height
        west, south, east, north = (
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )

        crs = rio_file.crs
        transform = rio_file.transform

        return GeographicInfo(
            crs=crs,
            trans=transform,
            bounds=(west, south, east, north),
            shape=(rows, cols),
        )
