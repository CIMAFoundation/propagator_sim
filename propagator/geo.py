from dataclasses import dataclass

import rasterio as rio
from pyproj import Proj
from rasterio import transform


@dataclass(frozen=True)
class GeographicInfo:
    prj: Proj
    trans: transform.Affine
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]

    def get_stepx_stepy(self) -> tuple[float, float]:
        step_x = self.bounds[2] - self.bounds[0] / self.shape[1]
        step_y = self.bounds[3] - self.bounds[1] / self.shape[0]
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
        trans = transform.from_bounds(west, south, east, north, cols, rows)
        bounds = (west, south, east, north)
        shape = (rows, cols)

        return GeographicInfo(prj=prj, trans=trans, bounds=bounds, shape=shape)

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

        proj = rio_file.crs.to_proj4()
        transform = rio_file.transform

        prj = Proj(proj)
        return GeographicInfo(
            prj=prj,
            trans=transform,
            bounds=(west, south, east, north),
            shape=(rows, cols),
        )
