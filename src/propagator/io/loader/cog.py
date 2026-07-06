"""Windowed input loading from cloud-optimized GeoTIFFs (COGs).

Feeds the auto-growing simulation mode: instead of loading a whole
raster, a window around the ignition is read from per-UTM-zone COGs
(local paths, ``s3://`` or ``http(s)://`` URLs) and further windows are
fetched on demand when the fire reaches the domain boundary and the
simulator expands.

World cell coordinates are the pixel coordinates of the selected COG,
so ``Propagator.origin`` maps 1:1 to COG pixels and every grown window
stays on the same grid. DEM and fuel COGs must share that grid exactly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import rasterio as rio
from pyproj import Transformer
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from propagator.core.constants import NO_FUEL
from propagator.io.geo import GeographicInfo
from propagator.io.loader.protocol import (
    PropagatorDataLoaderException,
    PropagatorInputDataProtocol,
)

logger = logging.getLogger(__name__)

_UTM_ZONE_RE = re.compile(r"utm[_-]?(\d{1,2})")


def gdal_path(url: str) -> str:
    """Translate an s3:// or http(s):// URL into a GDAL vsi path."""
    if url.startswith("s3://"):
        return "/vsis3/" + url[len("s3://") :]
    if url.startswith(("http://", "https://")):
        return "/vsicurl/" + url
    return url


def utm_zone_of_lon(lon: float) -> int:
    return int((lon + 180.0) // 6.0) + 1


def _zone_hint(url: str) -> int | None:
    match = _UTM_ZONE_RE.search(url.lower())
    return int(match.group(1)) if match else None


def _contains_point(src: rio.DatasetReader, lon: float, lat: float) -> bool:
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    bounds = src.bounds
    return (
        bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
    )


@dataclass
class PropagatorDataFromCogs(PropagatorInputDataProtocol):
    """Windowed DEM/fuel loader over per-UTM-zone COG sets.

    Selects the COG pair covering (`mid_lon`, `mid_lat`) — using a
    ``utm_<zone>`` filename hint when present, probing otherwise — and
    serves pixel-aligned windows from it. The initial window is
    `grid_dim` cells per side, centered on the point; `load_window`
    serves any further window (used for domain growth). Reads are
    boundless: cells outside the raster fill with 0 elevation and
    NO_FUEL vegetation.
    """

    dem_urls: list[str]
    fuel_urls: list[str]
    mid_lon: float
    mid_lat: float
    grid_dim: int = 3072

    _dem: rio.DatasetReader = field(init=False, repr=False)
    _fuel: rio.DatasetReader = field(init=False, repr=False)
    _initial_origin: tuple[int, int] = field(init=False)
    _initial: tuple[np.ndarray, np.ndarray, GeographicInfo] | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self):
        dem_url, fuel_url = self._select_pair()
        logger.info("Selected COGs: %s / %s", dem_url, fuel_url)
        self._dem = rio.open(gdal_path(dem_url))
        self._fuel = rio.open(gdal_path(fuel_url))
        self._check_same_grid()

        row, col = self._dem.index(*self._point_in_crs())
        half = self.grid_dim // 2
        self._initial_origin = (int(row) - half, int(col) - half)

    # --- COG selection ----------------------------------------------------

    def _point_in_crs(self) -> tuple[float, float]:
        transformer = Transformer.from_crs(
            "EPSG:4326", self._dem.crs, always_xy=True
        )
        return transformer.transform(self.mid_lon, self.mid_lat)

    def _pairs(self) -> list[tuple[str, str]]:
        """Pair DEM and fuel URLs, by utm_<zone> filename hint when every
        URL carries one (the lists may then differ in coverage),
        otherwise by position."""
        dem_by_zone = {_zone_hint(url): url for url in self.dem_urls}
        fuel_by_zone = {_zone_hint(url): url for url in self.fuel_urls}
        if None not in dem_by_zone and None not in fuel_by_zone:
            zones = sorted(set(dem_by_zone) & set(fuel_by_zone))
            return [(dem_by_zone[z], fuel_by_zone[z]) for z in zones]
        if len(self.dem_urls) != len(self.fuel_urls):
            raise PropagatorDataLoaderException(
                "dem and fuel COG lists must have the same length when "
                "they cannot be paired by utm_<zone> filename hints"
            )
        return list(zip(self.dem_urls, self.fuel_urls))

    def _select_pair(self) -> tuple[str, str]:
        pairs = self._pairs()
        # fast path: a utm_<zone> filename hint matching the point's zone
        zone = utm_zone_of_lon(self.mid_lon)
        hinted = [pair for pair in pairs if _zone_hint(pair[0]) == zone]
        for dem_url, fuel_url in hinted + [
            pair for pair in pairs if pair not in hinted
        ]:
            try:
                with rio.open(gdal_path(dem_url)) as src:
                    if _contains_point(src, self.mid_lon, self.mid_lat):
                        return dem_url, fuel_url
            except rio.errors.RasterioIOError as error:
                logger.warning("Cannot open %s: %s", dem_url, error)
        raise PropagatorDataLoaderException(
            f"No COG in the provided set contains point "
            f"({self.mid_lon}, {self.mid_lat})"
        )

    def _check_same_grid(self) -> None:
        if self._dem.crs != self._fuel.crs:
            raise PropagatorDataLoaderException(
                "DEM and fuel COGs have different CRS"
            )
        if self._dem.shape != self._fuel.shape or not np.allclose(
            tuple(self._dem.transform)[:6],
            tuple(self._fuel.transform)[:6],
            rtol=1e-9,
            atol=1e-6,
        ):
            raise PropagatorDataLoaderException(
                "DEM and fuel COGs are not pixel-aligned"
            )

    # --- windowed access ----------------------------------------------------

    @property
    def initial_origin(self) -> tuple[int, int]:
        """World (COG pixel) coordinates of the initial window's (0, 0)."""
        return self._initial_origin

    @property
    def cellsize(self) -> float:
        return float(self._dem.res[0])

    def load_window(
        self, origin: tuple[int, int], shape: tuple[int, int]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8], GeographicInfo]:
        """Read (dem, veg, geo_info) for a window in world pixel coords."""
        window = Window(origin[1], origin[0], shape[1], shape[0])

        dem = self._dem.read(
            1, window=window, boundless=True, fill_value=self._dem.nodata or 0
        )
        if self._dem.nodata is not None:
            dem = np.where(dem == self._dem.nodata, 0.0, dem)
        dem = dem.astype(np.float32)

        veg = self._fuel.read(
            1, window=window, boundless=True, fill_value=NO_FUEL
        ).astype(np.int8)
        if self._fuel.nodata is not None and self._fuel.nodata != NO_FUEL:
            veg[veg == np.int8(self._fuel.nodata)] = NO_FUEL

        trans = window_transform(window, self._dem.transform)
        bounds = rio.windows.bounds(window, self._dem.transform)
        geo_info = GeographicInfo(
            crs=self._dem.crs,
            trans=trans,
            bounds=bounds,
            shape=shape,
        )
        return dem, veg, geo_info

    def _load_initial(self):
        if self._initial is None:
            self._initial = self.load_window(
                self._initial_origin, (self.grid_dim, self.grid_dim)
            )
        return self._initial

    # --- PropagatorInputDataProtocol ---------------------------------------

    def get_dem(self) -> np.ndarray:
        return self._load_initial()[0]

    def get_veg(self) -> np.ndarray:
        return self._load_initial()[1]

    def get_geo_info(self) -> GeographicInfo:
        return self._load_initial()[2]

    def close(self) -> None:
        self._dem.close()
        self._fuel.close()
