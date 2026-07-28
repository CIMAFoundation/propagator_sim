"""Windowed input loading from cloud-optimized GeoTIFFs (COGs).

Feeds the auto-growing simulation mode: instead of loading a whole
raster, a window around the ignition is read from per-UTM-zone COGs
(local paths, ``s3://`` or ``http(s)://`` URLs) and further windows are
fetched on demand when the fire reaches the domain boundary and the
simulator expands.

World cell coordinates are the pixel coordinates of the selected COG,
so ``Propagator.origin`` maps 1:1 to COG pixels and every grown window
stays on the same grid. DEM and fuel COGs must share that grid exactly.

The initial window can be specified three ways (see
:class:`PropagatorDataFromCogs`): a square centered on a lon/lat
midpoint (`mid_lon`/`mid_lat` + `grid_dim`, the default), an explicit
bounding box in any CRS (`initial_bounds` + `initial_bounds_epsg`), or
an explicit pixel window already expressed in the selected COG's own
grid (`initial_pixel_origin` + `initial_pixel_shape`). The last form is
what lets a caller (e.g. a browser/WASM client) request exactly the
window the core will later `expand()` into, at native resolution,
guaranteed pixel-aligned.
"""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import rasterio as rio
from pyproj import CRS, Transformer
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
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
    serves pixel-aligned windows from it. `load_window` serves any
    further window on that same grid (used for domain growth). Reads
    are boundless: cells outside the raster fill with 0 elevation and
    NO_FUEL vegetation.

    Windows are composed from `block_size`-square blocks on a fixed
    lattice, of which the last `block_cache_size` are kept. Growth then
    re-reads only the ring of new ground around the domain rather than
    the whole window, the interior being blocks the run is already on.

    The initial window is chosen by exactly one of three mutually
    exclusive means:

    - Default: a `grid_dim` x `grid_dim` square centered on the
      (`mid_lon`, `mid_lat`) point.
    - `initial_bounds` (`(minx, miny, maxx, maxy)`, in
      `initial_bounds_epsg`): the smallest window, on the selected
      COG's pixel grid, that fully contains that box. Bounds are
      snapped outward to whole pixels, so the window stays exactly
      pixel-aligned. `mid_lon`/`mid_lat` are then optional — when
      omitted they default to the bounds' centre, used only to pick
      the COG pair — but may still be given explicitly (e.g. the
      original ignition point) if the bounds centre would select the
      wrong COG near a UTM zone boundary.
    - `initial_pixel_origin` + `initial_pixel_shape`: the window
      already expressed in the selected COG's own pixel grid (world
      cell coordinates), used verbatim. `mid_lon`/`mid_lat` are
      required here since they are the only way to pick which COG the
      pixel coordinates refer to.

    `snap_shape_to`, when set (typically to the core's `TILE_SIZE`),
    rounds the `initial_bounds`-derived shape up to a multiple of it.
    This is never required for correctness — `SimulationRunner`'s
    growth shift, not the initial window's own alignment, is what the
    Rust core's `expand()` constrains — but it keeps the window's
    absolute size a round number of tiles for callers that find that
    convenient (e.g. tile-cache bookkeeping).
    """

    dem_urls: list[str]
    fuel_urls: list[str]
    mid_lon: float | None = None
    mid_lat: float | None = None
    grid_dim: int = 3072
    initial_bounds: tuple[float, float, float, float] | None = None
    initial_bounds_epsg: int = 4326
    initial_pixel_origin: tuple[int, int] | None = None
    initial_pixel_shape: tuple[int, int] | None = None
    snap_shape_to: int | None = None
    block_size: int = 512
    block_cache_size: int = 32

    _dem: rio.DatasetReader = field(init=False, repr=False)
    _fuel: rio.DatasetReader = field(init=False, repr=False)
    _fuel_offset: tuple[int, int] = field(init=False, default=(0, 0))
    _initial_origin: tuple[int, int] = field(init=False)
    _initial_shape: tuple[int, int] = field(init=False)
    _initial: tuple[np.ndarray, np.ndarray, GeographicInfo] | None = field(
        init=False, default=None, repr=False
    )
    #: Geometry of the window `load_window` served last, which is the domain
    #: the simulation is on once it has grown. See `get_geo_info`.
    _current_geo_info: GeographicInfo | None = field(
        init=False, default=None, repr=False
    )
    #: LRU of `block_size`-square blocks, keyed by lattice position. At the
    #: default 512 a block is about 1.3 MB across the two bands, and a
    #: 2000x2000 domain spans 16 of them.
    _blocks: "OrderedDict[tuple[int, int], tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]]" = field(
        init=False, default_factory=OrderedDict, repr=False
    )

    def __post_init__(self):
        self._validate_init_args()
        if self.initial_bounds is not None and (
            self.mid_lon is None or self.mid_lat is None
        ):
            self.mid_lon, self.mid_lat = self._bounds_centre_lonlat()
        if self.mid_lon is None or self.mid_lat is None:
            raise PropagatorDataLoaderException(
                "mid_lon/mid_lat are required unless initial_bounds is "
                "given, from which the window centre is derived"
            )

        dem_url, fuel_url = self._select_pair()
        logger.info("Selected COGs: %s / %s", dem_url, fuel_url)
        self._dem = rio.open(gdal_path(dem_url))
        self._fuel = rio.open(gdal_path(fuel_url))
        self._check_same_grid()

        if self.initial_pixel_origin is not None:
            self._initial_origin = (
                int(self.initial_pixel_origin[0]),
                int(self.initial_pixel_origin[1]),
            )
            self._initial_shape = (
                int(self.initial_pixel_shape[0]),  # type: ignore[index]
                int(self.initial_pixel_shape[1]),  # type: ignore[index]
            )
        elif self.initial_bounds is not None:
            self._initial_origin, self._initial_shape = (
                self._window_from_bounds()
            )
        else:
            row, col = self._dem.index(*self._point_in_crs())
            half = self.grid_dim // 2
            self._initial_origin = (int(row) - half, int(col) - half)
            self._initial_shape = (self.grid_dim, self.grid_dim)

    def _validate_init_args(self) -> None:
        has_bounds = self.initial_bounds is not None
        has_pixel_origin = self.initial_pixel_origin is not None
        has_pixel_shape = self.initial_pixel_shape is not None
        if has_pixel_origin != has_pixel_shape:
            raise PropagatorDataLoaderException(
                "initial_pixel_origin and initial_pixel_shape must be "
                "given together"
            )
        if has_bounds and has_pixel_origin:
            raise PropagatorDataLoaderException(
                "initial_bounds and initial_pixel_origin/"
                "initial_pixel_shape are mutually exclusive ways to "
                "specify the initial window"
            )
        if has_pixel_origin and (self.mid_lon is None or self.mid_lat is None):
            raise PropagatorDataLoaderException(
                "mid_lon/mid_lat are required together with "
                "initial_pixel_origin/initial_pixel_shape: they are the "
                "only way to select which COG the pixel coordinates "
                "refer to"
            )

    # --- initial-bounds handling -------------------------------------------

    def _bounds_centre_lonlat(self) -> tuple[float, float]:
        """Centre of `initial_bounds`, reprojected to lon/lat, used only
        to pick the COG pair when `mid_lon`/`mid_lat` are not given."""
        assert self.initial_bounds is not None
        minx, miny, maxx, maxy = self.initial_bounds
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        if self.initial_bounds_epsg == 4326:
            return cx, cy
        transformer = Transformer.from_crs(
            CRS.from_epsg(self.initial_bounds_epsg),
            "EPSG:4326",
            always_xy=True,
        )
        lon, lat = transformer.transform(cx, cy)
        return lon, lat

    def _bounds_in_dem_crs(self) -> tuple[float, float, float, float]:
        """`initial_bounds` reprojected into the selected DEM COG's CRS,
        as an axis-aligned box enclosing all four corners."""
        assert self.initial_bounds is not None
        minx, miny, maxx, maxy = self.initial_bounds
        src_crs = CRS.from_epsg(self.initial_bounds_epsg)
        if src_crs == self._dem.crs:
            return minx, miny, maxx, maxy
        transformer = Transformer.from_crs(
            src_crs, self._dem.crs, always_xy=True
        )
        xs, ys = transformer.transform(
            [minx, maxx, maxx, minx], [miny, miny, maxy, maxy]
        )
        return min(xs), min(ys), max(xs), max(ys)

    def _window_from_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Smallest pixel window, on the DEM COG's own grid, that fully
        contains `initial_bounds` — snapped outward to whole pixels so
        `Propagator.origin` keeps mapping 1:1 to COG pixels."""
        minx, miny, maxx, maxy = self._bounds_in_dem_crs()
        win = window_from_bounds(
            minx, miny, maxx, maxy, transform=self._dem.transform
        )
        row0 = math.floor(win.row_off)
        col0 = math.floor(win.col_off)
        rows = math.ceil(win.row_off + win.height) - row0
        cols = math.ceil(win.col_off + win.width) - col0
        if self.snap_shape_to:
            rows = -(-rows // self.snap_shape_to) * self.snap_shape_to
            cols = -(-cols // self.snap_shape_to) * self.snap_shape_to
        return (row0, col0), (rows, cols)

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
        """Require a shared cell grid, not identical rasters.

        Window coordinates are expressed in the DEM's pixel grid, so the
        fuel COG only has to sample the same cells: same CRS, same
        resolution, and an origin offset by a whole number of cells. Its
        extent may differ (the published pairs routinely cover different
        areas); `_fuel_offset` shifts the window into the fuel's own grid
        and `boundless` reads fill whatever falls outside.
        """
        if self._dem.crs != self._fuel.crs:
            raise PropagatorDataLoaderException(
                "DEM and fuel COGs have different CRS"
            )
        if not np.allclose(
            self._dem.res, self._fuel.res, rtol=1e-9, atol=1e-6
        ):
            raise PropagatorDataLoaderException(
                "DEM and fuel COGs have different resolution"
            )
        row_off = (
            self._dem.transform.f - self._fuel.transform.f
        ) / self._fuel.transform.e
        col_off = (
            self._dem.transform.c - self._fuel.transform.c
        ) / self._fuel.transform.a
        if (
            abs(row_off - round(row_off)) > 1e-6
            or abs(col_off - round(col_off)) > 1e-6
        ):
            raise PropagatorDataLoaderException(
                "DEM and fuel COGs are not pixel-aligned"
            )
        self._fuel_offset = (int(round(row_off)), int(round(col_off)))

    # --- windowed access ----------------------------------------------------

    @property
    def initial_origin(self) -> tuple[int, int]:
        """World (COG pixel) coordinates of the initial window's (0, 0)."""
        return self._initial_origin

    @property
    def initial_shape(self) -> tuple[int, int]:
        """(rows, cols) of the initial window."""
        return self._initial_shape

    @property
    def cellsize(self) -> float:
        return float(self._dem.res[0])

    def load_window(
        self, origin: tuple[int, int], shape: tuple[int, int]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8], GeographicInfo]:
        """Read (dem, veg, geo_info) for a window in world pixel coords.

        Composed from cached square blocks on a fixed lattice, so growth
        re-reads only the ring of new ground around the domain instead of the
        whole window: the interior is the same blocks the run is already on.
        """
        rows, cols = shape
        row0, col0 = origin
        dem = np.empty((rows, cols), dtype=np.float32)
        veg = np.empty((rows, cols), dtype=np.int8)

        size = self.block_size
        for block_row in range(row0 // size, -(-(row0 + rows) // size)):
            for block_col in range(col0 // size, -(-(col0 + cols) // size)):
                block_dem, block_veg = self._block(block_row, block_col)
                pixel_row, pixel_col = block_row * size, block_col * size

                from_row = max(row0, pixel_row)
                to_row = min(row0 + rows, pixel_row + size)
                from_col = max(col0, pixel_col)
                to_col = min(col0 + cols, pixel_col + size)

                target = (
                    slice(from_row - row0, to_row - row0),
                    slice(from_col - col0, to_col - col0),
                )
                source = (
                    slice(from_row - pixel_row, to_row - pixel_row),
                    slice(from_col - pixel_col, to_col - pixel_col),
                )
                dem[target] = block_dem[source]
                veg[target] = block_veg[source]

        window = Window(col0, row0, cols, rows)
        trans = window_transform(window, self._dem.transform)
        bounds = rio.windows.bounds(window, self._dem.transform)
        geo_info = GeographicInfo(
            crs=self._dem.crs,
            trans=trans,
            bounds=bounds,
            shape=shape,
        )
        self._current_geo_info = geo_info
        return dem, veg, geo_info

    def _block(
        self, block_row: int, block_col: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
        """One block of the lattice, read through an LRU cache.

        Blocks are keyed by lattice position alone: their content depends only
        on the rasters, which do not change under a running simulation.
        """
        key = (block_row, block_col)
        cached = self._blocks.get(key)
        if cached is not None:
            self._blocks.move_to_end(key)
            return cached

        block = self._read_window(
            (block_row * self.block_size, block_col * self.block_size),
            (self.block_size, self.block_size),
        )
        self._blocks[key] = block
        while len(self._blocks) > self.block_cache_size:
            self._blocks.popitem(last=False)
        return block

    def _read_window(
        self, origin: tuple[int, int], shape: tuple[int, int]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
        """Read DEM and fuel for a window, straight from the COGs.

        Reads are boundless: a window past the raster's edge comes back filled
        with 0 elevation and NO_FUEL vegetation rather than failing, which is
        what lets a domain grow off the edge of the data.
        """
        window = Window(origin[1], origin[0], shape[1], shape[0])

        dem = self._dem.read(
            1, window=window, boundless=True, fill_value=self._dem.nodata or 0
        )
        if self._dem.nodata is not None:
            dem = np.where(dem == self._dem.nodata, 0.0, dem)
        dem = dem.astype(np.float32)

        # The window is in DEM pixel coordinates; shift it into the fuel's
        # own grid, which may have a different origin (see _check_same_grid).
        fuel_window = Window(
            origin[1] + self._fuel_offset[1],
            origin[0] + self._fuel_offset[0],
            shape[1],
            shape[0],
        )
        veg = self._fuel.read(
            1, window=fuel_window, boundless=True, fill_value=NO_FUEL
        ).astype(np.int8)
        if self._fuel.nodata is not None and self._fuel.nodata != NO_FUEL:
            # The published fuel COGs declare nodata=255, which is out of
            # int8 range: NumPy 2 raises on the scalar cast, so wrap the
            # value the same way `astype` wrapped the band itself (-> -1).
            nodata = np.array(self._fuel.nodata).astype(np.int8)
            veg[veg == nodata] = NO_FUEL

        return dem, veg

    def _load_initial(self):
        if self._initial is None:
            self._initial = self.load_window(
                self._initial_origin, self._initial_shape
            )
        return self._initial

    # --- PropagatorInputDataProtocol ---------------------------------------

    def get_dem(self) -> np.ndarray:
        return self._load_initial()[0]

    def get_veg(self) -> np.ndarray:
        return self._load_initial()[1]

    def get_geo_info(self) -> GeographicInfo:
        """Geometry of the window `load_window` served last.

        Growth serves wider windows through `load_window`, and a caller that
        places geometry on the grid — `SimulationRunner.grow_to_cover`, and
        the services above it — needs the domain the simulation is on *now*.
        Reporting the initial window forever made `grow_to_cover` measure its
        shortfall against a domain the run had already left behind: it grew
        by more than it had to, and could reach `max_domain_cells` without
        ever covering the bounds it was called for.

        Before any window is served this is the initial one, which is what
        every caller reading it at start-up sees.
        """
        self._load_initial()
        assert self._current_geo_info is not None
        return self._current_geo_info

    def close(self) -> None:
        self._dem.close()
        self._fuel.close()
