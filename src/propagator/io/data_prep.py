"""Build DEM/fuel GeoTIFFs for an area around a location, from public data.

Downloads Copernicus DEM GLO-30 (~30 m) and ESA WorldCover 10 m v200 (2021)
tiles (both public, no API key required, hosted as open COGs on AWS S3),
reprojects/merges them onto one aligned grid, and remaps WorldCover land
cover classes onto PROPAGATOR's legacy fuel codes (see
`example/fuel_config.yaml`): 1 broadleaves, 2 shrubs, 3 non-vegetated,
4 grassland, 5 conifers, 6 agro-forestry areas, 7 non-fire prone forests.

WorldCover does not distinguish broadleaf from conifer forest, so class 10
(tree cover) is mapped to "broadleaves" as a default reasonable for central
Italy; adjust WORLDCOVER_TO_LEGACY_FUEL for other regions. This is a
reasonable default for a quick simulation, not a substitute for a proper
fuel map (e.g. built from Corine Land Cover + a species layer).

Shared by `example/italy/prepare_area_data.py` (CLI) and `propagator.web`
(interactive backend) so both build DEM/fuel grids the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import numpy.typing as npt
import rasterio as rio
import requests
import utm
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject as rio_reproject

from propagator.io.geo import GeographicInfo

DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
WORLDCOVER_BUCKET = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
)

WORLDCOVER_TO_LEGACY_FUEL = {
    10: 1,  # tree cover -> broadleaves
    20: 2,  # shrubland -> shrubs
    30: 4,  # grassland -> grassland
    40: 6,  # cropland -> agro-forestry areas
    50: 3,  # built-up -> non-vegetated
    60: 3,  # bare / sparse vegetation -> non-vegetated
    70: 3,  # snow and ice -> non-vegetated
    80: 3,  # permanent water bodies -> non-vegetated
    90: 3,  # herbaceous wetland -> non-vegetated
    95: 1,  # mangroves -> broadleaves (not expected in Italy)
    100: 3,  # moss and lichen -> non-vegetated
}

NON_VEGETATED_FUEL = 3

# west, south, east, north — covers the mainland plus Sicily and Sardinia
ITALY_BBOX = (6.5, 35.2, 18.6, 47.15)


# --- pure helpers (no I/O) --------------------------------------------------


def utm_epsg_for(lat: float, lon: float) -> str:
    """Return the EPSG code (as 'EPSG:xxxxx') of the local UTM zone.

    Uses `utm.latlon_to_zone_number` (already relied on by
    `io.loader.tiles.PropagatorDataFromTiles`) rather than the plain
    `(lon + 180) // 6 + 1` formula, so this picks the same zone as the
    tile loader even near the Norway/Svalbard irregular zone boundaries.
    """
    zone = utm.latlon_to_zone_number(lat, lon)
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def wgs84_bbox_from_center(
    lat: float, lon: float, radius_km: float
) -> tuple[float, float, float, float, str]:
    """Return (west, south, east, north, utm_epsg) for a square buffer
    around (lat, lon), using the local UTM zone for the metric buffer."""
    utm_epsg = utm_epsg_for(lat, lon)

    to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    to_wgs84 = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)

    x, y = to_utm.transform(lon, lat)
    r = radius_km * 1000.0
    corners = [(x - r, y - r), (x + r, y - r), (x - r, y + r), (x + r, y + r)]
    lons, lats = zip(*(to_wgs84.transform(cx, cy) for cx, cy in corners))
    return min(lons), min(lats), max(lons), max(lats), utm_epsg


def dem_tile_urls(
    west: float, south: float, east: float, north: float
) -> Iterator[tuple[str, str]]:
    """Yield (url, tile_name) for the Copernicus GLO-30 tiles covering
    the given WGS84 bbox (tiles are 1x1 degree)."""
    lat0, lat1 = int(np.floor(south)), int(np.floor(north))
    lon0, lon1 = int(np.floor(west)), int(np.floor(east))
    for lat in range(lat0, lat1 + 1):
        for lon in range(lon0, lon1 + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            name = (
                f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_"
                f"{ew}{abs(lon):03d}_00_DEM"
            )
            yield f"{DEM_BUCKET}/{name}/{name}.tif", name


def worldcover_tile_urls(
    west: float, south: float, east: float, north: float
) -> Iterator[tuple[str, str]]:
    """Yield (url, tile_name) for the ESA WorldCover tiles covering the
    given WGS84 bbox (tiles are 3x3 degree, named by their SW corner
    rounded down to a multiple of 3)."""
    lat0 = int(np.floor(south / 3.0) * 3)
    lat1 = int(np.floor(north / 3.0) * 3)
    lon0 = int(np.floor(west / 3.0) * 3)
    lon1 = int(np.floor(east / 3.0) * 3)
    for lat in range(lat0, lat1 + 1, 3):
        for lon in range(lon0, lon1 + 1, 3):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            name = (
                f"ESA_WorldCover_10m_2021_v200_"
                f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map"
            )
            yield f"{WORLDCOVER_BUCKET}/{name}.tif", name


def build_target_grid(
    lat: float, lon: float, radius_km: float, cellsize: float, utm_epsg: str
) -> tuple[Affine, int, int]:
    """Return (transform, width, height) for a square grid of the given
    cellsize, centered on (lat, lon), covering at least radius_km on each
    side."""
    to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    r = radius_km * 1000.0
    half_cells = int(np.ceil(r / cellsize))
    size = half_cells * 2
    west = cx - half_cells * cellsize
    north = cy + half_cells * cellsize
    transform = rio.Affine(cellsize, 0.0, west, 0.0, -cellsize, north)
    return transform, size, size


def remap_worldcover_to_fuel(
    worldcover: npt.NDArray[np.integer],
    mapping: dict[int, int] = WORLDCOVER_TO_LEGACY_FUEL,
    default_fuel: int = NON_VEGETATED_FUEL,
) -> npt.NDArray[np.int16]:
    """Remap ESA WorldCover class codes to PROPAGATOR legacy fuel codes.
    Cells with no matching class (including WorldCover's 0/nodata) fall
    back to `default_fuel` (non-vegetated)."""
    fuel = np.full(worldcover.shape, default_fuel, dtype="int16")
    for wc_code, fuel_code in mapping.items():
        fuel[worldcover == wc_code] = fuel_code
    return fuel


def latlon_to_rowcol(
    transform: Affine, utm_epsg: str, lat: float, lon: float
) -> tuple[int, int]:
    """Return the (row, col) grid indices of (lat, lon) for a grid with
    the given affine transform, projected onto `utm_epsg`."""
    to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def ignition_cell_fuel(
    fuel: npt.NDArray[np.integer],
    transform: Affine,
    lat: float,
    lon: float,
    utm_epsg: str,
) -> int | None:
    """Return the fuel code at (lat, lon) within `fuel`'s grid, or None if
    the point falls outside the grid."""
    row, col = latlon_to_rowcol(transform, utm_epsg, lat, lon)
    height, width = fuel.shape
    if 0 <= row < height and 0 <= col < width:
        return int(fuel[row, col])
    return None


# --- I/O helpers -------------------------------------------------------------


def download(url: str, dest: Path) -> Path:
    """Download `url` to `dest`, skipping if already cached there."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def download_italy_tiles(
    cache_dir: Path | None = None,
    bbox: tuple[float, float, float, float] = ITALY_BBOX,
    progress_cb: Callable[[str, str, int], None] | None = None,
) -> dict[str, int]:
    """Pre-download every DEM/WorldCover tile covering `bbox` (Italy by
    default) into `cache_dir`, so that later `prepare_area_data` calls
    for any point inside it are served entirely from the local cache
    (`download()` skips tiles that already exist there) with no network
    wait during a simulation run.

    Tiles with no coverage (e.g. open sea, no Copernicus DSM tile) are
    skipped, same as in `prepare_area_data`. Safe to interrupt and
    re-run: already-downloaded tiles are not re-fetched.

    `progress_cb`, if given, is called as `progress_cb(kind, name,
    size_bytes)` after each tile attempt, with `kind` one of "dem_ok",
    "dem_skip", "wc_ok", "wc_skip" and `size_bytes` 0 for skips.
    """
    cache_dir = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home() / ".propagator" / "cache"
    )
    west, south, east, north = bbox

    summary = {
        "n_dem_downloaded": 0,
        "n_dem_missing": 0,
        "n_worldcover_downloaded": 0,
        "n_worldcover_missing": 0,
        "total_bytes": 0,
    }

    for url, name in dem_tile_urls(west, south, east, north):
        dest = cache_dir / "dem" / f"{name}.tif"
        try:
            download(url, dest)
        except requests.HTTPError:
            summary["n_dem_missing"] += 1
            if progress_cb:
                progress_cb("dem_skip", name, 0)
            continue
        size = dest.stat().st_size
        summary["n_dem_downloaded"] += 1
        summary["total_bytes"] += size
        if progress_cb:
            progress_cb("dem_ok", name, size)

    for url, name in worldcover_tile_urls(west, south, east, north):
        dest = cache_dir / "worldcover" / f"{name}.tif"
        try:
            download(url, dest)
        except requests.HTTPError:
            summary["n_worldcover_missing"] += 1
            if progress_cb:
                progress_cb("wc_skip", name, 0)
            continue
        size = dest.stat().st_size
        summary["n_worldcover_downloaded"] += 1
        summary["total_bytes"] += size
        if progress_cb:
            progress_cb("wc_ok", name, size)

    return summary


def reproject_sources_to_grid(
    source_paths: list[Path],
    dst_path: Path,
    dst_crs: str,
    dst_transform: Affine,
    width: int,
    height: int,
    dtype: str,
    resampling: Resampling,
    nodata: float,
) -> None:
    """Reproject and mosaic `source_paths` onto one GeoTIFF at `dst_path`,
    on the given grid. Later sources do not overwrite already-filled
    cells from earlier ones."""
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(dst_path, "w", **profile) as dst:
        dst_arr = np.full((height, width), nodata, dtype=dtype)
        for src_path in source_paths:
            with rio.open(src_path) as src:
                piece = np.full((height, width), nodata, dtype=dtype)
                rio_reproject(
                    source=rio.band(src, 1),
                    destination=piece,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=nodata,
                    resampling=resampling,
                )
                mask = piece != nodata
                dst_arr[mask] = piece[mask]
        dst.write(dst_arr, 1)


class AreaDataError(Exception):
    """Raised when no source tiles cover the requested area."""


@dataclass
class AreaDataResult:
    dem_path: Path
    fuel_path: Path
    geo_info: GeographicInfo
    utm_epsg: str
    ignition_fuel_code: int | None
    ignition_warning: str | None


def prepare_area_data(
    lat: float,
    lon: float,
    radius_km: float,
    *,
    cellsize: float = 30.0,
    ignition_lat: float | None = None,
    ignition_lon: float | None = None,
    output_dir: Path,
    cache_dir: Path | None = None,
    fuel_mapping: dict[int, int] = WORLDCOVER_TO_LEGACY_FUEL,
) -> AreaDataResult:
    """Download and build `dem.tif`/`fuel.tif` for a square area of
    `radius_km` around (lat, lon), written into `output_dir`.

    If an ignition point is given, checks whether it falls on a burnable
    fuel class and reports it via `AreaDataResult.ignition_warning` (fuel
    class 3, non-vegetated, does not burn) rather than raising, since the
    caller (CLI or web backend) decides how to surface it.
    """
    output_dir = Path(output_dir)
    cache_dir = (
        Path(cache_dir) if cache_dir is not None else output_dir / "_cache"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north, utm_epsg = wgs84_bbox_from_center(
        lat, lon, radius_km
    )

    dem_tiles = []
    for url, name in dem_tile_urls(west, south, east, north):
        try:
            dem_tiles.append(download(url, cache_dir / "dem" / f"{name}.tif"))
        except requests.HTTPError:
            continue

    wc_tiles = []
    for url, name in worldcover_tile_urls(west, south, east, north):
        try:
            wc_tiles.append(
                download(url, cache_dir / "worldcover" / f"{name}.tif")
            )
        except requests.HTTPError:
            continue

    if not dem_tiles or not wc_tiles:
        raise AreaDataError("No source tiles found for this area.")

    transform, width, height = build_target_grid(
        lat, lon, radius_km, cellsize, utm_epsg
    )

    dem_path = output_dir / "dem.tif"
    fuel_path = output_dir / "fuel.tif"

    reproject_sources_to_grid(
        dem_tiles,
        dem_path,
        utm_epsg,
        transform,
        width,
        height,
        dtype="float32",
        resampling=Resampling.bilinear,
        nodata=-9999.0,
    )

    worldcover_raw = output_dir / "_worldcover_raw.tif"
    reproject_sources_to_grid(
        wc_tiles,
        worldcover_raw,
        utm_epsg,
        transform,
        width,
        height,
        dtype="uint8",
        resampling=Resampling.nearest,
        nodata=0,
    )
    with rio.open(worldcover_raw) as src:
        worldcover = src.read(1)
        profile = src.profile
    fuel = remap_worldcover_to_fuel(worldcover, mapping=fuel_mapping)
    profile.update(dtype="int16", nodata=None)
    with rio.open(fuel_path, "w", **profile) as dst:
        dst.write(fuel, 1)
    worldcover_raw.unlink()

    ignition_fuel_code = None
    ignition_warning = None
    if ignition_lat is not None and ignition_lon is not None:
        ignition_fuel_code = ignition_cell_fuel(
            fuel, transform, ignition_lat, ignition_lon, utm_epsg
        )
        if ignition_fuel_code is None:
            ignition_warning = (
                f"Ignition point ({ignition_lat}, {ignition_lon}) falls "
                "outside the generated grid."
            )
        elif ignition_fuel_code == NON_VEGETATED_FUEL:
            ignition_warning = (
                f"Ignition point ({ignition_lat}, {ignition_lon}) falls on "
                "fuel class 3 (non-vegetated), which does not burn. Pick a "
                "vegetated ignition point or the simulation will "
                "self-extinguish immediately."
            )

    with rio.open(dem_path) as f:
        geo_info = GeographicInfo.from_file(f)

    return AreaDataResult(
        dem_path=dem_path,
        fuel_path=fuel_path,
        geo_info=geo_info,
        utm_epsg=utm_epsg,
        ignition_fuel_code=ignition_fuel_code,
        ignition_warning=ignition_warning,
    )
