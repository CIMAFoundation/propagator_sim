from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from propagator.core.constants import NO_FUEL  # type: ignore # noqa: E402
from propagator.io.loader.cog import (  # type: ignore # noqa: E402
    PropagatorDataFromCogs,
    gdal_path,
    utm_zone_of_lon,
)

# UTM zone 29N covers lon in [-12, -6); a point near Pedrogao Grande
LON, LAT = -8.147, 39.955
CRS_29 = "EPSG:32629"


def zone29_point():
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", CRS_29, always_xy=True).transform(
        LON, LAT
    )
    return x, y


def raster_anchor(size):
    """North-west corner of a size-cell raster centered on the point,
    snapped to the 20 m grid."""
    x, y = zone29_point()
    west = (round(x / 20.0) - size // 2) * 20.0
    north = (round(y / 20.0) + size // 2) * 20.0
    return west, north


def write_raster(
    path, data, dtype, nodata, crs=CRS_29, west=400_000.0, north=4_500_000.0
):
    transform = from_origin(west, north, 20.0, 20.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)


def make_cog_pair(tmp_path, size=256):
    rows = np.arange(size, dtype=np.float64)[:, None]
    dem = np.broadcast_to(rows * 5.0, (size, size)).copy()
    fuel = np.full((size, size), 6, dtype=np.uint8)
    fuel[: size // 4] = 0  # a fuel-free band at the top
    dem_path = tmp_path / "eu_dem_utm_29.tif"
    fuel_path = tmp_path / "eu_fuel12_utm_29.tif"
    west, north = raster_anchor(size)
    write_raster(dem_path, dem, "float64", -9999.0, west=west, north=north)
    write_raster(fuel_path, fuel, "uint8", 0.0, west=west, north=north)
    return str(dem_path), str(fuel_path)


def test_gdal_path_translation():
    assert gdal_path("s3://bucket/key.tif") == "/vsis3/bucket/key.tif"
    assert gdal_path("https://host/key.tif") == "/vsicurl/https://host/key.tif"
    assert gdal_path("/local/file.tif") == "/local/file.tif"


def test_utm_zone_of_lon():
    assert utm_zone_of_lon(LON) == 29
    assert utm_zone_of_lon(9.0) == 32


def test_selects_pair_by_zone_and_reads_window(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    # decoys in another zone that do not contain the point
    other = tmp_path / "other"
    other.mkdir()
    dem_32 = other / "eu_dem_utm_32.tif"
    fuel_32 = other / "eu_fuel12_utm_32.tif"
    write_raster(
        dem_32, np.zeros((16, 16)), "float64", -9999.0, crs="EPSG:32632"
    )
    write_raster(fuel_32, np.zeros((16, 16)), "uint8", 0.0, crs="EPSG:32632")

    # lists pair by utm_<zone> hint even with different coverage
    loader = PropagatorDataFromCogs(
        dem_urls=[str(dem_32), str(dem_29)],
        fuel_urls=[str(fuel_29)],
        mid_lon=LON,
        mid_lat=LAT,
        grid_dim=64,
    )
    dem = loader.get_dem()
    veg = loader.get_veg()
    geo_info = loader.get_geo_info()

    assert dem.shape == veg.shape == (64, 64)
    assert loader.cellsize == 20.0
    assert dem.dtype == np.float32
    # the window is centered on the point: origin + half = point pixel
    row0, col0 = loader.initial_origin
    with rasterio.open(str(dem_29)) as src:
        point_row, point_col = src.index(*zone29_point())
    assert (row0 + 32, col0 + 32) == (point_row, point_col)
    # geo_info transform matches the window position
    west, _ = raster_anchor(256)
    assert geo_info.trans.c == pytest.approx(west + col0 * 20.0)
    assert geo_info.shape == (64, 64)


def test_boundless_window_fills_nodata(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    loader = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        mid_lon=LON,
        mid_lat=LAT,
        grid_dim=64,
    )
    # window hanging off the north-west corner of the raster
    dem, veg, geo_info = loader.load_window((-32, -32), (64, 64))
    assert dem.shape == (64, 64)
    np.testing.assert_array_equal(dem[:32, :], 0.0)
    np.testing.assert_array_equal(veg[:32, :], NO_FUEL)
    np.testing.assert_array_equal(veg[:, :32], NO_FUEL)
    # in-raster quadrant carries real data (fuel-free band is class 0 too,
    # so check the dem gradient instead)
    assert dem[63, 63] == pytest.approx(31 * 5.0)


def test_grown_window_is_consistent_with_initial(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    loader = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        mid_lon=LON,
        mid_lat=LAT,
        grid_dim=64,
    )
    origin = loader.initial_origin
    initial_dem = loader.get_dem()
    # captured before the growth: `get_geo_info` follows the last window
    # served, so afterwards it reports the grown one
    initial_west = loader.get_geo_info().trans.c

    margin = 32
    grown_dem, grown_veg, grown_geo = loader.load_window(
        (origin[0] - margin, origin[1] - margin), (128, 128)
    )
    np.testing.assert_array_equal(
        grown_dem[margin : margin + 64, margin : margin + 64], initial_dem
    )
    assert grown_geo.trans.c == pytest.approx(initial_west - margin * 20.0)


def test_point_outside_all_cogs_raises(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    from propagator.io.loader.protocol import PropagatorDataLoaderException

    with pytest.raises(PropagatorDataLoaderException, match="contains"):
        PropagatorDataFromCogs(
            dem_urls=[dem_29],
            fuel_urls=[fuel_29],
            mid_lon=9.0,  # zone 32, far outside the zone-29 raster
            mid_lat=45.0,
            grid_dim=64,
        )


def test_initial_bounds_derives_pixel_aligned_window(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    x, y = zone29_point()
    # a box not aligned to whole pixels, and deliberately non-square:
    # ~40 cols wide, ~70 rows tall
    minx, maxx = x - 13 * 20.0 - 3.0, x + 27 * 20.0 + 4.0
    miny, maxy = y - 5 * 20.0 - 5.0, y + 65 * 20.0 + 1.0

    loader = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        initial_bounds=(minx, miny, maxx, maxy),
        initial_bounds_epsg=32629,
    )
    # mid_lon/mid_lat were derived from the bounds centre for COG selection
    assert loader.mid_lon is not None and loader.mid_lat is not None

    dem = loader.get_dem()
    veg = loader.get_veg()
    assert dem.shape == veg.shape == loader.initial_shape
    # window fully contains the requested box (snapped outward to pixels)
    west, south, east, north = loader.get_geo_info().bounds
    assert west <= minx and south <= miny
    assert east >= maxx and north >= maxy
    # pixel-aligned: the window is an integer number of 20 m cells wide
    assert (east - west) == pytest.approx(loader.initial_shape[1] * 20.0)
    assert (north - south) == pytest.approx(loader.initial_shape[0] * 20.0)
    # not a square: rows and cols differ (13+27 vs 7+33, before snapping)
    assert loader.initial_shape[0] != loader.initial_shape[1]


def test_initial_bounds_reprojects_from_other_crs(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    from pyproj import Transformer

    x, y = zone29_point()
    minx, maxx = x - 200.0, x + 200.0
    miny, maxy = y - 200.0, y + 200.0
    # same box, expressed in EPSG:4326 instead of the COG's own CRS
    transformer = Transformer.from_crs(CRS_29, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform([minx, maxx], [miny, maxy])

    loader_4326 = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        initial_bounds=(lons[0], lats[0], lons[1], lats[1]),
        initial_bounds_epsg=4326,
    )
    loader_native = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        initial_bounds=(minx, miny, maxx, maxy),
        initial_bounds_epsg=32629,
    )
    # allow a 1-pixel tolerance: the lon/lat round trip perturbs coordinates
    # by sub-millimeter amounts, which can flip the outward-snap of a box
    # edge that happens to sit extremely close to a pixel boundary
    for a, b in zip(loader_4326.initial_origin, loader_native.initial_origin):
        assert abs(a - b) <= 1
    for a, b in zip(loader_4326.initial_shape, loader_native.initial_shape):
        assert abs(a - b) <= 1


def test_initial_bounds_snap_shape_to_tile_size(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    x, y = zone29_point()
    minx, maxx = x - 13 * 20.0, x + 27 * 20.0
    miny, maxy = y - 7 * 20.0, y + 33 * 20.0

    loader = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        initial_bounds=(minx, miny, maxx, maxy),
        initial_bounds_epsg=32629,
        snap_shape_to=32,
    )
    rows, cols = loader.initial_shape
    assert rows % 32 == 0
    assert cols % 32 == 0


def test_initial_pixel_window_used_verbatim(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    loader = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        mid_lon=LON,
        mid_lat=LAT,
        initial_pixel_origin=(10, -5),
        initial_pixel_shape=(40, 96),
    )
    assert loader.initial_origin == (10, -5)
    assert loader.initial_shape == (40, 96)
    dem = loader.get_dem()
    assert dem.shape == (40, 96)


def test_initial_window_arg_validation(tmp_path):
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    from propagator.io.loader.protocol import PropagatorDataLoaderException

    with pytest.raises(
        PropagatorDataLoaderException, match="mutually exclusive"
    ):
        PropagatorDataFromCogs(
            dem_urls=[dem_29],
            fuel_urls=[fuel_29],
            initial_bounds=(0.0, 0.0, 1.0, 1.0),
            initial_pixel_origin=(0, 0),
            initial_pixel_shape=(10, 10),
        )

    with pytest.raises(PropagatorDataLoaderException, match="given together"):
        PropagatorDataFromCogs(
            dem_urls=[dem_29],
            fuel_urls=[fuel_29],
            mid_lon=LON,
            mid_lat=LAT,
            initial_pixel_origin=(0, 0),
        )

    with pytest.raises(PropagatorDataLoaderException, match="mid_lon"):
        PropagatorDataFromCogs(
            dem_urls=[dem_29],
            fuel_urls=[fuel_29],
            initial_pixel_origin=(0, 0),
            initial_pixel_shape=(10, 10),
        )


def test_get_geo_info_follows_the_last_window_served(tmp_path):
    """Growth serves wider windows through `load_window`, and callers that
    place geometry on the grid need the domain the run is on now — not the
    one it started with."""
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    loader = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        mid_lon=LON,
        mid_lat=LAT,
        grid_dim=64,
    )

    initial = loader.get_geo_info()
    assert initial.shape == (64, 64)

    row0, col0 = loader.initial_origin
    _, _, grown = loader.load_window((row0 - 32, col0 - 32), (128, 128))

    current = loader.get_geo_info()
    assert current.shape == (128, 128)
    assert current.bounds == grown.bounds
    # and it grew outwards from the initial window rather than replacing it
    assert current.bounds[0] < initial.bounds[0]
    assert current.bounds[3] > initial.bounds[3]

    # the initial dem/veg are still the initial window's: only the reported
    # geometry follows growth, since the core owns the grown rasters.
    assert loader.get_dem().shape == (64, 64)


def test_window_composed_from_blocks_matches_a_direct_read(tmp_path):
    """Composition is exact, including across block seams and off the raster."""
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    loader = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        mid_lon=LON,
        mid_lat=LAT,
        grid_dim=64,
        block_size=32,
    )
    row0, col0 = loader.initial_origin
    # deliberately unaligned to the block lattice, and reaching outside it
    origin = (row0 - 17, col0 - 5)
    shape = (100, 83)

    dem, veg, _ = loader.load_window(origin, shape)
    expected_dem, expected_veg = loader._read_window(origin, shape)

    np.testing.assert_array_equal(dem, expected_dem)
    np.testing.assert_array_equal(veg, expected_veg)


def test_growth_only_reads_the_new_ring(tmp_path):
    """The point of the block cache: the interior of a grown window is the
    ground the run is already on, and must not be read from the COGs again."""
    dem_29, fuel_29 = make_cog_pair(tmp_path)
    loader = PropagatorDataFromCogs(
        dem_urls=[dem_29],
        fuel_urls=[fuel_29],
        mid_lon=LON,
        mid_lat=LAT,
        grid_dim=64,
        block_size=32,
    )
    row0, col0 = loader.initial_origin

    reads = []
    original = loader._read_window

    def counting_read(origin, shape):
        reads.append(origin)
        return original(origin, shape)

    loader._read_window = counting_read  # type: ignore[method-assign]

    loader.load_window((row0, col0), (64, 64))
    first = len(reads)
    assert first == 4  # a 64-cell window spans 2x2 blocks of 32

    # grow by one block on every side: the original 2x2 is already cached, so
    # only the ring around it is read
    loader.load_window((row0 - 32, col0 - 32), (128, 128))
    ring = len(reads) - first
    assert ring == 4 * 4 - 2 * 2
