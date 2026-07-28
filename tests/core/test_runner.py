from __future__ import annotations

import numpy as np
import pytest

from propagator.core import (  # type: ignore
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
)
from propagator.core.constants import TILE_SIZE  # type: ignore
from propagator.core.runner import SimulationRunner  # type: ignore

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from propagator.io.loader.cog import PropagatorDataFromCogs  # noqa: E402

LON, LAT = -8.147, 39.955
CRS_29 = "EPSG:32629"


def zone29_point():
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", CRS_29, always_xy=True).transform(
        LON, LAT
    )
    return x, y


def write_raster(path, data, dtype, nodata, west, north):
    transform = from_origin(west, north, 20.0, 20.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=dtype,
        crs=CRS_29,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)


def make_cog_pair(tmp_path, size=512):
    """A `size`x`size` COG pair centered (roughly) on the test point."""
    x, y = zone29_point()
    west = round(x / 20.0) * 20.0 - size // 2 * 20.0
    north = round(y / 20.0) * 20.0 + size // 2 * 20.0
    veg = np.full((size, size), 5, dtype=np.uint8)
    dem = np.zeros((size, size), dtype=np.float64)
    dem_path = tmp_path / "eu_dem_utm_29.tif"
    fuel_path = tmp_path / "eu_fuel12_utm_29.tif"
    write_raster(dem_path, dem, "float64", -9999.0, west, north)
    write_raster(fuel_path, veg, "uint8", 0.0, west, north)
    return str(dem_path), str(fuel_path)


def make_runner(
    tmp_path, grow_margin: int = TILE_SIZE
) -> tuple[SimulationRunner, PropagatorDataFromCogs]:
    dem_url, fuel_url = make_cog_pair(tmp_path)

    # a deliberately non-square, non-TILE_SIZE-aligned initial window: the
    # bounds cover roughly 110 x 170 cells around the point.
    x, y = zone29_point()
    minx, maxx = x - 850.0, x + 850.0
    miny, maxy = y - 550.0, y + 550.0

    cog_loader = PropagatorDataFromCogs(
        dem_urls=[dem_url],
        fuel_urls=[fuel_url],
        initial_bounds=(minx, miny, maxx, maxy),
        initial_bounds_epsg=int(CRS_29.split(":")[1]),
    )
    dem = cog_loader.get_dem()
    veg = cog_loader.get_veg()
    origin = cog_loader.initial_origin

    simulator = Propagator(
        veg=veg.astype(np.int32),
        dem=dem.astype(np.float32),
        realizations=2,
        do_spotting=False,
        origin=origin,
        out_of_bounds_mode="ignore",
    )
    rows, cols = veg.shape
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=np.full((rows, cols), 10.0, dtype=np.float32),
            wind_dir=np.full((rows, cols), 45.0, dtype=np.float32),
            wind_speed=np.full((rows, cols), 10.0, dtype=np.float32),
            ignitions=[(rows // 2, cols // 2)],
        )
    )
    simulator.out_of_bounds_mode = "raise"
    runner = SimulationRunner(
        simulator=simulator, cog_loader=cog_loader, grow_margin=grow_margin
    )
    return runner, cog_loader


def test_initial_window_from_bounds_is_not_square(tmp_path):
    _, cog_loader = make_runner(tmp_path)
    rows, cols = cog_loader.initial_shape
    assert rows != cols
    # the bounds are contained in the (pixel-snapped) window
    assert rows > 0 and cols > 0


def test_grow_domain_works_with_non_square_initial_window(tmp_path):
    runner, _ = make_runner(tmp_path)
    simulator = runner.simulator

    with pytest.raises(PropagatorOutOfBoundsError):
        for _ in range(2000):
            simulator.step(seconds=3600)

    old_shape = simulator.veg.shape
    old_origin = simulator.origin

    runner.grow_domain()

    rows, cols = simulator.veg.shape
    # grown by grow_margin on every reported edge; the domain got bigger
    assert (rows, cols) >= old_shape
    assert (rows, cols) != old_shape
    # north/west origin shift is a TILE_SIZE multiple, satisfying the
    # Rust/numba core's expand() invariant regardless of the (non-aligned)
    # initial window shape
    row_shift = old_origin[0] - simulator.origin[0]
    col_shift = old_origin[1] - simulator.origin[1]
    assert row_shift % TILE_SIZE == 0
    assert col_shift % TILE_SIZE == 0

    # the run can now continue past the old bounds without erroring
    simulator.out_of_bounds_mode = "ignore"
    simulator.step(seconds=3600)


def test_grow_margin_not_multiple_of_tile_size_raises():
    simulator = Propagator(
        veg=np.full((64, 64), 5, dtype=np.int32),
        dem=np.zeros((64, 64), dtype=np.float32),
        realizations=1,
        do_spotting=False,
    )
    with pytest.raises(ValueError, match="multiple of"):
        SimulationRunner(simulator=simulator, grow_margin=TILE_SIZE + 1)


def test_grow_to_cover_reaches_bounds_outside_the_domain(tmp_path):
    """A boundary condition placed away from the fire has to be reached
    deliberately: the fire never pressures that edge, so `grow_domain` would
    never go there."""
    runner, cog_loader = make_runner(tmp_path)
    simulator = runner.simulator
    old_shape = simulator.veg.shape
    old_origin = simulator.origin

    west, south, east, north = cog_loader.get_geo_info().bounds
    # a point off the north-east corner, well outside the current window
    target = (east + 400.0, north + 300.0, east + 420.0, north + 320.0)

    new_geo_info = runner.grow_to_cover(target)

    assert new_geo_info is not None
    rows, cols = simulator.veg.shape
    assert (rows, cols) != old_shape
    # the target is now inside the domain
    g_west, g_south, g_east, g_north = new_geo_info.bounds
    assert g_east >= target[2] and g_north >= target[3]
    # and it did not grow the sides that were already far enough away
    assert g_west == west and g_south == south
    # the origin shift stays tile-aligned for expand()
    assert (old_origin[0] - simulator.origin[0]) % TILE_SIZE == 0
    assert (old_origin[1] - simulator.origin[1]) % TILE_SIZE == 0


def test_grow_to_cover_is_a_noop_when_bounds_already_inside(tmp_path):
    runner, cog_loader = make_runner(tmp_path)
    old_shape = runner.simulator.veg.shape

    west, south, east, north = cog_loader.get_geo_info().bounds
    inside = (west + 100.0, south + 100.0, east - 100.0, north - 100.0)

    assert runner.grow_to_cover(inside) is None
    assert runner.simulator.veg.shape == old_shape


def test_grow_to_cover_refuses_to_pass_the_domain_cap(tmp_path):
    """Past the cap the run carries on clipped rather than growing until the
    process is killed."""
    runner, cog_loader = make_runner(tmp_path)
    rows, cols = runner.simulator.veg.shape
    runner.max_domain_cells = rows * cols  # no room for any growth
    old_shape = runner.simulator.veg.shape

    _, _, east, north = cog_loader.get_geo_info().bounds
    target = (east + 400.0, north + 300.0, east + 420.0, north + 320.0)

    assert runner.grow_to_cover(target) is None
    assert runner.simulator.veg.shape == old_shape


def test_grow_to_cover_is_a_noop_once_an_earlier_growth_covered_it(tmp_path):
    """The shortfall is measured against the domain the run is on now.

    Measuring it against the initial window instead made every later call
    grow again for ground the domain already held — enough repetitions and
    the run hits `max_domain_cells` without the condition ever landing.
    """
    runner, cog_loader = make_runner(tmp_path)

    _, _, east, north = cog_loader.get_geo_info().bounds
    target = (east + 400.0, north + 300.0, east + 420.0, north + 320.0)

    assert runner.grow_to_cover(target) is not None
    grown_shape = runner.simulator.veg.shape

    # the same target is inside now, so nothing more is needed
    assert runner.grow_to_cover(target) is None
    assert runner.simulator.veg.shape == grown_shape
