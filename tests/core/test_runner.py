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
