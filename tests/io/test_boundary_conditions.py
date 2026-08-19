from __future__ import annotations

import numpy as np
import rasterio as rio
from pyproj import CRS
from shapely import LineString

from propagator.io.actions import CanadairAction, HeavyAction, HelicopterAction
from propagator.io.boundary_conditions import TimedInput
from propagator.io.geo import GeographicInfo

UTM_EPSG = "EPSG:32633"


def make_geo_info(size: int = 30) -> GeographicInfo:
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    return GeographicInfo(
        crs=CRS.from_epsg(32633),
        trans=transform,
        bounds=(500000.0, 4700000.0 - size * 30, 500000.0 + size * 30, 4700000.0),
        shape=(size, size),
    )


def _pixel_center_lonlat(geo_info: GeographicInfo, row: int, col: int):
    transform = geo_info.trans
    x = transform.c + (col + 0.5) * transform.a
    y = transform.f + (row + 0.5) * transform.e
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs(geo_info.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(x, y)
    return lon, lat


def _line_through(geo_info, row, col):
    lon0, lat0 = _pixel_center_lonlat(geo_info, row, col)
    lon1, lat1 = _pixel_center_lonlat(geo_info, row, col + 1)
    return LineString([(lon0, lat0), (lon1, lat1)])


def test_action_only_moisture_action_does_not_reset_whole_grid_moisture():
    """Regression test: a TimedInput carrying only a moisture-affecting
    action (canadair/waterline/helicopter) and no explicit `moisture`
    must leave the absolute moisture field unset (None) so it doesn't
    reset the whole grid's fuel moisture to 0% — the action's effect
    should only show up in `additional_moisture`, localized to its
    buffer area. Previously this incorrectly forced moisture=0
    everywhere, making the fire spread *faster* away from the action."""
    geo_info = make_geo_info()
    action = CanadairAction(geometries=[_line_through(geo_info, 15, 15)])
    ti = TimedInput(time=3600, actions=[action])

    bc = ti.get_boundary_conditions(geo_info, non_vegetated=3)

    assert bc.moisture is None
    assert bc.additional_moisture is not None
    # the action's relief is confined to a small buffer, not the whole grid
    assert np.any(bc.additional_moisture > 0)
    assert not np.all(bc.additional_moisture > 0)


def test_action_with_explicit_moisture_still_sets_absolute_moisture():
    geo_info = make_geo_info()
    action = CanadairAction(geometries=[_line_through(geo_info, 15, 15)])
    ti = TimedInput(time=3600, moisture=10.0, actions=[action])

    bc = ti.get_boundary_conditions(geo_info, non_vegetated=3)

    assert bc.moisture is not None
    assert np.all(bc.moisture == 10.0)
    assert bc.additional_moisture is not None


def test_helicopter_action_actually_affects_moisture():
    """Regression test: `HelicopterAction` used to define a method named
    `rasterize_action` instead of overriding `rasterize_action_moisture`,
    so it silently inherited the base class's no-op (returns None) and
    had zero effect on the simulation despite being a valid, accepted
    action."""
    geo_info = make_geo_info()
    action = HelicopterAction(geometries=[_line_through(geo_info, 15, 15)])
    ti = TimedInput(time=3600, actions=[action])

    bc = ti.get_boundary_conditions(geo_info, non_vegetated=3)

    assert bc.additional_moisture is not None
    assert np.any(bc.additional_moisture > 0)


def test_heavy_action_alone_does_not_touch_moisture():
    geo_info = make_geo_info()
    action = HeavyAction(geometries=[_line_through(geo_info, 15, 15)])
    ti = TimedInput(time=3600, actions=[action])

    bc = ti.get_boundary_conditions(geo_info, non_vegetated=3)

    assert bc.moisture is None
    assert bc.additional_moisture is None
    assert bc.vegetation_changes is not None
