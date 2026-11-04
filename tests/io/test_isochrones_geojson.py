from __future__ import annotations

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import MultiLineString

from propagator.io.writer.isochrones_geojson import extract_isochrone

TRANSFORM = Affine(30.0, 0.0, 0.0, 0.0, -30.0, 2000.0)


def test_extract_isochrone_keeps_all_disjoint_regions():
    """Regression test: the contour lines of every disjoint region above
    the threshold must be kept. The old implementation assigned
    ``results[t]`` inside the ``shapes(...)`` loop, so only the polygons
    found last survived (and ``rasterio.features.shapes`` also yields the
    background, whose interior is the fire boundary — scan order decided
    whether the isochrone was written at all)."""
    values = np.zeros((20, 20))
    values[2:8, 2:8] = 0.9
    values[2:8, 12:18] = 0.9

    results = extract_isochrone(values, TRANSFORM, thresholds=[0.5])

    assert set(results) == {0.5}
    geom = results[0.5]
    assert isinstance(geom, MultiLineString)
    assert len(geom.geoms) == 2


def test_extract_isochrone_empty_when_no_region_above_threshold():
    values = np.zeros((20, 20))
    values[2:8, 2:8] = 0.4

    results = extract_isochrone(values, TRANSFORM, thresholds=[0.5])

    assert results == {}
