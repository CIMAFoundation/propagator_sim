"""Turn a job's native-UTM-grid frame data into web-friendly output: a
WGS84 bounding rectangle (for `L.imageOverlay`), a colorized RGBA PNG of
`fire_probability`, and WGS84 isochrone line coordinates (reusing
`propagator.io.writer.isochrones_geojson.extract_isochrone` rather than
reimplementing contour extraction).

The bounding rectangle is computed once per job from the fixed native
grid (see `runner.py`) and reused for every frame/time step, so the
overlay never jitters while scrubbing the time slider. For a
radius-limited (<=50 km) UTM grid the four corners projected to WGS84
form a near-rectangle; we take their min/max as an axis-aligned bounds
box, which is the standard, small-distortion approximation `L.imageOverlay`
expects (a true reprojection would warp the raster itself, which is not
worth the extra cost at this scale).
"""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
from PIL import Image
from pyproj import Transformer

from propagator.io.geo import GeographicInfo
from propagator.io.geometry import reproject_geometry
from propagator.io.writer.isochrones_geojson import extract_isochrone

# ember -> fire -> deep red, matching the report style shown to the user
_FIRE_STOPS = [
    (0.0, (0xF2, 0xA9, 0x3C)),
    (0.45, (0xE8, 0x5D, 0x2B)),
    (1.0, (0x7A, 0x1E, 0x0F)),
]


def _fire_colormap() -> npt.NDArray[np.uint8]:
    """256-entry RGB lookup table interpolated across `_FIRE_STOPS`."""
    xs = np.array([s[0] for s in _FIRE_STOPS])
    channels = np.array([s[1] for s in _FIRE_STOPS], dtype=np.float64)
    positions = np.linspace(0.0, 1.0, 256)
    lut = np.stack(
        [np.interp(positions, xs, channels[:, c]) for c in range(3)],
        axis=1,
    )
    return lut.astype(np.uint8)


_LUT = _fire_colormap()


def bounds_wgs84(geo_info: GeographicInfo) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS84 for a UTM `geo_info`."""
    west, south, east, north = geo_info.bounds
    to_wgs84 = Transformer.from_crs(geo_info.crs, "EPSG:4326", always_xy=True)
    corners = [
        (west, south),
        (east, south),
        (west, north),
        (east, north),
    ]
    lons, lats = zip(*(to_wgs84.transform(x, y) for x, y in corners))
    return min(lons), min(lats), max(lons), max(lats)


def fire_probability_png(values: npt.NDArray[np.floating]) -> bytes:
    """Render `fire_probability` (values in [0, 1]) as an RGBA PNG, fully
    transparent where the probability is ~0."""
    clipped = np.clip(values, 0.0, 1.0)
    idx = (clipped * 255).astype(np.uint8)
    rgb = _LUT[idx]
    alpha = (clipped > 0.02).astype(np.uint8) * np.clip(
        (clipped * 255 * 1.3), 0, 235
    ).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    image = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def isochrones_wgs84(
    values: npt.NDArray[np.floating],
    geo_info: GeographicInfo,
    thresholds: list[float],
) -> list[tuple[float, list]]:
    """Return [(threshold, MultiLineString.coords-as-nested-lists)] in
    WGS84, for the given probability field."""
    raw = extract_isochrone(values, geo_info.trans, thresholds=thresholds)
    out = []
    for threshold, multiline in raw.items():
        reprojected = reproject_geometry(multiline, geo_info.crs, "EPSG:4326")
        coords = [list(line.coords) for line in reprojected.geoms]
        out.append((threshold, coords))
    return out
