from __future__ import annotations

import json
import re
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio.enums as rio_enums  # type: ignore
import shapely
from pyproj import CRS, Transformer
from rasterio.features import rasterize  # type: ignore
from shapely import (
    Geometry,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import transform

from propagator.io.geo import GeographicInfo

DEFAULT_EPSG_GEOMETRY = 4326  # default to WGS84


# ---- geometry models --------------------------------------------------------
class GeometryKind(str, Enum):
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"


def get_middle_point(ignition: Geometry) -> Optional[Tuple[float, float]]:
    """
    Calculate the barycenter (average of coordinates) of a Shapely geometry.
    Works for any geometry type. Returns None if geometry is None or empty.
    """
    if ignition is None:
        return None

    if isinstance(ignition, Point):
        return (ignition.x, ignition.y)

    elif isinstance(ignition, LineString):
        coords = np.array(ignition.coords)
        return tuple(coords.mean(axis=0))

    elif isinstance(ignition, MultiLineString):
        all_coords = np.concatenate(
            [np.array(line.coords) for line in ignition.geoms]
        )
        return tuple(all_coords.mean(axis=0))

    elif isinstance(ignition, Polygon):
        return ignition.centroid.x, ignition.centroid.y

    elif isinstance(ignition, MultiPolygon):
        centroids = np.array(
            [poly.centroid.coords[0] for poly in ignition.geoms]
        )
        return tuple(centroids.mean(axis=0))
    else:
        return None


def reproject_geometry(
    geom: Geometry, crs_from: str | CRS, crs_to: str | CRS
) -> Geometry:
    """Reproject a Shapely geometry from one CRS to another.

    Parameters
    ----------
    geom : Geometry
        Input geometry (Point, LineString, Polygon, etc.)
    crs_from : str
        Source CRS in any format recognized by pyproj (e.g., "EPSG:4326").
    crs_to : str
        Target CRS in any format recognized by pyproj (e.g., "EPSG:3857").

    Returns
    -------
    Geometry
        Reprojected geometry.
    """

    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    projected_geom = transform(transformer.transform, geom)

    return projected_geom


# ---- parsing ---------------------------------------------------------------


_POINT_RE = re.compile(
    r"""
    ^POINT:\s*
    (?:\[\s*)?                     # optional opening bracket
    (?P<y>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)   # y
    \s*(?:[,;])\s*                 # separator: comma or semicolon
    (?P<x>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)   # x
    \s*(?:\]\s*)?                  # optional closing bracket
    $""",
    re.IGNORECASE | re.VERBOSE,
)


_SERIES_RE = re.compile(
    r"""^(?P<kind>LINE|POLYGON):\[\s*
        (?P<ys>-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+)\s*
        \];\[\s*
        (?P<xs>-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+)\s*
        \]\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _split_floats(s: str) -> List[float]:
    return [float(x) for x in s.strip().split() if x.strip()]


def parse_geometry_string(s: str, epsg: int) -> Geometry:
    """Parse POINT/LINE/POLYGON strings into geometry objects."""
    s = s.strip()
    # crs = CRS.from_epsg(epsg)
    m_pt = _POINT_RE.match(s)
    if m_pt:
        y = float(m_pt.group("y"))
        x = float(m_pt.group("x"))
        point = Point((x, y))
        return point

    m_series = _SERIES_RE.match(s)
    if m_series:
        kind = m_series.group("kind").upper()
        ys_arr = np.asarray(_split_floats(m_series.group("ys")), dtype=float)
        xs_arr = np.asarray(_split_floats(m_series.group("xs")), dtype=float)
        if ys_arr.size != xs_arr.size:
            raise ValueError(
                f"{kind}: y/x counts differ \
                ({ys_arr.size} vs {xs_arr.size})"
            )
        if kind == "LINE":
            line = LineString(list(zip(xs_arr, ys_arr)))
            # return GeoLine(ys=ys_arr, xs=xs_arr, crs=crs)
            return line
        elif kind == "POLYGON":
            # return GeoPolygon(ys=ys_arr, xs=xs_arr, crs=crs)
            polygon = Polygon(list(zip(xs_arr, ys_arr)))
            return polygon
        else:
            raise ValueError(f"Unsupported geometry kind: {kind!r}")
    raise ValueError(f"Unsupported geometry string: {s!r}")


def is_allowed(geometry: Geometry, allowed: set[GeometryKind]) -> bool:
    if isinstance(geometry, Point) and GeometryKind.POINT not in allowed:
        return False
    if isinstance(geometry, LineString) and GeometryKind.LINE not in allowed:
        return False
    if isinstance(geometry, Polygon) and GeometryKind.POLYGON not in allowed:
        return False
    return True


def parse_geometry_list(
    v: list, allowed: set[GeometryKind], epsg: int
) -> List[Geometry]:
    if not isinstance(v, list):
        raise ValueError("expected a list")
    out: List[Geometry] = []
    for item in v:
        if isinstance(item, str):
            g = parse_geometry_string(item, epsg)
        else:
            raise ValueError(f"unsupported entry {item!r}")
        # allowed-kind check
        if not is_allowed(g, allowed):
            raise ValueError(f"Geometry {g} not allowed")
        out.append(g)
    return out


# --- rasterization ---


def rasterize_geometries(
    geometries: Sequence[Geometry],
    geo_info: GeographicInfo,
    fill: int = 0,
    default_value: Union[int, float] = 1,
    values: Optional[Sequence[Union[int, float]]] = None,
    all_touched: bool = True,
    dtype: str = "uint8",
    merge_alg: str = "replace",  # "replace" | "add"
) -> np.ndarray:
    """
    Rasterize a sequence of Geometry objects into a numpy array.

    Parameters
    ----------
    geometries : list of Geometry
        Geometry objects in the same CRS `src_crs`.
    geo_info: GeographicInfo
        Geographic information for the output raster.
    fill : scalar
        Background value.
    default_value : scalar
        Burn value when `values` not provided.
    values : optional sequence
        Per-geometry burn values; if provided, must match `len(geometries)`.
    all_touched : bool
        Pass-through to rasterize(); include all touched pixels if True.
    dtype : numpy dtype string
        Output dtype.
    merge_alg : str
        "replace" (last wins) or "add" (sum overlaps).

    Returns
    -------
    np.ndarray
        Rasterized array of shape `out_shape` and dtype `dtype`.
    """

    if values is not None and len(values) != len(geometries):
        raise ValueError("`values` length must match `geometries` length")

    # Prepare shapes in destination CRS
    shapes: List[Tuple[dict, Union[int, float]]] = []
    for i, g in enumerate(geometries):
        geom = reproject_geometry(g, "epsg:4326", str(geo_info.crs))
        gj = json.loads(shapely.to_geojson(geom))
        val = values[i] if values is not None else default_value
        shapes.append((gj, val))
    if merge_alg not in {"replace", "add"}:
        raise ValueError("merge_alg must be 'replace' or 'add'")

    # Rasterize
    out = rasterize(
        shapes=shapes,
        out_shape=geo_info.shape,
        transform=geo_info.trans,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
        merge_alg=rio_enums.MergeAlg.add
        if merge_alg == "add"
        else rio_enums.MergeAlg.replace,
    )
    return out
