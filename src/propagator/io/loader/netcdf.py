"""Loads a forefire-style landscape NetCDF file (`loadData[...]` in a `.ff`
script) into propagator's DEM/fuel/geo-info triple.

Real forefire landscape files (see `tests/runff/data.nc` upstream) don't
match the plain 2D-array layout their own docs describe: `fuel` and
`altitude` carry leading singleton (time, z) dimensions and are frequently
at *different* resolutions (a finer fuel grid than the DEM/wind grids), and
georeferencing lives in a scalar `domain` variable's attributes
(`BBoxWSEN`, `SWx`/`SWy`/`Lx`/`Ly`) rather than any standard CRS metadata.
A preceding `FireDomain[...;BBoxWSEN=...]` command is only used as a
fallback when the file itself carries no `domain` variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from propagator.io.geo import GeographicInfo
from propagator.io.loader.cog import utm_zone_of_lon
from propagator.io.loader.protocol import (
    PropagatorDataLoaderException,
    PropagatorInputDataProtocol,
)

# candidate variable names, in priority order, per forefire's landscape-file
# convention (see docs/source/user_guide/landscape_file.rst)
_FUEL_NAMES = ("fuel", "fuel_index", "land_cover")
_ALTITUDE_NAMES = ("altitude", "elevation", "dem", "hgt")
_WIND_U_NAMES = ("windU", "wind_u", "U")
_WIND_V_NAMES = ("windV", "wind_v", "V")


def _find_variable(dataset, candidates: tuple[str, ...]) -> Optional[str]:
    for name in candidates:
        if name in dataset.variables:
            return name
    return None


def _as_2d(values: np.ndarray) -> np.ndarray:
    """Drop leading singleton dims (time/z, as in `fuel(ft,fz,fy,fx)` and
    `altitude(nt,nz,ny,nx)`) down to the trailing (row, col) grid."""
    while values.ndim > 2:
        if values.shape[0] != 1:
            raise PropagatorDataLoaderException(
                f"cannot squeeze non-singleton leading dimension out of "
                f"array with shape {values.shape}"
            )
        values = values[0]
    return values


def _resample_to(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample `values` onto `shape` (integer ratios
    only, e.g. a coarser DEM grid resampled up to a finer fuel grid)."""
    if values.shape == shape:
        return values
    ry, rx = shape[0] / values.shape[0], shape[1] / values.shape[1]
    if ry != int(ry) or rx != int(rx):
        raise PropagatorDataLoaderException(
            f"grid shapes {values.shape} and {shape} aren't related by an "
            "integer resampling ratio"
        )
    return np.repeat(np.repeat(values, int(ry), axis=0), int(rx), axis=1)


def _read_domain_bbox(dataset) -> Optional[tuple[float, float, float, float]]:
    if "domain" not in dataset.variables:
        return None
    domain = dataset.variables["domain"]
    bbox_attr = getattr(domain, "BBoxWSEN", None)
    if bbox_attr is None:
        return None
    west, south, east, north = (float(v) for v in str(bbox_attr).split(","))
    return west, south, east, north


@dataclass
class PropagatorDataFromNetCDF(PropagatorInputDataProtocol):
    nc_file: str
    bbox_wsen: Optional[tuple[float, float, float, float]] = None
    """WGS84 (west, south, east, north), from `FireDomain[...;BBoxWSEN=...]`;
    only used when the NetCDF file itself carries no CRS."""

    _dem: npt.NDArray = field(init=False, repr=False)
    _veg: npt.NDArray = field(init=False, repr=False)
    _geo_info: GeographicInfo = field(init=False, repr=False)
    _extra_layers: dict[str, np.ndarray] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        try:
            import netCDF4  # type: ignore
        except ImportError as e:
            raise PropagatorDataLoaderException(
                "Reading NetCDF landscape files requires the 'netCDF4' "
                "package (install the 'propagator[ff]' extra)."
            ) from e

        with netCDF4.Dataset(self.nc_file) as ds:
            fuel_name = _find_variable(ds, _FUEL_NAMES)
            alt_name = _find_variable(ds, _ALTITUDE_NAMES)
            if fuel_name is None or alt_name is None:
                raise PropagatorDataLoaderException(
                    f"{self.nc_file}: could not find fuel/altitude "
                    f"variables (looked for {_FUEL_NAMES} / {_ALTITUDE_NAMES})"
                )

            self._veg = _as_2d(np.asarray(ds.variables[fuel_name][:])).astype(
                "int8"
            )
            dem = _as_2d(np.asarray(ds.variables[alt_name][:])).astype("int16")
            # fuel and altitude/wind grids may be at different resolutions
            self._dem = _resample_to(dem, self._veg.shape)

            file_bbox = _read_domain_bbox(ds)
            if file_bbox is not None:
                self.bbox_wsen = file_bbox

            u_name = _find_variable(ds, _WIND_U_NAMES)
            v_name = _find_variable(ds, _WIND_V_NAMES)
            if u_name is not None:
                self._extra_layers["windU"] = _as_2d(
                    np.asarray(ds.variables[u_name][:])
                )
            if v_name is not None:
                self._extra_layers["windV"] = _as_2d(
                    np.asarray(ds.variables[v_name][:])
                )
            self._extra_layers["altitude"] = self._dem
            self._extra_layers["fuel"] = self._veg

        self._geo_info = self._build_geo_info()

    def _build_geo_info(self) -> GeographicInfo:
        if self.bbox_wsen is None:
            raise PropagatorDataLoaderException(
                f"{self.nc_file}: no CRS in the NetCDF file and no "
                "FireDomain[...;BBoxWSEN=...] bounding box was supplied "
                "beforehand to georeference the grid."
            )
        from pyproj import Proj

        west, south, east, north = self.bbox_wsen
        rows, cols = self._dem.shape
        zone = utm_zone_of_lon((west + east) / 2.0)
        # `GeographicInfo.from_bounds` builds the affine transform straight
        # from the bounds it's given, so they must already be in the UTM
        # zone's projected meters, not the WGS84 degrees `BBoxWSEN` carries.
        proj = Proj(proj="utm", zone=zone, datum="WGS84")
        west_m, south_m = proj(west, south)
        east_m, north_m = proj(east, north)
        return GeographicInfo.from_bounds(
            west=west_m,
            south=south_m,
            east=east_m,
            north=north_m,
            rows=rows,
            cols=cols,
            zone=zone,
        )

    def get_dem(self) -> np.ndarray:
        return self._dem

    def get_veg(self) -> np.ndarray:
        return self._veg

    def get_geo_info(self) -> GeographicInfo:
        return self._geo_info

    def get_layer(self, name: str) -> np.ndarray:
        """Raw landscape layer by name (`altitude`, `fuel`, `windU`,
        `windV`), for the `save[filename=;fields=...]` command."""
        if name not in self._extra_layers:
            raise KeyError(
                f"layer {name!r} not available in {self.nc_file} "
                f"(have: {sorted(self._extra_layers)})"
            )
        return self._extra_layers[name]
