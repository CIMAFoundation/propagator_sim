from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
from pyproj import CRS
from rasterio.features import shapes  # type: ignore
from rasterio.transform import Affine  # type: ignore
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter1d
from scipy.signal import medfilt2d
from shapely.geometry import LineString, MultiLineString, shape

from propagator.core.models import PropagatorOutput
from propagator.io.geo import GeographicInfo, reproject
from propagator.io.writer.protocol import IsochronesWriterProtocol

TIME_TAG = "time"


def smooth_linestring(linestring, smooth_sigma):
    """
    Uses a gauss filter to smooth out the LineString coordinates.
    """
    smooth_x = np.array(gaussian_filter1d(linestring.xy[0], smooth_sigma))  # type: ignore # gaussian_filter1d has None typing in library
    smooth_y = np.array(gaussian_filter1d(linestring.xy[1], smooth_sigma))  # type: ignore

    # close the linestring
    smooth_y[-1] = smooth_y[0]
    smooth_x[-1] = smooth_x[0]

    smoothed_coords = np.hstack((smooth_x, smooth_y))
    smoothed_coords = zip(smooth_x, smooth_y)

    linestring_smoothed = LineString(smoothed_coords)

    return linestring_smoothed


def extract_isochrone(
    values: npt.NDArray[np.floating],
    transf: Affine,
    thresholds=[0.5, 0.75, 0.9],
    med_filt_val=9,
    min_length=0.0001,
    smooth_sigma=0.8,
    simp_fact=0.00001,
) -> dict[float, MultiLineString]:
    """
    extract isochrone from the propagation probability map values at the probanilities thresholds,
     applying filtering to smooth out the result
    :param values:
    :param transf:
    :param thresholds:
    :param med_filt_val:
    :param min_length:
    :param smooth_sigma:
    :param simp_fact:
    :return:
    """

    # if the dimension of the burned area is low, we do not filter it
    if np.sum(values > 0) <= 100:
        filt_values = values
    else:
        filt_values = medfilt2d(values, med_filt_val)  # type: ignore # medfilt2d has None typing in library
    results = {}

    for t in thresholds:
        over_t_ = (filt_values >= t).astype("uint8")
        over_t = binary_dilation(
            binary_erosion(over_t_).astype("uint8")  # type: ignore #binary_erosion has None typing in library
        ).astype(  # type: ignore #binary_erosion has None typing in library
            "uint8"
        )
        if np.any(over_t):
            for s, v in shapes(over_t, transform=transf):
                sh = shape(s)

                ml = [
                    smooth_linestring(
                        interior_line, smooth_sigma
                    )  # .simplify(simp_fact)
                    for interior_line in sh.interiors  # type: ignore # sh.interiors is missing in typing
                    if interior_line.length > min_length
                ]

                results[t] = MultiLineString(ml)

    return results


@dataclass
class IsochronesGeoJSONWriter(IsochronesWriterProtocol):
    start_date: datetime
    output_folder: Path
    prefix: str
    geo_info: GeographicInfo
    dst_crs: CRS

    thresholds: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    med_filt_val: int = 9
    min_length: float = 0.0001
    smooth_sigma: float = 0.8
    simp_fact: float = 0.00001

    _isochrones: gpd.GeoDataFrame = field(init=False)

    def __post_init__(self):
        self.dst_crs = CRS.from_wkt(self.dst_crs.to_wkt())
        self._isochrones = gpd.GeoDataFrame(
            crs=self.dst_crs,
            columns=["geometry", "date"],
            geometry="geometry",
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["threshold", "time"]
            ),
        )

    def write_isochrones(self, output: PropagatorOutput) -> None:
        json_file = self.output_folder / f"{self.prefix}_{output.time}.json"
        ref_date = self.ref_date(output)

        values = output.fire_probability
        dst_trans = self.geo_info.trans
        crs = self.geo_info.crs
        if crs != self.dst_crs:
            values, dst_trans = reproject(
                values,
                self.geo_info.trans,
                self.geo_info.crs,
                self.dst_crs,
            )

        isochrones_geoms = extract_isochrone(
            values,
            dst_trans,
            thresholds=self.thresholds,
            med_filt_val=self.med_filt_val,
            min_length=self.min_length,
            smooth_sigma=self.smooth_sigma,
            simp_fact=self.simp_fact,
        )

        # iterate over threshold/geometry and add it to the _isochrones
        for threshold, geom in isochrones_geoms.items():
            self._isochrones = gpd.GeoDataFrame(
                pd.concat(
                    [
                        self._isochrones,
                        pd.DataFrame(
                            {
                                "geometry": geom,
                                "date": ref_date.isoformat(),
                            },
                            index=pd.MultiIndex.from_tuples(
                                [(threshold, output.time)],
                                names=["threshold", "time"],
                            ),
                        ),
                    ]
                ),
                geometry="geometry",
                crs=self.dst_crs,
            )

        self._isochrones.to_file(json_file, driver="GeoJSON")


IsochronesMode = Literal["single", "multiple", "jsonl"]
IsochronesFormat = Literal["geojson", "gpkg"]

_FORMAT_EXT_DRIVER = {
    "geojson": (".geojson", "GeoJSON"),
    "gpkg": (".gpkg", "GPKG"),
}


@dataclass
class IsochronesWriter(IsochronesWriterProtocol):
    """Isochrones writer with selectable output mode and file format.

    Modes
    -----
    ``single``
        One consolidated file (``<prefix><ext>``), rewritten each step with
        every isochrone accumulated so far.
    ``multiple``
        One file per timestep (``<prefix>_<time><ext>``) holding only that
        step's isochrones.
    ``jsonl``
        One GeoJSON ``FeatureCollection`` per line appended to
        ``<prefix>.geojsonl`` (always GeoJSON, regardless of ``fmt``).

    The writer is persistent across domain growth: update :attr:`geo_info`
    in place rather than rebuilding it, so accumulated state survives.
    """

    start_date: datetime
    output_folder: Path
    prefix: str
    geo_info: GeographicInfo
    dst_crs: CRS
    mode: IsochronesMode = "multiple"
    fmt: IsochronesFormat = "geojson"

    thresholds: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    med_filt_val: int = 9
    min_length: float = 0.0001
    smooth_sigma: float = 0.8
    simp_fact: float = 0.00001

    _accum: gpd.GeoDataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self.dst_crs = CRS.from_wkt(self.dst_crs.to_wkt())
        self._accum = self._empty_gdf()

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "threshold": pd.Series([], dtype=float),
                "time": pd.Series([], dtype="int64"),
                "date": pd.Series([], dtype="object"),
            },
            geometry=gpd.GeoSeries([], crs=self.dst_crs),
            crs=self.dst_crs,
        )

    def _step_gdf(self, output: PropagatorOutput) -> gpd.GeoDataFrame:
        ref_date = self.ref_date(output)
        values = output.fire_probability
        dst_trans = self.geo_info.trans
        if self.geo_info.crs != self.dst_crs:
            values, dst_trans = reproject(
                values, self.geo_info.trans, self.geo_info.crs, self.dst_crs
            )
        geoms = extract_isochrone(
            values,
            dst_trans,
            thresholds=self.thresholds,
            med_filt_val=self.med_filt_val,
            min_length=self.min_length,
            smooth_sigma=self.smooth_sigma,
            simp_fact=self.simp_fact,
        )
        rows = [
            {
                "threshold": float(threshold),
                "time": int(output.time),
                "date": ref_date.isoformat(),
                "geometry": geom,
            }
            for threshold, geom in geoms.items()
        ]
        if not rows:
            return self._empty_gdf()
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=self.dst_crs)

    def write_isochrones(self, output: PropagatorOutput) -> None:
        gdf = self._step_gdf(output)

        if self.mode == "jsonl":
            if len(gdf) == 0:
                return
            path = self.output_folder / f"{self.prefix}.geojsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(gdf.to_json() + "\n")
            return

        ext, driver = _FORMAT_EXT_DRIVER[self.fmt]

        if self.mode == "multiple":
            if len(gdf) == 0:
                return
            path = self.output_folder / f"{self.prefix}_{output.time}{ext}"
            gdf.to_file(path, driver=driver)
            return

        # single: accumulate then rewrite the consolidated file
        if len(gdf):
            self._accum = gpd.GeoDataFrame(
                pd.concat([self._accum, gdf], ignore_index=True),
                geometry="geometry",
                crs=self.dst_crs,
            )
        if len(self._accum) == 0:
            return
        path = self.output_folder / f"{self.prefix}{ext}"
        self._accum.to_file(path, driver=driver)


def build_isochrones_writer(
    mode: Literal["none", "single", "multiple", "jsonl"],
    *,
    start_date: datetime,
    output_folder: Path,
    prefix: str,
    geo_info: GeographicInfo,
    dst_crs: CRS,
    fmt: IsochronesFormat = "geojson",
    thresholds: Optional[list[float]] = None,
) -> Optional[IsochronesWriter]:
    """Construct an :class:`IsochronesWriter` for ``mode`` (``None`` when
    ``mode == 'none'``)."""
    if mode == "none":
        return None
    return IsochronesWriter(
        start_date=start_date,
        output_folder=output_folder,
        prefix=prefix,
        geo_info=geo_info,
        dst_crs=dst_crs,
        mode=mode,
        fmt=fmt,
        thresholds=thresholds if thresholds is not None else [0.5, 0.75, 0.9],
    )
