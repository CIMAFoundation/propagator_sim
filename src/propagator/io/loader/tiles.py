import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import rasterio as rio  # type: ignore[import]
import scipy
import utm  # type: ignore[import]

from propagator.core.models import PropagatorError
from propagator.io.geo import GeographicInfo

from .protocol import (
    PropagatorDataLoaderException,
    PropagatorInputDataProtocol,
)

DEFAULT_TILES_TAG = "default"

logger = logging.getLogger(__name__)


class NoTilesError(PropagatorError):
    def __init__(self):
        self.message = (
            """Can't initialize simulation, no data on the selected area"""
        )
        super().__init__(self.message)


@dataclass(frozen=True)
class TileReference:
    step_x: int
    step_y: int
    max_y: float
    min_x: float
    tile_dim: int


@dataclass(frozen=True)
class AxisCoverage:
    tile_min: int
    tile_max: int
    idx_min: int
    idx_max: int

    @property
    def tile_range(self) -> range:
        return range(self.tile_min, self.tile_max + 1)

    @property
    def tile_count(self) -> int:
        return self.tile_max - self.tile_min + 1

    def slice_bounds(self, tile_dim: int) -> tuple[int, int]:
        count = self.tile_count
        if count <= 0:
            raise NoTilesError()
        end = self.idx_max + tile_dim * (count - 1)
        return self.idx_min, end


@dataclass
class PropagatorDataFromTiles(PropagatorInputDataProtocol):
    base_path: str
    mid_lat: float
    mid_lon: float
    grid_dim: int

    tileset: str = field(default=DEFAULT_TILES_TAG)
    step_x: float = field(init=False)
    step_y: float = field(init=False)

    zone_number: int = field(init=False)
    easting: float = field(init=False)
    northing: float = field(init=False)

    def __post_init__(self):
        self.easting, self.northing, self.zone_number, _ = utm.from_latlon(
            self.mid_lat, self.mid_lon
        )
        ref = self.load_tile_ref(self.zone_number, "quo", self.tileset)
        self.step_x = ref.step_x
        self.step_y = ref.step_y

    def get_dem(self) -> np.ndarray:
        return np.ascontiguousarray(
            self.load_tiles(
                self.zone_number,
                self.easting,
                self.northing,
                self.grid_dim,
                "quo",
                self.tileset,
            ),
            dtype=np.float64,
        )

    def get_veg(self) -> npt.NDArray[np.int_]:
        return np.ascontiguousarray(
            self.load_tiles(
                self.zone_number,
                self.easting,
                self.northing,
                self.grid_dim,
                "prop",
                self.tileset,
            ),
            dtype=np.int_,
        )

    def get_geo_info(self) -> GeographicInfo:
        rows = self.grid_dim
        cols = self.grid_dim
        north = self.northing + ((rows / 2) * self.step_y)
        east = self.easting + ((cols / 2) * self.step_x)
        south = self.northing - ((rows / 2) * self.step_y)
        west = self.easting - ((cols / 2) * self.step_x)

        geo_info = GeographicInfo.from_bounds(
            west, south, east, north, rows, cols, self.zone_number
        )

        return geo_info

    def load_tile(
        self,
        zone_number: int,
        var: str,
        tile_i: int,
        tile_j: int,
        tileset: str = DEFAULT_TILES_TAG,
    ) -> npt.NDArray[np.floating]:
        """
        Load a tile from the data directory, either as a .mat or .tif file.
        :param zone_number: UTM zone number
        :param var: Variable name (e.g., "quo" or "prop")
        :param tile_i: Tile index in the i direction
        :param tile_j: Tile index in the j direction
        :param tileset: Tileset name (default is "tiles")
        """
        stem = f"{var}_{tile_j}_{tile_i}"
        zone_dir = self._zone_dir(zone_number, tileset)
        loaders = (
            (zone_dir / f"{stem}.tif", self._read_tif_tile),
            (zone_dir / f"{stem}.mat", self._read_mat_tile),
        )

        for path, loader in loaders:
            try:
                data = loader(path)
            except FileNotFoundError:
                continue
            except PropagatorDataLoaderException:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise PropagatorDataLoaderException(
                    f"Failed to load tile '{stem}' from {path}"
                ) from exc

            logger.debug("Loaded tile data from %s", path)
            return np.ascontiguousarray(data)

        raise PropagatorDataLoaderException(
            f"Missing tile '{stem}' in {zone_dir}"
        )

    def load_tile_ref(
        self, zone_number: int, var: str, tileset: str = DEFAULT_TILES_TAG
    ) -> TileReference:
        """
        Load the reference file for the zone, which contains metadata such as
        step size and tile dimensions.
        :param zone_number: UTM zone number
        :param var: Variable name (e.g., "quo" or "prop")
        :param tileset: Tileset name
        """
        path = self._zone_dir(zone_number, tileset) / f"{var}_ref.mat"
        try:
            mat_file = scipy.io.loadmat(path)
        except FileNotFoundError as exc:
            raise PropagatorDataLoaderException(
                f"Missing reference file '{path.name}' in {path.parent}"
            ) from exc

        def extract_scalar(key: str) -> float:
            if key not in mat_file:
                raise PropagatorDataLoaderException(
                    f"Reference file '{path.name}' missing '{key}'"
                )
            try:
                return float(np.asarray(mat_file[key]).item())
            except (TypeError, ValueError) as exc:
                raise PropagatorDataLoaderException(
                    f"Invalid value for '{key}' in reference file '{path.name}'"
                ) from exc

        step_x = int(extract_scalar("stepx"))
        step_y = int(extract_scalar("stepy"))
        max_y = float(extract_scalar("maxy"))
        min_x = float(extract_scalar("minx"))
        tile_dim = int(extract_scalar("tileDim"))

        if tile_dim <= 0:
            raise PropagatorDataLoaderException(
                f"Reference file '{path.name}' has non-positive tile dimension"
            )

        return TileReference(step_x, step_y, max_y, min_x, tile_dim)

    def load_tiles(
        self,
        zone_number: int,
        x: float,
        y: float,
        dim: int,
        var: str,
        tileset: str = DEFAULT_TILES_TAG,
    ) -> npt.NDArray[np.floating]:
        ref = self.load_tile_ref(zone_number, var, tileset)
        i_center = 1 + math.floor((ref.max_y - y) / ref.step_y)
        j_center = 1 + math.floor((x - ref.min_x) / ref.step_x)
        half_dim = math.ceil(dim / 2)

        row_coverage = self._compute_axis_coverage(
            i_center,
            half_dim,
            ref.tile_dim,
        )
        col_coverage = self._compute_axis_coverage(
            j_center,
            half_dim,
            ref.tile_dim,
        )

        tile_rows = []
        for tile_i in row_coverage.tile_range:
            tiles_in_row = [
                self.load_tile(zone_number, var, tile_i, tile_j, tileset)
                for tile_j in col_coverage.tile_range
            ]
            tile_rows.append(self._stack_blocks(tiles_in_row, axis=1))

        mosaic = self._stack_blocks(tile_rows, axis=0)
        row_start, row_end = row_coverage.slice_bounds(ref.tile_dim)
        col_start, col_end = col_coverage.slice_bounds(ref.tile_dim)

        return np.ascontiguousarray(
            mosaic[row_start:row_end, col_start:col_end]
        )

    def _zone_dir(self, zone_number: int, tileset: str) -> Path:
        return Path(self.base_path) / tileset / str(zone_number)

    @staticmethod
    def _compute_axis_coverage(
        center: int,
        half_dim: int,
        tile_dim: int,
    ) -> AxisCoverage:
        start = center - half_dim
        end = center + half_dim

        tile_min = 1 + math.floor(start / tile_dim)
        tile_max = 1 + math.floor(end / tile_dim)
        idx_min = int(start % tile_dim)
        idx_max = int(end % tile_dim)

        coverage = AxisCoverage(tile_min, tile_max, idx_min, idx_max)
        if coverage.tile_count <= 0:
            raise NoTilesError()

        return coverage

    @staticmethod
    def _stack_blocks(blocks: Sequence[np.ndarray], axis: int) -> np.ndarray:
        if not blocks:
            raise NoTilesError()
        if len(blocks) == 1:
            return blocks[0]
        return np.concatenate(blocks, axis=axis)

    @staticmethod
    def _read_mat_tile(path: Path) -> np.ndarray:
        try:
            mat_file = scipy.io.loadmat(path)
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise PropagatorDataLoaderException(
                f"Unable to open MAT tile '{path}'"
            ) from exc

        try:
            return np.asarray(mat_file["M"])
        except KeyError as exc:
            raise PropagatorDataLoaderException(
                f"Tile file '{path}' missing 'M' dataset"
            ) from exc

    @staticmethod
    def _read_tif_tile(path: Path) -> np.ndarray:
        try:
            with rio.open(path) as src:
                return src.read(1)
        except FileNotFoundError:
            raise
        except rio.errors.RasterioIOError as exc:
            raise PropagatorDataLoaderException(
                f"Unable to open raster tile '{path}'"
            ) from exc
