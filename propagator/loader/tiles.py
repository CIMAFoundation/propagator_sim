import logging
from dataclasses import dataclass, field
from os.path import join

import numpy as np
import rasterio as rio
import scipy
import utm

from propagator.geo import GeographicInfo
from propagator.input import PropagatorDataLoaderException
from propagator.propagator import PropagatorError
from propagator.settings import DEFAULT_TILES_TAG

from .protocol import PropagatorInputDataProtocol

DATA_DIR = "./data/"


def load_tile(zone_number, var, tile_i, tile_j, tileset=DEFAULT_TILES_TAG):
    """
    Load a tile from the data directory, either as a .mat or .tif file.
    :param zone_number: UTM zone number
    :param var: Variable name (e.g., "quo" or "prop")
    :param tile_i: Tile index in the i direction
    :param tile_j: Tile index in the j direction
    :param tileset: Tileset name (default is "tiles")
    """
    filename = var + "_" + str(tile_j) + "_" + str(tile_i) + ".mat"
    filename_tif = var + "_" + str(tile_j) + "_" + str(tile_i) + ".tif"

    filepath = join(DATA_DIR, tileset, str(zone_number), filename)
    logging.debug(filepath)
    try:
        mat_file = scipy.io.loadmat(filepath)
        m = mat_file["M"]
    except FileNotFoundError:
        try:
            filepath = join(DATA_DIR, tileset, str(zone_number), filename_tif)
            logging.debug(filepath)
            with rio.open(filepath) as src:
                m = src.read(1)
        except FileNotFoundError:
            raise PropagatorDataLoaderException()

    return np.ascontiguousarray(m)


def load_tile_ref(
    zone_number: int, var: str, tileset=DEFAULT_TILES_TAG
) -> tuple[int, int, float, float, int]:
    """
    Load the reference file for the zone, which contains metadata such as step size and tile dimensions.
    :param zone_number: UTM zone number
    :param var: Variable name (e.g., "quo" or "prop")
    :param tileset: Tileset name
    """
    filename = join(DATA_DIR, tileset, str(zone_number), var + "_ref.mat")
    logging.debug(filename)
    mat_file = scipy.io.loadmat(filename)
    step_x, step_y, max_y, min_x, tile_dim = (
        mat_file["stepx"][0][0],
        mat_file["stepy"][0][0],
        mat_file["maxy"][0][0],
        mat_file["minx"][0][0],
        mat_file["tileDim"][0][0],
    )

    step_x = int(step_x)
    step_y = int(step_y)

    return step_x, step_y, max_y, min_x, tile_dim


def load_tiles(
    zone_number: int,
    x: float,
    y: float,
    dim: int,
    var: str,
    tileset: str = DEFAULT_TILES_TAG,
) -> np.ndarray:
    step_x, step_y, max_y, min_x, tile_dim = load_tile_ref(zone_number, var, tileset)
    i = 1 + np.floor((max_y - y) / step_y)
    j = 1 + np.floor((x - min_x) / step_x)

    half_dim = np.ceil(dim / 2)
    i_min = i - half_dim
    j_min = j - half_dim
    i_max = i + half_dim
    j_max = j + half_dim

    def get_tile(t_i, t_dim):
        return int(1 + np.floor(t_i / t_dim))

    def get_idx(t_i, t_dim):
        return int(t_i % t_dim)

    tile_i_min = get_tile(i_min, tile_dim)
    idx_i_min = get_idx(i_min, tile_dim)
    tile_i_max = get_tile(i_max, tile_dim)
    idx_i_max = get_idx(i_max, tile_dim)

    tile_j_min = get_tile(j_min, tile_dim)
    idx_j_min = get_idx(j_min, tile_dim)
    tile_j_max = get_tile(j_max, tile_dim)
    idx_j_max = get_idx(j_max, tile_dim)

    if tile_i_max == tile_i_min and tile_j_max == tile_j_min:
        m = load_tile(zone_number, var, tile_i_min, tile_j_min, tileset)
        mat = m[idx_i_min:idx_i_max, idx_j_min:idx_j_max]
    elif tile_i_min == tile_i_max:
        m1 = load_tile(zone_number, var, tile_i_min, tile_j_min, tileset)
        m2 = load_tile(zone_number, var, tile_i_min, tile_j_max, tileset)
        m = np.concatenate([m1, m2], axis=1)
        mat = m[idx_i_min:idx_i_max, idx_j_min : (tile_dim + idx_j_max)]

    elif tile_j_min == tile_j_max:
        m1 = load_tile(zone_number, var, tile_i_min, tile_j_min, tileset)
        m2 = load_tile(zone_number, var, tile_i_max, tile_j_min, tileset)
        m = np.concatenate([m1, m2], axis=0)
        mat = m[idx_i_min : (tile_dim + idx_i_max), idx_j_min:idx_j_max]
    else:
        m1 = load_tile(zone_number, var, tile_i_min, tile_j_min, tileset)
        m2 = load_tile(zone_number, var, tile_i_min, tile_j_max, tileset)
        m3 = load_tile(zone_number, var, tile_i_max, tile_j_min, tileset)
        m4 = load_tile(zone_number, var, tile_i_max, tile_j_max, tileset)
        m = np.concatenate(
            [np.concatenate([m1, m2], axis=1), np.concatenate([m3, m4], axis=1)], axis=0
        )
        mat = m[idx_i_min : (tile_dim + idx_i_max), idx_j_min : (tile_dim + idx_j_max)]

    return mat


class NoTilesError(PropagatorError):
    def __init__(self):
        self.message = """Can't initialize simulation, no data on the selected area"""
        super().__init__(self.message)


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
        step_x, step_y, *_ = load_tile_ref(self.zone_number, "quo", self.tileset)
        self.step_x = step_x
        self.step_y = step_y

    def get_dem(self) -> np.ndarray:
        dem, _, _, _, _ = load_tiles(
            self.zone_number,
            self.easting,
            self.northing,
            self.grid_dim,
            "quo",
            self.tileset,
        )
        dem = dem.astype("float")

        return dem

    def get_veg(self) -> np.ndarray:
        _, veg, _, _, _ = load_tiles(
            self.zone_number,
            self.easting,
            self.northing,
            self.grid_dim,
            "prop",
            self.tileset,
        )
        veg[:, (0, 1, 2, -3, -2, -1)] = 0
        veg[(0, 1, 2, -3, -2, -1), :] = 0
        veg = veg.astype("int8")

        return veg

    def get_geo_info(self) -> GeographicInfo:
        rows = self.grid_dim
        cols = self.grid_dim
        north = self.northing + ((rows / 2) * self.step_y)
        east = self.easting + ((cols / 2) * self.step_x)
        south = self.northing - ((rows / 2) * self.step_y)
        west = self.easting - ((cols / 2) * self.step_x)

        geo_info = GeographicInfo.from_bounds(
            west, south, east, north, self.step_x, self.step_y, self.zone_number
        )

        return geo_info
