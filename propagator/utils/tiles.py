from itertools import count
import logging
from os.path import join

import fiona
import numpy as np
import rasterio as rio
import scipy.io
import utm
from numpy import pi
from pyproj import Proj
from rasterio import band, crs, transform, warp, enums
from rasterio.features import shapes
from scipy.ndimage import filters
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.signal.signaltools import medfilt2d
from shapely.geometry import mapping
from shapely.geometry import shape, MultiLineString, LineString
from sortedcontainers import SortedDict

from ..constants import *

DATA_DIR = 'data'

def load_tile_ref(zone_number, var, tileset=DEFAULT_TAG):
    filename = join(DATA_DIR, tileset, str(zone_number), var + '_ref.mat')
    logging.debug(filename)
    mat_file = scipy.io.loadmat(filename)
    step_x, step_y, max_y, min_x, tile_dim = \
        mat_file['stepx'][0][0], mat_file['stepy'][0][0], \
        mat_file['maxy'][0][0], mat_file['minx'][0][0], mat_file['tileDim'][0][0]
    return step_x, step_y, max_y, min_x, tile_dim


def load_tiles(zone_number, x, y, dim, var, tileset=DEFAULT_TAG):
    step_x, step_y, max_y, min_x, tile_dim = load_tile_ref(zone_number, var, tileset)
    i = 1 + np.floor((max_y - y) / step_y)
    j = 1 + np.floor((x - min_x) / step_x)

    half_dim = np.ceil(dim / 2)
    i_min = i - half_dim
    j_min = j - half_dim
    i_max = i + half_dim
    j_max = j + half_dim
    min_easting = (j_min * step_x) + min_x
    max_northing = max_y - (i_min * step_y)

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
        m = load_tile(zone_number, var, tile_i_min, tile_j_min, dim, tileset)
        mat = m[idx_i_min:idx_i_max, idx_j_min: idx_j_max]
    elif tile_i_min == tile_i_max:
        m1 = load_tile(zone_number, var, tile_i_min, tile_j_min, dim, tileset)
        m2 = load_tile(zone_number, var, tile_i_min, tile_j_max, dim, tileset)
        m = np.concatenate([m1, m2], axis=1)
        mat = m[idx_i_min:idx_i_max, idx_j_min: (tile_dim + idx_j_max)]

    elif tile_j_min == tile_j_max:

        m1 = load_tile(zone_number, var, tile_i_min, tile_j_min, dim, tileset)
        m2 = load_tile(zone_number, var, tile_i_max, tile_j_min, dim, tileset)
        m = np.concatenate([m1, m2], axis=0)
        mat = m[idx_i_min:(tile_dim + idx_i_max), idx_j_min: idx_j_max]
    else:
        m1 = load_tile(zone_number, var, tile_i_min, tile_j_min, dim, tileset)
        m2 = load_tile(zone_number, var, tile_i_min, tile_j_max, dim, tileset)
        m3 = load_tile(zone_number, var, tile_i_max, tile_j_min, dim, tileset)
        m4 = load_tile(zone_number, var, tile_i_max, tile_j_max, dim, tileset)
        m = np.concatenate([
            np.concatenate([m1, m2], axis=1),
            np.concatenate([m3, m4], axis=1)
        ], axis=0)
        mat = m[idx_i_min:(tile_dim + idx_i_max), idx_j_min: (tile_dim + idx_j_max)]

    return mat, min_easting, max_northing, step_x, step_y

def load_tile(zone_number, var, tile_i, tile_j, dim,  tileset=DEFAULT_TAG):
    filename = var + '_' + str(tile_j) + '_' + str(tile_i) + '.mat'

    filepath = join(DATA_DIR, tileset, str(zone_number), filename)
    logging.debug(filepath)
    try:
       mat_file = scipy.io.loadmat(filepath)
       m = mat_file['M']
    except:
       m = np.nan * np.ones((dim, dim))
    return np.ascontiguousarray(m)
