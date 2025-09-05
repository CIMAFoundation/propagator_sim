import logging
from os.path import join

import fiona
import numpy as np
import rasterio as rio
import scipy.io
import utm
from numpy import pi
from rasterio import band, crs, transform, warp, enums
from rasterio.features import shapes
from scipy.ndimage import filters
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.signal.signaltools import medfilt2d
from shapely.geometry import mapping
from shapely.geometry import shape, MultiLineString, LineString

from propagator.constants import DEFAULT_TAG, TIME_TAG, PROPAGATOR_PATH

DATA_DIR = PROPAGATOR_PATH + '/data'

def normalize(angle_to_norm):
    return (angle_to_norm + pi) % (2 * pi) - pi



def load_tile(zone_number, var, tile_i, tile_j, dim,  tileset=DEFAULT_TAG):
    filename = var + '_' + str(tile_j) + '_' + str(tile_i) + '.mat'
    filename_tif = var + '_' + str(tile_j) + '_' + str(tile_i) + '.tif'

    filepath = join(DATA_DIR, tileset, str(zone_number), filename)
    logging.debug(filepath)
    try:
       mat_file = scipy.io.loadmat(filepath)
       m = mat_file['M']
    except FileNotFoundError:
        try:
            filepath = join(DATA_DIR, tileset, str(zone_number), filename_tif)
            logging.debug(filepath)
            with rio.open(filepath) as src:
                m = src.read(1)
        except FileNotFoundError:
            m = np.nan * np.ones((dim, dim))
            
    return np.ascontiguousarray(m)


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


def add_point(img, c, r, val):
    if 0 <= c < img.shape[1] and 0 <= r < img.shape[0]:
        img[r, c] = val
    return [(r, c)]


def add_segment(img, c0, r0, c1, r1, value):
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    if c0 < c1:
        sc = 1
    else:
        sc = -1

    if r0 < r1:
        sr = 1
    else:
        sr = -1

    err = dc - dr
    points = []
    while True:
        if 0 <= c0 < img.shape[1] and 0 <= r0 < img.shape[0]:
            img[r0, c0] = value
            points.append((r0, c0))

        if c0 == c1 and r0 == r1:
            break

        e2 = 2 * err
        if e2 > -dr:
            err = err - dr
            c0 += sc

        if e2 < dc:
            err = err + dc
            r0 += sr

    return points


def add_line(img, cs, rs, val):
    contour = []
    img_temp = np.zeros(img.shape)

    for idx in range(len(cs) - 1):
        points = add_segment(img_temp, cs[idx], rs[idx], cs[idx + 1], rs[idx + 1], 1)
        if idx > 0:
            contour.extend(points[1:])
        else:
            contour.extend(points)

    img[img_temp == 1] = val

    return contour


def add_poly(img, cs, rs, val):
    img_temp = np.ones(img.shape)
    contour = []

    for idx in range(len(cs) - 1):
        points = add_segment(img_temp, cs[idx], rs[idx], cs[idx + 1], rs[idx + 1], 2)
        if idx > 0:
            contour.extend(points[1:])
        else:
            contour.extend(points)

    points = add_segment(img_temp, cs[-1], rs[-1], cs[0], rs[0], 2)
    contour.extend(points[1:])

    pp = [(0, 0)]
    dim_y, dim_x = img_temp.shape

    while len(pp) > 0:
        pp_n = []
        for (x, y) in pp:
            if y < dim_y - 1 and img_temp[y + 1, x] == 1:
                img_temp[y + 1, x] = 0
                pp_n.append((x, y + 1))

            if x < dim_x - 1 and img_temp[y, x + 1] == 1:
                img_temp[y, x + 1] = 0
                pp_n.append((x + 1, y))

            if y > 0 and img_temp[y - 1, x] == 1:
                img_temp[y - 1, x] = 0
                pp_n.append((x, y - 1))

            if x > 0 and img_temp[y, x - 1] == 1:
                img_temp[y, x - 1] = 0
                pp_n.append((x - 1, y))

        pp = pp_n

    img[img_temp > 0] = val
    return contour

def read_actions(imp_points_string):
    strings = imp_points_string.split('\n')

    polys, lines, points = [], [], []
    max_lat, max_lon, min_lat, min_lon = -np.inf, -np.inf, np.inf, np.inf

    for s in strings:
        f_type, values = s.split(':')
        values = values.replace('[', '').replace(']', '')
        if f_type == 'POLYGON':
            s_lats, s_lons = values.split(';')
            lats = [float(sv) for sv in s_lats.split()]
            lons = [float(sv) for sv in s_lons.split()]
            polys.append((lats, lons))

        elif f_type == 'LINE':
            s_lats, s_lons = values.split(';')
            lats = [float(sv) for sv in s_lats.split()]
            lons = [float(sv) for sv in s_lons.split()]
            lines.append((lats, lons))

        elif f_type == 'POINT':
            s_lat, s_lon = values.split(';')
            lat, lon = float(s_lat), float(s_lon)
            lats = [lat]
            lons = [lon]
            points.append((lat, lon))

        max_lat = max(max(lats), max_lat)
        min_lat = min(min(lats), min_lat)
        max_lon = max(max(lons), max_lon)
        min_lon = min(min(lons), min_lon)

    mid_lat = (max_lat + min_lat) / 2
    mid_lon = (max_lon + min_lon) / 2

    return mid_lat, mid_lon, polys, lines, points 


def rasterize_actions(dim, points, lines, polys, lonmin, latmax, stepx, stepy, zone_number, base_value=0, value=1):
    img = np.ones(dim) * base_value
    active_points = []
    for line in lines:
        xs, ys, _, _ = zip(*[
            utm.from_latlon(p[0], p[1], force_zone_number=zone_number)
            for p in zip(*line)
        ])
        x = np.floor((np.array(xs) - lonmin) / stepx).astype('int')
        y = np.floor((latmax - np.array(ys)) / stepy).astype('int')
        active = add_line(img, x, y, 1)
        active_points.extend(active)
    for point in points:
        xs, ys, _, _ = utm.from_latlon(point[0], point[1], force_zone_number=zone_number)
        x = int(np.floor((xs - lonmin) / stepx))
        y = int(np.floor((latmax - ys) / stepy))
        active = add_point(img, x, y, 1)
        active_points.extend(active)
    for poly in polys:
        xs, ys, _, _ = zip(*[
            utm.from_latlon(p[0], p[1], force_zone_number=zone_number)
            for p in zip(*poly)
        ])
        x = np.floor((np.array(xs) - lonmin) / stepx).astype('int')
        y = np.floor((latmax - np.array(ys)) / stepy).astype('int')
        active = add_poly(img, x, y, 1)
        active_points.extend(active)

    return img, active_points


def trim_values(values, src_trans):
    rows, cols = values.shape
    min_row, max_row = int(rows / 2 - 1), int(rows / 2 + 1)
    min_col, max_col = int(cols / 2 - 1), int(cols / 2 + 1)

    v_rows = np.where(values.sum(axis=1) > 0)[0]
    if len(v_rows) > 0:
        min_row, max_row = v_rows[0] - 1, v_rows[-1] + 2

    v_cols = np.where(values.sum(axis=0) > 0)[0]
    if len(v_cols) > 0:
        min_col, max_col = v_cols[0] - 1, v_cols[-1] + 2

    trim_values = values[min_row:max_row, min_col:max_col]    
    rows, cols = trim_values.shape

    (west, east), (north, south) = rio.transform.xy(
        src_trans, [min_row, max_row], [min_col, max_col],
        offset='ul'
    )
    trim_trans = transform.from_bounds(west, south, east, north, cols, rows)
    return trim_values, trim_trans


def reproject(values, src_trans, src_crs, dst_crs, trim=True):
    if trim:
        values, src_trans = trim_values(values, src_trans)

    rows, cols = values.shape
    (west, east), (north, south) = rio.transform.xy(
        src_trans, [0, rows], [0, cols],
        offset='ul'
    )

    with rio.Env():
        dst_trans, dw, dh = warp.calculate_default_transform(
            src_crs=src_crs,
            dst_crs=dst_crs,
            width=cols,
            height=rows,
            left=west,
            bottom=south,
            right=east,
            top=north,
            resolution=None
        )
        dst = np.empty((dh, dw))

        warp.reproject(
            source=np.ascontiguousarray(values), 
            destination=dst,
            src_crs=src_crs, 
            dst_crs=dst_crs,
            dst_transform=dst_trans, 
            src_transform=src_trans,
            resampling=enums.Resampling.nearest,
            num_threads=1
        )
    
    return dst, dst_trans

def write_geotiff(filename, values, dst_trans, dst_crs, dtype=np.uint8):
    with rio.Env():
        with rio.open(
                filename,
                'w',
                driver='GTiff',
                width=values.shape[1],
                height=values.shape[0],
                count=1,
                dtype=dtype,
                nodata=0,
                transform=dst_trans,
                crs=dst_crs) as f:
            f.write(values.astype(dtype), indexes=1)


def smooth_linestring(linestring, smooth_sigma):
    """
    Uses a gauss filter to smooth out the LineString coordinates.
    """
    smooth_x = np.array(
        filters.gaussian_filter1d(
            linestring.xy[0],
            smooth_sigma
        ))
    smooth_y = np.array(
        filters.gaussian_filter1d(
            linestring.xy[1],
            smooth_sigma
        ))

    # close the linestring
    smooth_y[-1] = smooth_y[0]
    smooth_x[-1] = smooth_x[0]

    smoothed_coords = np.hstack((smooth_x, smooth_y))
    smoothed_coords = zip(smooth_x, smooth_y)

    linestring_smoothed = LineString(smoothed_coords)

    return linestring_smoothed


def extract_isochrone(values, transf,
                      thresholds=[0.5, 0.75, 0.9],
                      med_filt_val=9, min_length=0.0001,
                      smooth_sigma=0.8, simp_fact=0.00001):
    '''
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
    '''

    # if the dimension of the burned area is low, we do not filter it
    if np.sum(values > 0) <= 100:
        filt_values = values
    else:
        filt_values = medfilt2d(values, med_filt_val)
    results = {}

    for t in thresholds:
        over_t_ = (filt_values >= t).astype('uint8')
        over_t = binary_dilation(binary_erosion(over_t_).astype('uint8')).astype('uint8')
        if np.any(over_t):
            for s, v in shapes(over_t, transform=transf):
                sh = shape(s)

                ml = [
                    smooth_linestring(l, smooth_sigma) # .simplify(simp_fact)
                    for l in sh.interiors
                    if l.length > min_length
                ]

                results[t] = MultiLineString(ml)

    return results


def save_isochrones(results, filename, format='geojson'):
    if format == 'shp':
        schema = {
            'geometry': 'MultiLineString',
            'properties': {'value': 'float', TIME_TAG: 'int'},
        }
        # Write a new Shapefile
        with fiona.open(filename, 'w', 'ESRI Shapefile', schema) as c:
            for t in results:
                for p in results[t]:
                    if results[t][p].geom_type == 'MultiLineString':
                        c.write({
                            'geometry': mapping(results[t][p]),
                            'properties': {
                                'value': p,
                                TIME_TAG: t
                            },
                        })

    if format == 'geojson':
        import json
        features = []
        geojson_obj = dict(type='FeatureCollection', features=features)
        for t in results:
            for p in results[t]:
                if results[t][p].geom_type == 'MultiLineString':
                    features.append({
                        'geom_type': 'Feature',
                        'geometry': mapping(results[t][p]),
                        'properties': {
                            'value': p,
                            TIME_TAG: t
                        },
                    })
        with open(filename, "w") as f:
            f.write(json.dumps(geojson_obj))


if __name__ == '__main__':
    grid_dim = 1000
    tileset = DEFAULT_TAG
    s1 = [
        "LINE:[44.3204247306364 44.320317268240956 ];[8.44812858849764 8.449995405972006 ]",
        "POLYGON:[44.32214410219511 44.320869929892176 44.32083922660368 44.32214410219511 ];[8.454050906002523 8.453171141445639 8.45463026314974 8.454050906002523 ]",
        "POINT:44.32372526549074;8.45040310174227"]
    ignition_string = '\n'.join(s1)
    mid_lat, mid_lon, polys, lines, points = read_actions(ignition_string)
    easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)
    src, west, north, step_x, step_y = load_tiles(zone_number, easting, northing, grid_dim, 'prop', tileset)

    dst, dst_trans, dst_crs = reproject(src, (west, north, step_x, step_y), zone_number, zone_letter)
    write_geotiff('test_latolng.tiff', dst, dst_trans, dst_crs)
