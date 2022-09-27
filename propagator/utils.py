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

from .constants import *

DATA_DIR = 'data'


def normalize(angle_to_norm):
    return (angle_to_norm + pi) % (2 * pi) - pi


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
                    if results[t][p].type == 'MultiLineString':
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
                if results[t][p].type == 'MultiLineString':
                    features.append({
                        'type': 'Feature',
                        'geometry': mapping(results[t][p]),
                        'properties': {
                            'value': p,
                            TIME_TAG: t
                        },
                    })
        with open(filename, "w") as f:
            f.write(json.dumps(geojson_obj))


