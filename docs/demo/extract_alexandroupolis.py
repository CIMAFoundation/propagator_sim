"""Extract an Alexandroupolis / Evros 2023 window from the EU COGs on S3 and
pack it into a compact binary the WebAssembly demo can fetch.

Layout of the output file (little-endian):
    uint32  rows
    uint32  cols
    float32 cellsize (metres)
    int32   ignition_row
    int32   ignition_col
    int8    veg[rows*cols]          (eu12 fuel codes, 0 = no fuel)
    (padding to a 2-byte boundary)
    int16   dem[rows*cols]          (elevation, metres; sea/nodata -> 0)

The whole payload is gzip-compressed on disk (the demo decompresses it in the
browser with DecompressionStream).
"""

import gzip
import os
import struct

import numpy as np
import rasterio as rio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

FUEL = "/vsis3/cima-propagator-return/cogs/eu/eu_fuel12_utm_35.tif"
DEM = "/vsis3/cima-propagator-return/cogs/eu/eu_dem_utm_35.tif"
LON, LAT = 26.122, 40.932  # ignition (config.json)
NATIVE = 1024  # native window side (20 m cells -> ~20 km)
FACTOR = 1  # full 20 m resolution, 1024 x 1024 grid
OUT = "alexandroupolis.bin.gz"  # written next to this script (docs/demo/)

N = NATIVE // FACTOR  # 256

with rio.open(DEM) as dem_src, rio.open(FUEL) as fuel_src:
    tf = Transformer.from_crs("EPSG:4326", dem_src.crs, always_xy=True)
    x, y = tf.transform(LON, LAT)
    prow, pcol = dem_src.index(x, y)
    r0, c0 = int(prow) - NATIVE // 2, int(pcol) - NATIVE // 2
    win = Window(c0, r0, NATIVE, NATIVE)
    cellsize = float(dem_src.res[0]) * FACTOR

    dem = dem_src.read(
        1,
        window=win,
        out_shape=(N, N),
        resampling=Resampling.average,
        boundless=True,
        fill_value=-9999.0,
    ).astype(np.float32)
    dem[dem == -9999.0] = 0.0
    dem[dem < 0] = 0.0

    veg = fuel_src.read(
        1,
        window=win,
        out_shape=(N, N),
        resampling=Resampling.mode,
        boundless=True,
        fill_value=0,
    ).astype(np.int8)

ig_r, ig_c = N // 2, N // 2
uniq, counts = np.unique(veg, return_counts=True)
print(
    "grid",
    N,
    "x",
    N,
    "cellsize",
    cellsize,
    "m  extent",
    N * cellsize / 1000,
    "km",
)
print("fuel classes:", dict(zip(uniq.tolist(), counts.tolist())))
print("dem range", float(dem.min()), "-", float(dem.max()))
print("ignition fuel code:", int(veg[ig_r, ig_c]))

veg_bytes = veg.tobytes()
pad = (-len(veg_bytes)) % 2  # int16 dem needs 2-byte alignment
dem_i16 = np.clip(np.round(dem), 0, 32767).astype("<i2")
header = struct.pack("<IIfii", N, N, cellsize, ig_r, ig_c)
payload = header + veg_bytes + b"\x00" * pad + dem_i16.tobytes()
with gzip.open(OUT, "wb", compresslevel=9) as f:
    f.write(payload)
print(
    "wrote", OUT, "raw bytes:", len(payload), "gzipped:", os.path.getsize(OUT)
)
