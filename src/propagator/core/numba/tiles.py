"""Block-sparse (tiled) storage for per-realization state arrays.

Per-cell state (arrival time, rate of spread, fireline intensity) is stored
in fixed-size square tiles allocated on demand as the fire reaches them, so
memory scales with the burned area instead of grid size x realizations.

Each realization owns a tile-index grid mapping every (TILE_SIZE x
TILE_SIZE) block of the domain to either -1 (unallocated) or a slot in a
growable per-realization tile pool.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange  # type: ignore

TILE_SHIFT = 5
TILE_SIZE = 1 << TILE_SHIFT
TILE_MASK = TILE_SIZE - 1

# Bitfield flags stored in the per-cell uint8 tile pool.
FLAG_FIRE = np.uint8(1)
FLAG_SPOT_GEN = np.uint8(2)
FLAG_SPOT_RECV = np.uint8(4)


@njit(cache=False, inline="always")
def read_flags(
    tile_idx_r: npt.NDArray[np.int32],
    tile_flags_r: npt.NDArray[np.uint8],
    row: int,
    col: int,
) -> np.uint8:
    """Read the flags byte for a cell in one realization (0 if unallocated)."""
    tile = tile_idx_r[row >> TILE_SHIFT, col >> TILE_SHIFT]
    if tile < 0:
        return np.uint8(0)
    return tile_flags_r[tile, row & TILE_MASK, col & TILE_MASK]


@njit(cache=False, parallel=True)
def fold_state_tiles(
    tile_idx: npt.NDArray[np.int32],
    tile_flags: npt.NDArray[np.uint8],
    tile_arrival: npt.NDArray[np.int32],
    tile_ros: npt.NDArray[np.float32],
    tile_fli: npt.NDArray[np.float32],
    count: npt.NDArray[np.int32],
    spot_gen_count: npt.NDArray[np.int32],
    spot_recv_count: npt.NDArray[np.int32],
    arrival_min: npt.NDArray[np.int32],
    arrival_sum: npt.NDArray[np.float64],
    ros_sum: npt.NDArray[np.float64],
    ros_max: npt.NDArray[np.float32],
    fli_sum: npt.NDArray[np.float64],
    fli_max: npt.NDArray[np.float32],
) -> None:
    """Aggregate tiled state across realizations into per-cell 2D maps.

    Spotting flags are counted for every flagged cell; arrival/ros/fli only
    contribute for burned cells (FLAG_FIRE set). Callers must pass `count`,
    spot counts, sums and maxes zero-initialized and `arrival_min`
    initialized to int32 max. NaN ros/fli values contribute to `count` but
    are skipped in sums and maxes, matching the previous dense
    nansum/nanmax behaviour.
    """
    n_realizations, tiles_h, tiles_w = tile_idx.shape
    n_rows = count.shape[0]
    n_cols = count.shape[1]

    # parallel over spatial tiles: each output cell is owned by exactly one
    # thread, so the in-place accumulation is race-free
    for tile_flat in prange(tiles_h * tiles_w):
        tile_row = tile_flat // tiles_w
        tile_col = tile_flat % tiles_w
        row0 = tile_row << TILE_SHIFT
        col0 = tile_col << TILE_SHIFT
        for realization in range(n_realizations):
            tile = tile_idx[realization, tile_row, tile_col]
            if tile < 0:
                continue
            for local_row in range(TILE_SIZE):
                row = row0 + local_row
                if row >= n_rows:
                    break
                for local_col in range(TILE_SIZE):
                    col = col0 + local_col
                    if col >= n_cols:
                        break
                    flags = tile_flags[realization, tile, local_row, local_col]
                    if flags == 0:
                        continue
                    if flags & FLAG_SPOT_GEN:
                        spot_gen_count[row, col] += 1
                    if flags & FLAG_SPOT_RECV:
                        spot_recv_count[row, col] += 1
                    if not flags & FLAG_FIRE:
                        continue
                    count[row, col] += 1
                    arrival = tile_arrival[
                        realization, tile, local_row, local_col
                    ]
                    if arrival < arrival_min[row, col]:
                        arrival_min[row, col] = arrival
                    arrival_sum[row, col] += arrival
                    ros_value = tile_ros[
                        realization, tile, local_row, local_col
                    ]
                    if not np.isnan(ros_value):
                        ros_sum[row, col] += ros_value
                        if ros_value > ros_max[row, col]:
                            ros_max[row, col] = ros_value
                    fli_value = tile_fli[
                        realization, tile, local_row, local_col
                    ]
                    if not np.isnan(fli_value):
                        fli_sum[row, col] += fli_value
                        if fli_value > fli_max[row, col]:
                            fli_max[row, col] = fli_value


@njit(cache=False, parallel=True)
def materialize_tiles(
    tile_idx: npt.NDArray[np.int32],
    pool: npt.NDArray,
    out: npt.NDArray,
) -> None:
    """Scatter allocated tiles into a dense (realizations, rows, cols) array.

    Cells in unallocated tiles keep whatever value `out` was pre-filled with.
    """
    n_realizations, tiles_h, tiles_w = tile_idx.shape
    n_rows = out.shape[1]
    n_cols = out.shape[2]
    for realization in prange(n_realizations):
        for tile_row in range(tiles_h):
            for tile_col in range(tiles_w):
                tile = tile_idx[realization, tile_row, tile_col]
                if tile < 0:
                    continue
                for local_row in range(TILE_SIZE):
                    row = (tile_row << TILE_SHIFT) + local_row
                    if row >= n_rows:
                        break
                    for local_col in range(TILE_SIZE):
                        col = (tile_col << TILE_SHIFT) + local_col
                        if col >= n_cols:
                            break
                        out[realization, row, col] = pool[
                            realization, tile, local_row, local_col
                        ]
