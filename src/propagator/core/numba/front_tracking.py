"""Numba-friendly front-tracking propagation kernel."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange  # type: ignore

from propagator.core.constants import NO_FUEL

from .propagation import (
    MAX_SPOTTING_EMBERS,
    N_NEIGHBOURS,
    compute_spotting,
    try_spread_to_neighbour,
)
from .tiles import (
    FLAG_FIRE,
    FLAG_SPOT_GEN,
    FLAG_SPOT_RECV,
    TILE_MASK,
    TILE_SHIFT,
    read_flags,
)

# Worst-case number of events a single pop can push (all neighbours plus a
# capped ember burst). The kernel only pops when this headroom is available,
# so an overflow always suspends with the heap intact and the caller can
# regrow the buffers and resume.
FRONT_RESERVE = N_NEIGHBOURS + MAX_SPOTTING_EMBERS

# Source code of a front event not produced by contact spread (an ignition
# boundary condition or an ember landing). Contact-spread events store the
# index of the NEIGHBOURS offset that produced them: the source cell is
# ``target - NEIGHBOURS[src]``. Used by ``Propagator.extinguish``.
SRC_EXTERNAL = 255

# Worst-case number of tiles a single pop can allocate: the popped cell's
# tile plus one tile per ember landing cell.
TILE_RESERVE = 1 + MAX_SPOTTING_EMBERS


@njit(cache=False)
def _heap_swap(
    times: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32],
    ros: npt.NDArray[np.float32],
    fli: npt.NDArray[np.float32],
    src: npt.NDArray[np.uint8],
    i: int,
    j: int,
) -> None:
    times[i], times[j] = times[j], times[i]
    rows[i], rows[j] = rows[j], rows[i]
    cols[i], cols[j] = cols[j], cols[i]
    ros[i], ros[j] = ros[j], ros[i]
    fli[i], fli[j] = fli[j], fli[i]
    src[i], src[j] = src[j], src[i]


@njit(cache=False)
def _heap_push(
    times: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32],
    ros: npt.NDArray[np.float32],
    fli: npt.NDArray[np.float32],
    src: npt.NDArray[np.uint8],
    size: int,
    time: int,
    row: int,
    col: int,
    ros_value: float,
    fli_value: float,
    src_value: int,
) -> int:
    times[size] = time
    rows[size] = row
    cols[size] = col
    ros[size] = ros_value
    fli[size] = fli_value
    src[size] = src_value

    idx = size
    while idx > 0:
        parent = (idx - 1) // 2
        if times[parent] <= times[idx]:
            break
        _heap_swap(times, rows, cols, ros, fli, src, parent, idx)
        idx = parent
    return size + 1


@njit(cache=False)
def _heap_pop_min(
    times: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32],
    ros: npt.NDArray[np.float32],
    fli: npt.NDArray[np.float32],
    src: npt.NDArray[np.uint8],
    size: int,
) -> tuple[int, int, int, float, float, int]:
    if size == 0:
        return 0, 0, 0, 0.0, 0.0, 0

    time = times[0]
    row = rows[0]
    col = cols[0]
    ros_value = ros[0]
    fli_value = fli[0]

    size -= 1
    if size > 0:
        times[0] = times[size]
        rows[0] = rows[size]
        cols[0] = cols[size]
        ros[0] = ros[size]
        fli[0] = fli[size]
        src[0] = src[size]

        idx = 0
        while True:
            left = idx * 2 + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and times[right] < times[left]:
                smallest = right
            if times[idx] <= times[smallest]:
                break
            _heap_swap(times, rows, cols, ros, fli, src, idx, smallest)
            idx = smallest

    return time, row, col, ros_value, fli_value, size


@njit(cache=False, parallel=True, fastmath=True)
def advance_front_until(
    end_time: int,
    max_events: int,
    event_times: npt.NDArray[np.int32],
    event_rows: npt.NDArray[np.int32],
    event_cols: npt.NDArray[np.int32],
    event_ros: npt.NDArray[np.float32],
    event_fli: npt.NDArray[np.float32],
    event_src: npt.NDArray[np.uint8],
    sizes: npt.NDArray[np.int32],
    overflow: npt.NDArray[np.int8],
    cellsize: float,
    veg: npt.NDArray[np.integer],
    fuel_idx: npt.NDArray[np.int64],
    dem: npt.NDArray[np.floating],
    tile_idx: npt.NDArray[np.int32],
    tile_counts: npt.NDArray[np.int32],
    tile_capacity: int,
    tile_flags: npt.NDArray[np.uint8],
    tile_arrival: npt.NDArray[np.int32],
    tile_ros: npt.NDArray[np.float32],
    tile_fli: npt.NDArray[np.float32],
    moisture: npt.NDArray[np.floating],
    wind_dir: npt.NDArray[np.floating],
    wind_speed: npt.NDArray[np.floating],
    fuels,
    p_time_fn,
    p_moist_fn,
    out_of_bounds: npt.NDArray[np.int8],
    track_spotting: bool,
    halt_on_boundary: bool,
) -> None:
    n_realizations = sizes.shape[0]
    n_rows = veg.shape[0]
    n_cols = veg.shape[1]
    total_tiles = tile_idx.shape[1] * tile_idx.shape[2]

    for realization in prange(n_realizations):
        if overflow[realization] != 0:
            continue

        while sizes[realization] > 0:
            if event_times[realization, 0] > end_time:
                break

            # suspend (heap intact) if the next pop cannot be guaranteed to
            # fit: worst-case heap pushes, or worst-case tile allocations
            # (bounded by the tiles that can still be allocated at all)
            tile_demand = total_tiles - tile_counts[realization]
            if tile_demand > TILE_RESERVE:
                tile_demand = TILE_RESERVE
            if (
                sizes[realization] + FRONT_RESERVE > max_events
                or tile_counts[realization] + tile_demand > tile_capacity
            ):
                overflow[realization] = 1
                break

            # suspend (heap intact, event not popped) if the next event
            # ignites a boundary-ring cell: the caller can checkpoint,
            # expand the domain and resume without losing any spread
            if halt_on_boundary:
                peek_row = event_rows[realization, 0]
                peek_col = event_cols[realization, 0]
                if (
                    peek_row <= 0
                    or peek_col <= 0
                    or peek_row >= n_rows - 1
                    or peek_col >= n_cols - 1
                ) and not (
                    read_flags(
                        tile_idx[realization],
                        tile_flags[realization],
                        peek_row,
                        peek_col,
                    )
                    & FLAG_FIRE
                ):
                    out_of_bounds[realization] = 1
                    break

            (
                time,
                row,
                col,
                ros_value,
                fli_value,
                new_size,
            ) = _heap_pop_min(
                event_times[realization],
                event_rows[realization],
                event_cols[realization],
                event_ros[realization],
                event_fli[realization],
                event_src[realization],
                sizes[realization],
            )
            sizes[realization] = new_size

            tile_idx_r = tile_idx[realization]
            tile_flags_r = tile_flags[realization]

            if read_flags(tile_idx_r, tile_flags_r, row, col) & FLAG_FIRE:
                continue

            if row <= 0 or col <= 0 or row >= n_rows - 1 or col >= n_cols - 1:
                out_of_bounds[realization] = 1

            tile_row = row >> TILE_SHIFT
            tile_col = col >> TILE_SHIFT
            tile = tile_idx_r[tile_row, tile_col]
            if tile < 0:
                tile = tile_counts[realization]
                tile_counts[realization] = tile + 1
                tile_idx_r[tile_row, tile_col] = tile
            local_row = row & TILE_MASK
            local_col = col & TILE_MASK
            tile_flags_r[tile, local_row, local_col] |= FLAG_FIRE
            tile_arrival[realization, tile, local_row, local_col] = time
            tile_ros[realization, tile, local_row, local_col] = ros_value
            tile_fli[realization, tile, local_row, local_col] = fli_value

            if veg[row, col] == NO_FUEL:
                continue

            idx_from = fuel_idx[row, col]
            if not fuels.burn[idx_from]:
                # Forced ignitions may mark a non-combustible cell as
                # burning, but it must not propagate fire outward.
                continue

            w_dir_r = wind_dir[row, col]
            w_speed_r = wind_speed[row, col]

            for neighbour_index in range(N_NEIGHBOURS):
                (
                    ignites,
                    row_to,
                    col_to,
                    delta_time,
                    ros_to,
                    fli_to,
                ) = try_spread_to_neighbour(
                    neighbour_index,
                    row,
                    col,
                    cellsize,
                    veg,
                    fuel_idx,
                    dem,
                    tile_idx_r,
                    tile_flags_r,
                    moisture,
                    w_dir_r,
                    w_speed_r,  # type: ignore
                    fuels,
                    p_time_fn,
                    p_moist_fn,
                )
                if not ignites:
                    continue
                sizes[realization] = _heap_push(
                    event_times[realization],
                    event_rows[realization],
                    event_cols[realization],
                    event_ros[realization],
                    event_fli[realization],
                    event_src[realization],
                    sizes[realization],
                    time + int(delta_time),
                    row_to,
                    col_to,
                    ros_to,
                    fli_to,
                    neighbour_index,
                )

            if fuels.spotting[idx_from]:
                spotting_updates = compute_spotting(
                    row,
                    col,
                    cellsize,
                    veg,
                    fuel_idx,
                    tile_idx_r,
                    tile_flags_r,
                    w_dir_r,
                    w_speed_r,  # type: ignore
                    fli_value,
                    fuels,
                )
                for update in spotting_updates:
                    (
                        delta_time,
                        row_to,
                        col_to,
                        ros_to,
                        fli_to,
                        _is_spotting,
                    ) = update
                    if track_spotting:
                        tile_flags_r[tile, local_row, local_col] |= (
                            FLAG_SPOT_GEN
                        )
                        recv_tile_row = row_to >> TILE_SHIFT
                        recv_tile_col = col_to >> TILE_SHIFT
                        recv_tile = tile_idx_r[recv_tile_row, recv_tile_col]
                        if recv_tile < 0:
                            recv_tile = tile_counts[realization]
                            tile_counts[realization] = recv_tile + 1
                            tile_idx_r[recv_tile_row, recv_tile_col] = (
                                recv_tile
                            )
                        tile_flags_r[
                            recv_tile, row_to & TILE_MASK, col_to & TILE_MASK
                        ] |= FLAG_SPOT_RECV
                    sizes[realization] = _heap_push(
                        event_times[realization],
                        event_rows[realization],
                        event_cols[realization],
                        event_ros[realization],
                        event_fli[realization],
                        event_src[realization],
                        sizes[realization],
                        time + int(delta_time),
                        row_to,
                        col_to,
                        ros_to,
                        fli_to,
                        SRC_EXTERNAL,
                    )


@njit(cache=False)
def heap_retain(
    times: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32],
    ros: npt.NDArray[np.float32],
    fli: npt.NDArray[np.float32],
    src: npt.NDArray[np.uint8],
    size: int,
    keep: npt.NDArray[np.bool_],
) -> int:
    """Keep the first ``size`` events whose ``keep`` flag is set, then
    restore the heap invariant (Floyd heapify). Returns the new size.

    Same algorithm as the Rust core's ``FrontHeap::retain``: stable
    in-place compaction followed by sifting every internal node down.
    """
    w = 0
    for i in range(size):
        if keep[i]:
            _heap_swap(times, rows, cols, ros, fli, src, w, i)
            w += 1
    start = w // 2 - 1
    while start >= 0:
        idx = start
        while True:
            left = idx * 2 + 1
            if left >= w:
                break
            right = left + 1
            smallest = left
            if right < w and times[right] < times[left]:
                smallest = right
            if times[idx] <= times[smallest]:
                break
            _heap_swap(times, rows, cols, ros, fli, src, idx, smallest)
            idx = smallest
        start -= 1
    return w


@njit(cache=False, parallel=True)
def draw_extinguish(
    probabilities: npt.NDArray[np.float32], realizations: int
) -> npt.NDArray[np.bool_]:
    """Outcome of an extinguish attempt per realization and attempted cell.

    ``probabilities`` lists the attempted cells (p > 0) in row-major order.
    Certain cells (p >= 1) draw nothing; the others draw one uniform each,
    realization by realization, from the kernels' per-thread RNGs. The loop
    over realizations is a ``prange`` like the one in
    :func:`advance_front_until`, so each realization draws on the same
    thread as its spread and a run seeded with ``Propagator.reseed`` stays
    reproducible (same machine and thread count).
    """
    n = probabilities.shape[0]
    out = np.zeros((realizations, n), dtype=np.bool_)
    for r in prange(realizations):
        for i in range(n):
            p = probabilities[i]
            if p >= 1.0:
                out[r, i] = True
            else:
                out[r, i] = np.random.random() < p
    return out
