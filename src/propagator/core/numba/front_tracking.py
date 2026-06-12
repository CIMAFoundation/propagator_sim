"""Numba-friendly front-tracking propagation kernel."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange  # type: ignore

from propagator.core.constants import NO_FUEL

from .propagation import (
    N_NEIGHBOURS,
    compute_spotting,
    try_spread_to_neighbour,
)


@njit(cache=False)
def _heap_swap(
    times: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32],
    ros: npt.NDArray[np.float32],
    fli: npt.NDArray[np.float32],
    i: int,
    j: int,
) -> None:
    times[i], times[j] = times[j], times[i]
    rows[i], rows[j] = rows[j], rows[i]
    cols[i], cols[j] = cols[j], cols[i]
    ros[i], ros[j] = ros[j], ros[i]
    fli[i], fli[j] = fli[j], fli[i]


@njit(cache=False)
def _heap_push(
    times: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32],
    ros: npt.NDArray[np.float32],
    fli: npt.NDArray[np.float32],
    size: int,
    time: int,
    row: int,
    col: int,
    ros_value: float,
    fli_value: float,
) -> int:
    times[size] = time
    rows[size] = row
    cols[size] = col
    ros[size] = ros_value
    fli[size] = fli_value

    idx = size
    while idx > 0:
        parent = (idx - 1) // 2
        if times[parent] <= times[idx]:
            break
        _heap_swap(times, rows, cols, ros, fli, parent, idx)
        idx = parent
    return size + 1


@njit(cache=False)
def _heap_pop_min(
    times: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32],
    ros: npt.NDArray[np.float32],
    fli: npt.NDArray[np.float32],
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
            _heap_swap(times, rows, cols, ros, fli, idx, smallest)
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
    sizes: npt.NDArray[np.int32],
    overflow: npt.NDArray[np.int8],
    cellsize: float,
    veg: npt.NDArray[np.integer],
    fuel_idx: npt.NDArray[np.int64],
    dem: npt.NDArray[np.floating],
    fire: npt.NDArray[np.int8],
    spotting_generation: npt.NDArray[np.bool_],
    spotting_receiving: npt.NDArray[np.bool_],
    state_arrival_time: npt.NDArray[np.int32],
    state_ros: npt.NDArray[np.float32],
    state_fli: npt.NDArray[np.float32],
    moisture: npt.NDArray[np.floating],
    wind_dir: npt.NDArray[np.floating],
    wind_speed: npt.NDArray[np.floating],
    fuels,
    p_time_fn,
    p_moist_fn,
    out_of_bounds: npt.NDArray[np.int8],
    track_spotting: bool,
) -> None:
    n_realizations = sizes.shape[0]
    n_rows = veg.shape[0]
    n_cols = veg.shape[1]

    for realization in prange(n_realizations):
        if overflow[realization] != 0:
            continue

        while sizes[realization] > 0:
            if event_times[realization, 0] > end_time:
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
                sizes[realization],
            )
            sizes[realization] = new_size

            if fire[realization, row, col] != 0:
                continue

            if row <= 0 or col <= 0 or row >= n_rows - 1 or col >= n_cols - 1:
                out_of_bounds[realization] = 1

            fire[realization, row, col] = 1
            state_arrival_time[realization, row, col] = time
            state_ros[realization, row, col] = ros_value
            state_fli[realization, row, col] = fli_value

            if veg[row, col] == NO_FUEL:
                continue

            fire_slice = fire[realization]
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
                    fire_slice,
                    moisture,
                    w_dir_r,
                    w_speed_r,  # type: ignore
                    fuels,
                    p_time_fn,
                    p_moist_fn,
                )
                if not ignites:
                    continue
                if sizes[realization] >= max_events:
                    overflow[realization] = 1
                    break
                sizes[realization] = _heap_push(
                    event_times[realization],
                    event_rows[realization],
                    event_cols[realization],
                    event_ros[realization],
                    event_fli[realization],
                    sizes[realization],
                    time + int(delta_time),
                    row_to,
                    col_to,
                    ros_to,
                    fli_to,
                )

            if fuels.spotting[fuel_idx[row, col]]:
                spotting_updates = compute_spotting(
                    row,
                    col,
                    cellsize,
                    veg,
                    fuel_idx,
                    fire_slice,
                    w_dir_r,
                    w_speed_r,  # type: ignore
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
                        spotting_generation[realization, row, col] = True
                        spotting_receiving[realization, row_to, col_to] = True
                    if sizes[realization] >= max_events:
                        overflow[realization] = 1
                        break
                    sizes[realization] = _heap_push(
                        event_times[realization],
                        event_rows[realization],
                        event_cols[realization],
                        event_ros[realization],
                        event_fli[realization],
                        sizes[realization],
                        time + int(delta_time),
                        row_to,
                        col_to,
                        ros_to,
                        fli_to,
                    )
