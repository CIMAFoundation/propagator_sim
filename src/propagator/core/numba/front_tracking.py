"""Numba-friendly front-tracking propagation kernel."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange  # type: ignore

from .propagation import single_cell_updates


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

            if fire[row, col, realization] != 0:
                continue

            # Reaching the outer frame of the domain (not an actual
            # out-of-array index: single_cell_updates() already discards
            # neighbours outside [0, shape) before they reach the heap) is
            # flagged as a deliberate early warning that the fire may spread
            # past the simulated area, so callers can enlarge the domain.
            # See out_of_bounds_mode on Propagator.
            if row <= 0 or col <= 0 or row >= n_rows - 1 or col >= n_cols - 1:
                out_of_bounds[realization] = 1

            fire[row, col, realization] = 1
            state_arrival_time[row, col, realization] = time
            state_ros[row, col, realization] = ros_value
            state_fli[row, col, realization] = fli_value

            updates = single_cell_updates(
                row,
                col,
                cellsize,
                veg,
                dem,
                fire[:, :, realization],
                moisture,
                wind_dir,
                wind_speed,  # type: ignore
                fuels,
                p_time_fn,
                p_moist_fn,
                fli_value,
            )

            for update in updates:
                (delta_time, row_to, col_to, ros_to, fli_to, is_spotting) = (
                    update
                )
                if track_spotting and is_spotting:
                    spotting_generation[row, col, realization] = True
                    spotting_receiving[row_to, col_to, realization] = True
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
