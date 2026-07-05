"""File-backed storage for frozen (inactive) state tiles.

Burned-out interior tiles — tiles where every in-domain cell is burnt or
fuel-free — never change again during propagation, but their per-cell
tracking (flags, arrival time, rate of spread, fireline intensity) is
still needed for outputs. Freezing them to disk keeps the in-memory
working set proportional to the active front while preserving full
interior tracking; SSD reads make retrieval cheap.

Records are fixed-size (one tile's four arrays back to back), keyed by
``(realization, world_row, world_col)`` of the tile's top-left cell in
world coordinates, so keys survive domain growth. Slots are reused when a
tile is re-frozen after a thaw, so the file never fragments. The store is
session-scoped: checkpoints thaw everything first, so nothing needs to
survive a restart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt

from propagator.core.numba import TILE_SIZE

TileKey = tuple[int, int, int]
TileRecord = tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]

_CELLS = TILE_SIZE * TILE_SIZE
_FLAGS_BYTES = _CELLS
_ARRIVAL_BYTES = _CELLS * 4
_ROS_BYTES = _CELLS * 4
_FLI_BYTES = _CELLS * 4
RECORD_SIZE = _FLAGS_BYTES + _ARRIVAL_BYTES + _ROS_BYTES + _FLI_BYTES


class TileStore:
    """Fixed-record file store for frozen tiles."""

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "frozen_tiles.bin"
        self._file = open(self.path, "w+b")
        # every record slot ever written (offsets are reused on re-freeze)
        self._slots: dict[TileKey, int] = {}
        # keys currently frozen -> record offset
        self._frozen: dict[TileKey, int] = {}
        self._end = 0

    def __len__(self) -> int:
        return len(self._frozen)

    def __contains__(self, key: TileKey) -> bool:
        return key in self._frozen

    def keys(self) -> Iterator[TileKey]:
        return iter(self._frozen)

    def freeze(
        self,
        key: TileKey,
        flags: npt.NDArray[np.uint8],
        arrival: npt.NDArray[np.int32],
        ros: npt.NDArray[np.float32],
        fli: npt.NDArray[np.float32],
    ) -> None:
        """Write one tile's state to disk and mark it frozen."""
        offset = self._slots.get(key)
        if offset is None:
            offset = self._end
            self._end += RECORD_SIZE
            self._slots[key] = offset
        self._file.seek(offset)
        self._file.write(flags.tobytes())
        self._file.write(arrival.astype(np.int32, copy=False).tobytes())
        self._file.write(ros.astype(np.float32, copy=False).tobytes())
        self._file.write(fli.astype(np.float32, copy=False).tobytes())
        self._frozen[key] = offset

    def read(self, key: TileKey) -> TileRecord:
        """Read a frozen tile's arrays without unfreezing it."""
        offset = self._frozen[key]
        self._file.seek(offset)
        buffer = self._file.read(RECORD_SIZE)
        shape = (TILE_SIZE, TILE_SIZE)
        flags = np.frombuffer(
            buffer, dtype=np.uint8, count=_CELLS, offset=0
        ).reshape(shape)
        arrival = np.frombuffer(
            buffer, dtype=np.int32, count=_CELLS, offset=_FLAGS_BYTES
        ).reshape(shape)
        ros = np.frombuffer(
            buffer,
            dtype=np.float32,
            count=_CELLS,
            offset=_FLAGS_BYTES + _ARRIVAL_BYTES,
        ).reshape(shape)
        fli = np.frombuffer(
            buffer,
            dtype=np.float32,
            count=_CELLS,
            offset=_FLAGS_BYTES + _ARRIVAL_BYTES + _ROS_BYTES,
        ).reshape(shape)
        return flags, arrival, ros, fli

    def thaw(self, key: TileKey) -> TileRecord:
        """Read a frozen tile's arrays and unfreeze it (writable copies).

        The record slot is kept for reuse if the tile is frozen again.
        """
        record = self.read(key)
        del self._frozen[key]
        return tuple(array.copy() for array in record)  # type: ignore[return-value]

    def clear(self) -> None:
        """Drop all frozen tiles and reclaim the file."""
        self._frozen.clear()
        self._slots.clear()
        self._end = 0
        self._file.seek(0)
        self._file.truncate(0)

    def close(self) -> None:
        self._file.close()
