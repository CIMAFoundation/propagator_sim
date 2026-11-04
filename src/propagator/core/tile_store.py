"""File-backed storage for frozen (inactive) state tiles.

Burned-out interior tiles — tiles where propagation can provably never
ignite anything again — never change during propagation, but their
per-cell tracking (flags, arrival time, rate of spread, fireline
intensity) is still needed for outputs. Freezing them to disk keeps the
in-memory working set proportional to the active front while preserving
full interior tracking; SSD reads make retrieval cheap.

Records are fixed-size (one tile's four arrays back to back), keyed by
``(realization, world_row, world_col)`` of the tile's top-left cell in
world coordinates, so keys survive domain growth. Records are immutable
and append-only within a session: re-freezing a thawed tile appends a
new record instead of overwriting, so the ``{key: offset}`` index a
checkpoint captures stays valid for incremental checkpointing. The file
is reclaimed only by ``clear()`` (which invalidates any checkpoint still
referencing this store) or by ending the session — checkpoints persisted
with ``PropagatorCheckpoint.save`` copy their records to a sidecar file
and do not need the store to survive a restart.
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
TileIndex = dict[TileKey, int]

_CELLS = TILE_SIZE * TILE_SIZE
_FLAGS_BYTES = _CELLS
_ARRIVAL_BYTES = _CELLS * 4
_ROS_BYTES = _CELLS * 4
_FLI_BYTES = _CELLS * 4
RECORD_SIZE = _FLAGS_BYTES + _ARRIVAL_BYTES + _ROS_BYTES + _FLI_BYTES


def parse_record(buffer: bytes) -> TileRecord:
    """Decode one raw record into its four (TILE_SIZE, TILE_SIZE) arrays.

    The returned arrays are read-only views over `buffer`.
    """
    shape = (TILE_SIZE, TILE_SIZE)
    flags = np.frombuffer(buffer, np.uint8, _CELLS, 0).reshape(shape)
    arrival = np.frombuffer(buffer, np.int32, _CELLS, _FLAGS_BYTES).reshape(
        shape
    )
    ros = np.frombuffer(
        buffer, np.float32, _CELLS, _FLAGS_BYTES + _ARRIVAL_BYTES
    ).reshape(shape)
    fli = np.frombuffer(
        buffer,
        np.float32,
        _CELLS,
        _FLAGS_BYTES + _ARRIVAL_BYTES + _ROS_BYTES,
    ).reshape(shape)
    return flags, arrival, ros, fli


def iter_records(
    source: "TileStore | str | Path", index: TileIndex
) -> Iterator[tuple[TileKey, bytes]]:
    """Yield (key, raw record bytes) for every entry of an index.

    `source` is either a live TileStore or the path of a record file
    written by ``PropagatorCheckpoint.save`` (same fixed-record layout).
    """
    if isinstance(source, TileStore):
        for key, offset in index.items():
            yield key, source.read_bytes(offset)
    else:
        with open(source, "rb") as records:
            for key, offset in index.items():
                records.seek(offset)
                yield key, records.read(RECORD_SIZE)


class TileStore:
    """Append-only fixed-record file store for frozen tiles."""

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "frozen_tiles.bin"
        self._file = open(self.path, "w+b")
        # keys currently frozen -> record offset (records themselves are
        # immutable; older offsets stay readable for checkpoints)
        self._frozen: TileIndex = {}
        self._end = 0

    def __len__(self) -> int:
        return len(self._frozen)

    def __contains__(self, key: TileKey) -> bool:
        return key in self._frozen

    def keys(self) -> Iterator[TileKey]:
        return iter(self._frozen)

    def _append(self, key: TileKey, payload: bytes) -> None:
        offset = self._end
        self._file.seek(offset)
        self._file.write(payload)
        self._end = offset + RECORD_SIZE
        self._frozen[key] = offset

    def freeze(
        self,
        key: TileKey,
        flags: npt.NDArray[np.uint8],
        arrival: npt.NDArray[np.int32],
        ros: npt.NDArray[np.float32],
        fli: npt.NDArray[np.float32],
    ) -> None:
        """Append one tile's state to disk and mark it frozen."""
        self._append(
            key,
            flags.tobytes()
            + arrival.astype(np.int32, copy=False).tobytes()
            + ros.astype(np.float32, copy=False).tobytes()
            + fli.astype(np.float32, copy=False).tobytes(),
        )

    def write_record(self, key: TileKey, payload: bytes) -> None:
        """Append a raw record (e.g. imported from another store)."""
        if len(payload) != RECORD_SIZE:
            raise ValueError("invalid tile record size")
        self._append(key, payload)

    def read_bytes(self, offset: int) -> bytes:
        """Read one raw record at a known offset."""
        self._file.seek(offset)
        return self._file.read(RECORD_SIZE)

    def read(self, key: TileKey) -> TileRecord:
        """Read a frozen tile's arrays without unfreezing it."""
        return parse_record(self.read_bytes(self._frozen[key]))

    def thaw(self, key: TileKey) -> TileRecord:
        """Read a frozen tile's arrays and unfreeze it (writable copies).

        The record stays in the file so checkpoint indices that
        reference it remain valid.
        """
        record = self.read(key)
        del self._frozen[key]
        return tuple(array.copy() for array in record)  # type: ignore[return-value]

    def snapshot_index(self) -> TileIndex:
        """Copy of the current {key: offset} index (for checkpoints)."""
        return dict(self._frozen)

    def restore_index(self, index: TileIndex) -> None:
        """Replace the frozen index (rollback to a checkpoint's view)."""
        self._frozen = dict(index)

    def clear(self) -> None:
        """Drop all frozen tiles and reclaim the file.

        Invalidates any incremental checkpoint that still references
        this store; only call when no such checkpoint will be used.
        """
        self._frozen.clear()
        self._end = 0
        self._file.seek(0)
        self._file.truncate(0)

    def close(self) -> None:
        self._file.close()
