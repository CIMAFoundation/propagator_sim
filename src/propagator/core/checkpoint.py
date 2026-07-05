"""Checkpointing for the propagator state.

A :class:`PropagatorCheckpoint` is a self-contained, picklable snapshot of
every dynamic quantity of a running simulation: simulation clock, front
event heaps, block-sparse tile state, weather fields, (possibly mutated)
vegetation and the pending boundary-condition queue.

Checkpoints are anchored in a *world* cell coordinate system: `origin` is
the world (row, col) of the grid's local cell (0, 0). This allows a
checkpoint taken on one grid to be restored onto a larger grid with a
different origin (see ``Propagator.from_checkpoint``), enabling domain
growth when the fire reaches a boundary.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from propagator.core.scheduler import SchedulerEvent
from propagator.core.tile_store import (
    RECORD_SIZE,
    TileIndex,
    TileStore,
    iter_records,
)

# version 2 adds incremental frozen-tile references (sidecar .tiles file
# on disk); version 1 files load fine with an empty frozen index
CHECKPOINT_VERSION = 2

# Array fields serialized directly as npz entries (None allowed for the
# optional weather/actions fields).
_ARRAY_FIELDS = (
    "veg",
    "dem",
    "moisture",
    "wind_dir",
    "wind_speed",
    "actions_moisture",
    "front_times",
    "front_rows",
    "front_cols",
    "front_ros",
    "front_fli",
    "front_sizes",
    "tile_idx",
    "tile_counts",
    "tile_flags",
    "tile_arrival",
    "tile_ros",
    "tile_fli",
)


@dataclass(frozen=True, eq=False)
class PropagatorCheckpoint:
    """Immutable snapshot of a Propagator's dynamic state.

    Front heap arrays are trimmed to the largest per-realization heap size
    and tile pools to the largest per-realization tile count, so checkpoint
    size scales with the active front and burned area, not with allocated
    capacity.
    """

    version: int
    time: int
    origin: tuple[int, int]
    cellsize: float
    realizations: int
    do_spotting: bool

    # inputs that can mutate during the run (vegetation_changes) or are
    # needed to rebuild the same domain from file
    veg: npt.NDArray[np.integer]
    dem: npt.NDArray[np.floating]

    # environmental fields (None if never set)
    moisture: npt.NDArray[np.floating] | None
    wind_dir: npt.NDArray[np.floating] | None
    wind_speed: npt.NDArray[np.floating] | None
    actions_moisture: npt.NDArray[np.floating] | None

    # front event heaps, trimmed to max(front_sizes) slots
    front_times: npt.NDArray[np.int32]
    front_rows: npt.NDArray[np.int32]
    front_cols: npt.NDArray[np.int32]
    front_ros: npt.NDArray[np.float32]
    front_fli: npt.NDArray[np.float32]
    front_sizes: npt.NDArray[np.int32]

    # block-sparse tile state, pools trimmed to max(tile_counts) tiles
    tile_idx: npt.NDArray[np.int32]
    tile_counts: npt.NDArray[np.int32]
    tile_flags: npt.NDArray[np.uint8]
    tile_arrival: npt.NDArray[np.int32]
    tile_ros: npt.NDArray[np.float32]
    tile_fli: npt.NDArray[np.float32]

    # pending boundary-condition queue as (time, event) pairs
    scheduler_events: list[tuple[int, SchedulerEvent]]

    # incremental frozen tiles: {key: record offset} into frozen_source,
    # which is the live session TileStore (fresh checkpoints) or the
    # sidecar record file written by save() (loaded checkpoints). Frozen
    # tiles are NOT copied into the checkpoint, so snapshotting a run
    # with a huge frozen interior stays cheap.
    frozen_index: TileIndex = field(default_factory=dict)
    frozen_source: TileStore | Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (rows, cols) the checkpoint was taken on."""
        return self.veg.shape  # type: ignore[return-value]

    def world_bounds(self) -> tuple[int, int, int, int]:
        """Inclusive world-cell bounds (row0, col0, row1, col1)."""
        rows, cols = self.shape
        return (
            self.origin[0],
            self.origin[1],
            self.origin[0] + rows - 1,
            self.origin[1] + cols - 1,
        )

    @staticmethod
    def _normalize_path(path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        return path

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        return path.with_suffix(".tiles")

    def save(self, path: str | Path) -> None:
        """Persist the checkpoint to a compressed ``.npz`` file.

        The scheduler queue is pickled into an object entry; everything
        else is stored as plain arrays. Frozen tile records are streamed
        (memory-bounded) into a sidecar ``.tiles`` file next to the
        archive, which must be kept alongside it.
        """
        path = self._normalize_path(path)
        extra: dict[str, npt.NDArray] = {}
        if self.frozen_index:
            if self.frozen_source is None:
                raise ValueError(
                    "Checkpoint references frozen tiles but has no source"
                )
            keys = []
            offsets = []
            with open(self._sidecar_path(path), "wb") as sidecar:
                for key, payload in iter_records(
                    self.frozen_source, self.frozen_index
                ):
                    keys.append(key)
                    offsets.append(sidecar.tell())
                    sidecar.write(payload)
            extra["frozen_keys"] = np.asarray(keys, dtype=np.int64)
            extra["frozen_offsets"] = np.asarray(offsets, dtype=np.int64)

        arrays = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        present = {k: v for k, v in arrays.items() if v is not None}
        np.savez_compressed(
            path,
            version=np.int64(self.version),
            time=np.int64(self.time),
            origin=np.asarray(self.origin, dtype=np.int64),
            cellsize=np.float64(self.cellsize),
            realizations=np.int64(self.realizations),
            do_spotting=np.bool_(self.do_spotting),
            scheduler_events=np.frombuffer(
                pickle.dumps(self.scheduler_events), dtype=np.uint8
            ),
            **extra,
            **present,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PropagatorCheckpoint":
        """Load a checkpoint previously written by :meth:`save`.

        Checkpoints with frozen tiles read their records lazily from the
        sidecar ``.tiles`` file, which must sit next to the archive.
        """
        path = cls._normalize_path(path)
        with np.load(path) as data:
            version = int(data["version"])
            if version > CHECKPOINT_VERSION:
                raise ValueError(
                    f"Checkpoint version {version} is newer than supported "
                    f"version {CHECKPOINT_VERSION}"
                )
            arrays = {
                name: (data[name] if name in data.files else None)
                for name in _ARRAY_FIELDS
            }
            scheduler_events = pickle.loads(data["scheduler_events"].tobytes())

            frozen_index: TileIndex = {}
            frozen_source: Path | None = None
            if "frozen_keys" in data.files:
                keys = data["frozen_keys"]
                offsets = data["frozen_offsets"]
                frozen_index = {
                    (int(k[0]), int(k[1]), int(k[2])): int(offset)
                    for k, offset in zip(keys, offsets)
                }
                if frozen_index:
                    frozen_source = cls._sidecar_path(path)
                    expected = len(frozen_index) * RECORD_SIZE
                    if (
                        not frozen_source.exists()
                        or frozen_source.stat().st_size < expected
                    ):
                        raise ValueError(
                            f"Checkpoint sidecar {frozen_source} is missing "
                            "or truncated; it must be kept next to the "
                            ".npz archive."
                        )

            return cls(
                version=version,
                time=int(data["time"]),
                origin=(int(data["origin"][0]), int(data["origin"][1])),
                cellsize=float(data["cellsize"]),
                realizations=int(data["realizations"]),
                do_spotting=bool(data["do_spotting"]),
                scheduler_events=scheduler_events,
                frozen_index=frozen_index,
                frozen_source=frozen_source,
                **arrays,
            )


def reanchor_event(
    event: SchedulerEvent,
    row_shift: int,
    col_shift: int,
    old_shape: tuple[int, int],
    new_shape: tuple[int, int],
) -> SchedulerEvent:
    """Rewrite a pending scheduler event (in place) for a re-anchored grid.

    Update coordinates are shifted; 2D condition fields are padded to the
    new grid shape — weather fields replicate their edge values, action
    moisture pads with zeros and vegetation changes with NaN (no change).
    The event must already be a private copy (see ``Scheduler.snapshot``).
    """
    if old_shape == new_shape and row_shift == 0 and col_shift == 0:
        return event

    updates = event.updates
    updates.rows = updates.rows + row_shift
    updates.cols = updates.cols + col_shift
    object.__setattr__(updates, "bbox", None)
    object.__setattr__(updates, "_bbox_computed", False)

    pad = (
        (row_shift, new_shape[0] - old_shape[0] - row_shift),
        (col_shift, new_shape[1] - old_shape[1] - col_shift),
    )
    for name in ("moisture", "wind_dir", "wind_speed"):
        value = getattr(event, name)
        if value is not None:
            setattr(event, name, np.pad(value, pad, mode="edge"))
    if event.additional_moisture is not None:
        event.additional_moisture = np.pad(
            event.additional_moisture, pad, mode="constant", constant_values=0
        )
    if event.vegetation_changes is not None:
        event.vegetation_changes = np.pad(
            event.vegetation_changes,
            pad,
            mode="constant",
            constant_values=np.nan,
        )
    return event
