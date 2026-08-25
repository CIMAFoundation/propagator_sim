"""Lightweight event scheduler for propagation updates.

Stores future updates grouped by simulation time and exposes utilities to push
events, pop the earliest batch, and inspect active realizations.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from propagator.core.models import UpdateBatch

PopResult = Tuple[int, "SchedulerEvent"]


@dataclass
class SchedulerEvent:
    """Represents a scheduled event in the simulation."""

    updates: UpdateBatch = field(default_factory=UpdateBatch)

    # boundary_conditions
    moisture: Optional[npt.NDArray[np.floating]] = None
    wind_dir: Optional[npt.NDArray[np.floating]] = None
    wind_speed: Optional[npt.NDArray[np.floating]] = None

    # actions
    additional_moisture: Optional[npt.NDArray[np.floating]] = None
    vegetation_changes: Optional[npt.NDArray[np.floating]] = None

    def update(self, other: SchedulerEvent) -> None:
        self.updates.extend(other.updates)

        # overwrite boundary_conditions if already set
        if other.moisture is not None:
            self.moisture = other.moisture

        if other.wind_dir is not None:
            self.wind_dir = other.wind_dir

        if other.wind_speed is not None:
            self.wind_speed = other.wind_speed

        # in this case changes are added
        if self.additional_moisture is None:
            if other.additional_moisture is not None:
                self.additional_moisture = other.additional_moisture
        elif other.additional_moisture is not None:
            self.additional_moisture += other.additional_moisture

        if self.vegetation_changes is None:
            self.vegetation_changes = other.vegetation_changes
        elif other.vegetation_changes is not None:
            # NaN means "no change" for vegetation_changes (see
            # Propagator._update_vegetation), not NO_FUEL (0, a real fuel
            # code) — comparing against NO_FUEL here would treat `other`'s
            # NaN cells as "set", wiping out `self`'s changes there.
            self.vegetation_changes = np.where(
                np.isnan(other.vegetation_changes),
                self.vegetation_changes,
                other.vegetation_changes,
            )


@dataclass(frozen=True)
class SortedDict:
    """Sorted dictionary optimized for time-ordered event scheduling.

    This data structure maintains events in sorted order by time (key) while
    allowing O(1) lookup and efficient insertion. It's specifically optimized
    for the common pattern of:
    1. Inserting events at various future times
    2. Always popping the earliest (minimum) event

    Performance characteristics:
    - Insertion: O(n) with bisect.insort vs O(n log n) with naive sort
    - Lookup: O(1) via dict
    - Pop earliest: O(1) via list indexing
    - Memory: O(n) for both dict and sorted list

    The frozen=True decorator prevents accidental modification of the
    data structure internals, enforcing use through the defined interface.
    """

    # Internal storage: dict for fast lookup, list for sorted order
    _data: Dict[int, SchedulerEvent] = field(
        default_factory=dict, init=False, repr=False
    )
    # Sorted list of keys (times) maintained in ascending order
    _order: List[int] = field(default_factory=list, init=False, repr=False)

    def __setitem__(self, key: int, value: SchedulerEvent) -> None:
        """Insert or update an event at a given time.

        Uses bisect.insort for O(n) insertion into the sorted list instead of
        append + sort, and the 'if key not in self._order' check prevents
        duplicate keys when updating an existing time slot.
        """
        self._data[key] = value
        if key not in self._order:
            # Binary search to find insertion point, then insert
            # Maintains sorted order without full resort
            bisect.insort(self._order, key)

    def __getitem__(self, key: int) -> SchedulerEvent:
        return self._data[key]

    def __delitem__(self, key: int) -> None:
        del self._data[key]
        self._order.remove(key)

    def __iter__(self) -> Iterator[int]:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._data)

    def get(
        self, key: int, default: Optional[SchedulerEvent] = None
    ) -> Optional[SchedulerEvent]:
        return self._data.get(key, default)

    def popitem(self, index: int) -> tuple[int, SchedulerEvent]:
        key = self._order.pop(index)
        value = self._data.pop(key)
        return key, value

    def values(self) -> Iterator[SchedulerEvent]:
        return iter(self._data.values())

    def items(self) -> Iterator[Tuple[int, SchedulerEvent]]:
        for key in self._order:
            yield key, self._data[key]

    def clear(self) -> None:
        self._data.clear()
        self._order.clear()

    def peekitem(self, index: int) -> Tuple[int, SchedulerEvent]:
        key = self._order[index]
        value = self._data[key]
        return key, value


@dataclass
class Scheduler:
    """
    Lightweight event scheduler for propagation updates.

    Generic over the time key type (int or float), so your inputs and outputs
    stay consistent.
    """

    realizations: int
    _queue: SortedDict = field(
        default_factory=SortedDict, init=False, repr=False
    )

    # --- Basic queue ops -----------------------------------------------------

    def pop(self) -> PopResult:
        if not self:
            raise IndexError("pop from empty Scheduler")
        time, updates = self._queue.popitem(index=0)
        return time, updates

    def add_event(self, time: int, event: SchedulerEvent):
        """
        Adds an event to the scheduler.

        Parameters
        ----------
        time : int
            Time for the event
        event : SchedulerEvent
            New event structure
        """
        entry = self._queue.get(time, None)
        if entry is None:
            self._queue[time] = event
        else:
            entry.update(event)

    def __len__(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self) == 0

    def clear(self) -> None:
        self._queue.clear()

    def next_time(self) -> Optional[int]:
        if not self:
            return None
        t, _ = self._queue.peekitem(index=0)
        return t  # type: ignore[return-value]

    # --- Iteration utilities -------------------------------------------------

    def iterate(self) -> Iterator[PopResult]:
        while self:
            yield self.pop()
