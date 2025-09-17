"""Lightweight event scheduler for propagation updates.

Stores future updates grouped by simulation time and exposes utilities to push
events, pop the earliest batch, and inspect active realizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from propagator.core.constants import NO_FUEL
from propagator.core.models import (
    UpdateBatch,
    UpdateBatchWithTime,
)

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
            self.vegetation_changes = np.where(
                other.vegetation_changes == NO_FUEL,
                self.vegetation_changes,
                other.vegetation_changes,
            )


@dataclass(frozen=True)
class SortedDict:
    """Represents a sorted dictionary for scheduling events."""

    _data: Dict[int, SchedulerEvent] = field(
        default_factory=dict, init=False, repr=False
    )
    _order: List[int] = field(default_factory=list, init=False, repr=False)

    def __setitem__(self, key: int, value: SchedulerEvent) -> None:
        self._data[key] = value
        self._order.append(key)
        self._order.sort()

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

    def push_updates(self, updates: UpdateBatchWithTime) -> None:
        event: SchedulerEvent | None
        updates_at_time_dict = updates.split_by_time()

        for time, update in updates_at_time_dict.items():
            if time in self._queue:
                event = self._queue.get(time)
            else:
                event = SchedulerEvent()
                self._queue[time] = event
            if event is None:
                raise ValueError("SchedulerEvent should not be None here")

            event.updates.extend(update)

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

    def active(self) -> npt.NDArray[np.integer]:
        arrays = [event.updates.realizations for event in self._queue.values()]
        stacked = np.concatenate(arrays)
        return np.unique(stacked)

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
