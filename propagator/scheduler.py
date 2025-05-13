from dataclasses import dataclass, field
from sortedcontainers import SortedDict
from .events import EventData
from threading import Lock

@dataclass
class Scheduler:
    """
    Scheduler class for managing the timing of events.
    """
    events: SortedDict[int, list[EventData]] = field(default_factory=SortedDict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def add(self, event: EventData):
         with self._lock:
            if event.time not in self.events:
                events_at_time = []
                self.events[event.time] = events_at_time
            else:
                events_at_time = self.events[event.time]
            events_at_time.append(event)

    def pop(self) -> tuple[int, list[EventData]]:
        with self._lock:        
            time, events = self.events.popitem(index=0)
            return time, events

    def __len__(self):
        return len(self.events)

    def next_time(self):
        """
        get the next time step
        :return:
        """
        with self._lock:        
            if len(self) == 0:
                return None

            next_time, _next_events = self.events.peekitem(index=0)
            return next_time
    
    def peek(self) -> tuple[int, list[EventData]] | None:
        with self._lock:
            if not self.events:
                return None
            return self.events.peekitem(index=0)

    def next_events_times(self) -> list[int]:
        """
        get the list of scheduld events times
        :return: list of times
        """
        with self._lock:        
            if len(self) == 0:
                return None

            times = self.events.keys()
            return times
