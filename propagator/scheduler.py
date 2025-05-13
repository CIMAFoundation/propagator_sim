from dataclasses import dataclass, field
from sortedcontainers import SortedDict
from .events import EventData


@dataclass(frozen=True)
class Scheduler:
    """
    handles the scheduling of the propagation procedure
    """
    events: SortedDict[int, list[EventData]] = field(default_factory=SortedDict)

    def add(self, event: EventData):
        if event.time not in self.events:
            events_at_time = []
            self.events[event.time] = events_at_time
        else:
            events_at_time = self.events[event.time]
        events_at_time.append(event)

    def pop(self) -> tuple[int, list[EventData]]:
        time, events = self.events.popitem(index=0)
        return time, events

    # def active(self):
    #     """
    #     get all the threads that have a scheduled update
    #     :return:
    #     """
    #     active_t = np.unique(
    #         [e for k in self.updates.keys() for c in self.updates[k] for e in c[:, 2]]
    #     )
    #     return active_t

    def __len__(self):
        return len(self.updates)

    def next_time(self):
        """
        get the next time step
        :return:
        """
        if len(self) == 0:
            return None

        next_time, _next_events = self.events.peekitem(index=0)
        return next_time

