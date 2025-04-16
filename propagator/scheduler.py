from sortedcontainers import SortedDict
import numpy as np


class Scheduler:
    """
    handles the scheduling of the propagation procedure
    """

    def __init__(self):
        self.list = SortedDict()

        # fix the change in SortedDict api
        self.list_kw = {"last": False}
        try:
            self.list.popitem(**self.list_kw)
        except KeyError:
            pass
        except TypeError:
            self.list_kw = {"index": 0}

    def push(self, coords, time):
        if time not in self.list:
            self.list[time] = []
        self.list[time].append(coords)

    def push_all(self, updates):
        for t, u in updates:
            self.push(u, t)

    def pop(self):
        item = self.list.popitem(**self.list_kw)
        return item

    def active(self):
        """
        get all the threads that have a scheduled update
        :return:
        """
        active_t = np.unique(
            [e for k in self.list.keys() for c in self.list[k] for e in c[:, 2]]
        )
        return active_t

    def __len__(self):
        return len(self.list)

    def next_time(self):
        """
        get the next time step
        :return:
        """
        if len(self) == 0:
            return None

        next_time, _next_updates = self.list.peekitem(index=0)
        return next_time

    def __call__(self):
        while len(self) > 0:
            c_time, updates = self.pop()
            print("u")
            new_updates = yield c_time, updates
            print("n")
            self.push_all(new_updates)
