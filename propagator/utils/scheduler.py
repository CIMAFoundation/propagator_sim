import numpy as np
from sortedcontainers import SortedDict

class Scheduler:
    """
    handles the scheduling of the propagation procedure
    """

    def __init__(self):
        self.list = SortedDict()

        # fix the change in SortedDict api
        self.list_kw = {'last': False}
        try:
            self.list.popitem(**self.list_kw)
        except KeyError:
            pass
        except TypeError:
            self.list_kw = {'index': 0}

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
        active_t = np.unique([e for k in self.list.keys() for c in self.list[k] for e in c[:, 2]])
        return active_t

    def len(self):
        return len(self.list)

    def __len__(self):
        return len(self.list)

    def __call__(self):
        while len(self)>0:
            c_time, updates = self.pop()
            print('u')
            new_updates = yield c_time, updates
            print('n')
            self.push_all(new_updates)

