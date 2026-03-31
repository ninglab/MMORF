from typing import Union, Dict
from multiprocessing import Pool

class InterruptManager(object):
    def __init__(self, parallel : int = 0):
        self.parallel = parallel
        if self.parallel > 1:
            self.pool = Pool(processes=self.parallel)
        self.signals = {}
        self.schedule = {}

    def register(self, signal, callback, scheduler=None):
        if scheduler is None:
            scheduler = lambda x: True
        if signal not in self.signals:
            self.signals[signal] = []
        self.signals[signal].append((callback, scheduler))

    def emit(self, signal, *args, **kwargs) -> bool:
        self.schedule[signal] = self.schedule.get(signal, 0) + 1
        if signal in self.signals:
            cb_list = [sig[0] for sig in self.signals[signal] if sig[1](self.schedule[signal])]
            if self.parallel > 1:
                return all(self.pool.map(lambda cb: cb(*args, **kwargs), cb_list))
            return all([callback(*args, **kwargs) for callback in cb_list])
        return True