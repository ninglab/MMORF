from .interrupt_manager import InterruptManager
class BaseSearch(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.interrupt_mgr = InterruptManager(parallel=kwargs.get("parallel_interrupts", 0))

    def __call__(self, mol, **kwargs):
        return NotImplemented