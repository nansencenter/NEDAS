from abc import ABC, abstractmethod
from NEDAS import config, io_backends
from .context import Context
from .state import State
from .obs import Obs

class Scheme(ABC):
    """
    Runtime scheme base class.

    The Scheme coordinates all runtime generation and manipulation of objects.
    """
    c: Context

    def __init__(self, config: config.Config) -> None:
        # parse configuration
        self.c = Context(config)

        # initialize io backend
        self.c.io = io_backends.get_io_backend(self.c)

    @abstractmethod
    def __call__(self) -> None:
        """
        A runtime scheme must have a __call__ func.
        """
        ...
