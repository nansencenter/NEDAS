from abc import ABC, abstractmethod
from NEDAS import config, io_backends
from NEDAS.io_backends.file_io import FileIO
from NEDAS.datasets.synthetic import SyntheticObs
from .context import Context
from .state import State
from .obs import Obs

class Scheme(ABC):
    """
    Runtime scheme base class.

    The Scheme coordinates all runtime generation and manipulation of objects.
    """
    c: Context
    offline: bool
    synthethic: bool

    def __init__(self, config: config.Config) -> None:
        # parse configuration
        self.c = Context(config)

        # initialize io backend
        self.c.io = io_backends.get_io_backend(self.c)

        self.c.state = State(self.c)
        self.c.obs = Obs(self.c)

        self.offline = isinstance(self.c.io, FileIO)
        self.synthetic = any([isinstance(d, SyntheticObs) for _,d in self.c.datasets.items()])

    @abstractmethod
    def __call__(self) -> None:
        """
        A runtime scheme must have a __call__ func.
        """
        ...
