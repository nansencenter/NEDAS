from abc import ABC, abstractmethod
from NEDAS import assim_tools, config
from .context import Context

class Scheme(ABC):
    """
    Runtime scheme base class.

    The Scheme coordinates all runtime generation and manipulation of objects.
    """
    c: Context
    supported_io_modes: list[str] = ['online', 'offline']

    def __init__(self, config: config.Config):
        self.c = Context(config)

    @abstractmethod
    def __call__(self) -> None:
        """
        A runtime scheme must have a __call__ func.
        """
        pass
