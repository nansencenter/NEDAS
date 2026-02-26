from abc import ABC, abstractmethod
import numpy as np
from .context import Context
from .types import FieldRecord

class IOBackend(ABC):
    """
    Base class for handling input/output for state and observation variables

    Attributes:
        tags (list[str]): List of names for copies of state/obs data

    """
    tags = ['fields_prior', 'fields_post', 'obs_seq', 'obs_prior', 'obs_post']

    @abstractmethod
    def __init__(self, c: Context):
        ...

    @abstractmethod
    def read_field(self, c: Context, rec: FieldRecord, member: int) -> np.ndarray:
        """
        Read a 2D field from the State

        Args:
            c (Context): the runtime context
            rec (FieldRecord): field record object
            member (int): ensemble member index from 0 to nens-1

        Returns:
            np.ndarray: the 2D field data
        """
        ...

    @abstractmethod
    def write_field(self, c: Context, fld: np.ndarray, rec: FieldRecord, member: int) -> None:
        """
        Write a 2D field to the State

        Args:
            c (Context): the runtime context
            fld (np.ndarray): the 2D field data
            rec (FieldRecord): field record object
            member (int): ensemble member index
        """
        ...

def get_io_backend(c: Context) -> IOBackend:
    """
    Factory function to return the correct IOBackend subclass instance.

    Args:
        c (Context): the runtime context

    Returns:
        IOBackend: Corresponding io backend subclass instance
    """
    cf = c.config
    if cf.io_mode == 'offline':
        from NEDAS.io_backends.file_io import FileIO
        return FileIO(c)

    elif cf.io_mode == 'online':
        from NEDAS.io_backends.memory_io import MemoryIO
        return MemoryIO(c)

    else:
        raise ValueError(f"Unsupported io_mode '{cf.io_mode}', only 'online' or 'offline'.")
