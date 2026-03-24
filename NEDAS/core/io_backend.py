from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, get_args, TYPE_CHECKING
import os
import numpy as np
from .types import IOMode, IOTag
if TYPE_CHECKING:
    from .context import Context

class IOBackend(ABC):
    """
    Base class for handling runtime input/output for state and observation variables
    and OS-level operation such as file manipulation and running commands

    Attributes:
        io_mode (IOMode): 'offline' for file I/O and 'online' for persistent memory I/O
        tags (list[str]): List of names for copies of state/obs data

    """
    io_mode: IOMode = 'offline'
    tags: list[str] = list(get_args(IOTag))

    def validate_tag(self, tag: str):
        if tag not in self.tags:
            raise ValueError(f"IOBackend: unknown tag '{tag}', supported: {self.tags}")

    def prepare_fields_storage(self, c: Context, tag: str) -> None:
        """
        Prepare for storage of fields data.
        Only needed for offline io modes: initialize the binary file that stores fields
        and write its metadata.
        """
        pass

    @abstractmethod
    def read_field(self, c: Context, tag: str, rec_id: int, mem_id: int) -> np.ndarray:
        """
        Read a 2D field data from the state

        Args:
            c (Context): the runtime context
            tag (str): which copy of the state to read from
            rec_id (int): field record id
            mem_id (int): ensemble member index from 0 to nens-1

        Returns:
            np.ndarray: the 2D field data
        """
        ...

    @abstractmethod
    def write_field(self, fld: np.ndarray, c: Context, tag: str, rec_id: int, mem_id: int) -> None:
        """
        Write a 2D field data to the state

        Args:
            fld (np.ndarray): the 2D field data
            c (Context): the runtime context
            tag (str): which copy of the state to write to
            rec_id (int): field record object
            mem_id (int): ensemble member index
        """
        ...

    @abstractmethod
    def call_method(self, c: Context, tag: str, method: Callable, *args, **kwargs) -> Any:
        """
        Call a method to perform some tasks.

        Args:
            c (Context): the runtime context
            tag (str): which copy of the model state to request io from: "prior", "post" or "truth"
            method (Callable): method name
            *args, **kwargs: will be passed to the method

        Returns:
            Any: whatever the method(**kwargs) returns
        """
        ...

    @abstractmethod
    def save_ndarray(self, c: Context, name: str, data: np.ndarray, path: str | None=None) -> None:
        """
        Save ndarray data

        Args:
            c (Context): the runtime context
            name (str): the name of the data
            data (np.ndarray): the data
            path (str, optional): system path to save the data to.
        """
        ...

    @abstractmethod
    def load_ndarray(self, c: Context, name: str, path: str | None=None) -> np.ndarray | None:
        """
        Load ndarray from saved data
        
        Args:
            c (Context): the runtime context
            name (str): the name of the data
            path (str, optional): system path to the saved data.

        Returns:
            np.ndarray: the data
        """
        ...

    def save_debug_data(self, c: Context, name: str, data: dict, path: str | None=None) -> None:
        """
        Save debug data in npz format

        Args:
            c (Context): the runtime context
            name (str): the name of the data
            data (dict): the data
            path (str, optional): system path to save the data to.
        """
        if path is None:
            path = c.config.work_dir  # default path

        file = os.path.join(path, f"{name}.npz")

        # make sure directory exists
        c.fs.make_dir(os.path.dirname(file))

        # save the data to file
        np.savez(file, **data)

        print(f"saved debug data '{file}'", flush=True)
