from abc import ABC, abstractmethod
from typing import Any, Callable
import os
import numpy as np
from .context import Context

class IOBackend(ABC):
    """
    Base class for handling input/output for state and observation variables

    Attributes:
        tags (list[str]): List of names for copies of state/obs data

    """
    tags: list[str] = ['prior', 'prior_mean', 'posterior', 'posterior_mean', 'truth', 'z']

    def __init__(self, c: Context):
        pass

    def validate_tag(self, tag: str):
        if tag not in self.tags:
            raise ValueError(f"IOBackend: unknown tag '{tag}', supported: {self.tags}")

    def prepare_collective_io(self, c: Context, tag: str) -> None:
        """
        Prepare for collective io, needed typically for offline io modes.

        Creating files for writing if not existing yet. Creating file locks for parallel io, etc.
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
    def call_io_method(self, c: Context, tag: str, method: Callable, *args, **kwargs) -> Any:
        """
        Call a method to perform some io tasks.

        Args:
            c (Context): the runtime context
            tag (str): which copy of the model state to request io from: "prior", "posterior" or "truth"
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
        os.makedirs(os.path.dirname(file), exist_ok=True)

        # save the data to file
        np.savez(file, **data)
