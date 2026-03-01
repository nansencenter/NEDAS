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

    def save_debug_data(self, c: Context, filename: str, data: dict) -> None:
        file = os.path.join(c.config.work_dir, f"{filename}.npz")
        np.savez(file, **data)
