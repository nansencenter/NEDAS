from abc import ABC, abstractmethod
from typing import Any
import os
import numpy as np
from .context import Context
from .types import FieldRecord

class IOBackend(ABC):
    """
    Base class for handling input/output for state and observation variables

    Attributes:
        tags (list[str]): List of names for copies of state/obs data

    """
    tags = ['prior', 'prior_mean', 'post', 'post_mean', 'z_coords']

    def __init__(self, c: Context):
        pass

    @abstractmethod
    def read_field(self, c: Context, tag: str, rec: FieldRecord, member: int) -> np.ndarray:
        """
        Read a 2D field data from the state

        Args:
            c (Context): the runtime context
            tag (str): which copy of the state to read from
            rec (FieldRecord): field record
            member (int): ensemble member index from 0 to nens-1

        Returns:
            np.ndarray: the 2D field data
        """
        ...

    @abstractmethod
    def write_field(self, fld: np.ndarray, c: Context, tag: str, rec: FieldRecord, member: int) -> None:
        """
        Write a 2D field data to the state

        Args:
            fld (np.ndarray): the 2D field data
            c (Context): the runtime context
            tag (str): which copy of the state to write to
            rec (FieldRecord): field record object
            member (int): ensemble member index
        """
        ...

    @abstractmethod
    def call_model_io(self, c: Context, model_name: str, method: str, **kwargs) -> Any:
        """
        Call a model class method to perform some io tasks.

        Args:
            c (Context): the runtime context
            model_name (str): the model module name
            method (str): method name
            **kwargs: will be passed to the method

        Returns:
            Any: whatever the getattr(model, method)(**kwargs) returns
        """
        ...

    def save_debug_data(self, c: Context, filename: str, data: dict) -> None:
        file = os.path.join(c.config.work_dir, f"{filename}.npz")
        np.savez(file, **data)
