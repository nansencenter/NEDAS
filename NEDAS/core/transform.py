from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import importlib
import numpy as np
from NEDAS.utils.conversion import ensure_list
from .types import FieldRecord, ObsRecord
if TYPE_CHECKING:
    from .context import Context

"""
Transform functions
"""

registry = {
    'identity': 'Identity',
    'scale_bandpass': 'ScaleBandpass',
}

class Transform(ABC):
    """
    Base class for miscellaneous transform functions
    """
    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def forward_state(self, c: Context, rec: FieldRecord, field: np.ndarray) -> np.ndarray:
        """
        Forward transform for the model state variables.

        Args:
            c (Context): the runtime context.
            rec (FieldRecord): State information record for the variable.
            field (np.ndarray): The state variable.

        Returns:
            np.ndarray: The transformed state variable.
        """
        ...

    @abstractmethod
    def backward_state(self, c: Context, rec: FieldRecord, field: np.ndarray) -> np.ndarray:
        """
        Backward (inverse) transform for the model state variables.

        Args:
            c (Context): the runtime context.
            rec (FieldRecord): State information record for the variable.
            field (np.ndarray): The transformed state variable.

        Returns:
            np.ndarray: The state variable transformed back to the original space.
        """
        ...

    @abstractmethod
    def forward_obs(self, c: Context, obs_rec: ObsRecord, obs_seq: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Forward transform for the observation sequence.

        Args:
            c (Context): the runtime context.
            obs_rec (ObsRecord): Observation information record.
            obs_seq (dict[str, np.ndarray]): The observation sequence.
                With keys ``'obs'`` the observation;
                ``'x', 'y', 'z', 't'`` the space/time coordinates;
                and ``'err_std'`` the observation errors.

        Returns:
            dict[str, np.ndarray]: The transformed observation sequence.
        """
        ...

    @abstractmethod
    def backward_obs(self, c: Context, obs_rec: ObsRecord, obs_seq: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Backward (inverse) transform for the observation sequence.

        Args:
            c (Context): the runtime context.
            obs_rFieldRecord)ict): Observation information record.
            obs_seq (dict): The transformed observation sequence.
                With keys ``'obs'`` the observation;
                ``'x', 'y', 'z', 't'`` the space/time coordinates;
                and ``'err_std'`` the observation errors.

        Returns:
            dict: The observation sequence transformed back to the original space.
        """
        ...


def get_transform_funcs(c: Context) -> list[Transform]:
    if c.config.transform_def is None:
        c.config.transform_def = {'type':'identity'}
    
    transform_funcs = []
    for transform_func_def in ensure_list(c.config.transform_def):

        if 'type' not in transform_func_def.keys():
            raise KeyError("'type' needs to be specified in transform_def entries")
        transform_func_type = transform_func_def['type'].lower()

        if transform_func_type not in registry.keys():
            raise NotImplementedError("Transform function type '{transform_func_type}' is not implemented.")
        
        module = importlib.import_module('NEDAS.assim_tools.transforms.'+transform_func_type)
        TransformClass = getattr(module, registry[transform_func_type])
        transform_func = TransformClass(**transform_func_def)
        transform_funcs.append(transform_func)

    return transform_funcs
