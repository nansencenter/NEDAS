import os
import inspect
import numpy as np
from typing import Optional, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from NEDAS.config import parse_config
from .types import VarName, VarDesc, LevelID

class Dataset(ABC):
    """
    Dataset class (template for specific dataset sources)
    """
    variables: dict[VarName, VarDesc] = {}
    obs_operator: dict[VarName, Callable] = {}

    def  __init__(self, config_file: Optional[str]=None, parse_args: Optional[bool]=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        if not hasattr(self, 'dataset_dir'):
            self.dataset_dir = '.'

        if not hasattr(self, 'obs_window_min'):
            self.obs_window_min = 0
        if not hasattr(self, 'obs_window_max'):
            self.obs_window_max = 0

        self.variables = {}

    def parse_kwargs(self, **kwargs):
        """
        Parse the input kwargs to pinpoint a specific file/variable...
        """
        ##args to pinpoint a certain observatino record, used by read_obs, etc.
        if 'path' not in kwargs:
            kwargs['path'] = self.dataset_dir

        if 'name' not in kwargs:
            kwargs['name'] = list(self.variables.keys())[0]
        assert kwargs['name'] in self.variables, f"'{kwargs['name']}' is not defined in dataset variables"

        if 'time' not in kwargs:
            kwargs['time'] = None
        if kwargs['time'] is not None:
            assert isinstance(kwargs['time'], datetime), "kwargs 'time' is not a datetime object"

        if 'units' not in kwargs:
            kwargs['units'] = self.variables[kwargs['name']].units   ##TODO: potential key error here if variables is not defined.

        # other args, set default values if not specified
        ##TODO: how to type hint these runtime kwargs?
        # model (Model): model class instance
        # grid (GridType): target grid
        # mask (np.ndarray): target grid mask (True if grid point is not part of the state)
        # z (dict[LevelID, np.ndarray]): z coordinates at each level k
        for key in ['model', 'grid', 'mask', 'z']:
            if key not in kwargs:
                kwargs[key] = None
        # nobs (int): number of observations
        # obs_window_min (int)
        # obs_window_max (int)  ##TODO: maybe setting them both to 0 is incorrect? need to iterate from min to max...
        for key in ['nobs', 'obs_window_min', 'obs_window_max']:
            if key not in kwargs:
                kwargs[key] = 0

        return kwargs

    @abstractmethod
    def read_obs(self, **kwargs) -> dict[str, np.ndarray]:
        """
        Return observation sequence matching the given kwargs
        """
        obs_seq = {
            'obs':np.array([]),
            't':np.array([]),
            'z':np.array([]),
            'y':np.array([]),
            'x':np.array([]),
            'err_std':np.array([]),
        }
        return obs_seq

    def random_network(self, **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError(f"'random_network is not implemented for {self.__class__.__name__}")
