import os
import inspect
import importlib
from typing import Literal, Generic, TypeVar, Type
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from NEDAS.config import parse_config
from NEDAS.grid import GridType

"""
Model module

    Model base class

    registry

    get_model_class
"""

GridT = TypeVar("GridT", bound=GridType)

class Model(Generic[GridT], ABC):
    """
    Class for configuring and running a model
    """
    io_mode: Literal['online', 'offline'] = 'offline'
    grid: GridT
    z_untis: str = '*'
    z = None
    variables: dict = {}
    run_process = None
    run_status = 'pending'

    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        ##get a list of values from default.yml and update with kwargs, save to config_dict
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

    def parse_kwargs(self, **kwargs):
        ##args that pinpoints a certain model state variable
        if 'path' not in kwargs:
            kwargs['path'] = '.'  ##default path is current dir

        if 'member' not in kwargs:
            kwargs['member'] = None
        if kwargs['member'] is not None:
            assert kwargs['member'] >= 0, f"member index should be >= 0, got {kwargs['member']}"

        if 'name' not in kwargs:
            kwargs['name'] = list(self.variables.keys())[0]  ##if not specified, use first variable listed
        assert kwargs['name'] in self.variables, f"'{kwargs['name']}' is not defined in model variables"

        if 'time' not in kwargs:
            kwargs['time'] = None
        if kwargs['time'] is not None:
            assert isinstance(kwargs['time'], datetime), "kwargs 'time' is expected to be a datetime object'"

        levels = list(self.variables[kwargs['name']]['levels'])
        if 'k' not in kwargs:
            kwargs['k'] = levels[0]  ##set to the first level if not specified
        assert kwargs['k'] in levels, f"level {kwargs['k']} is not available for variable {kwargs['name']}"

        if 'units' not in kwargs:
            kwargs['units'] = self.variables[kwargs['name']]['units']

        for key in ['restart_dir', 'forecast_period', 'comm']:
            if key not in kwargs:
                kwargs[key] = None

        return kwargs

    @abstractmethod
    def read_grid(self, **kwargs) -> None:
        """
        Read the grid information from the model output.

        Args:
            **kwargs: Keyword arguments for reading the grid.
        """
        pass

    @abstractmethod
    def read_var(self, **kwargs) -> np.ndarray:
        """
        Read a variable from the model output.

        Args:
            **kwargs: Keyword arguments for reading the variable.

        Returns:
            np.ndarray: The read variable.
        """
        pass

    @abstractmethod
    def write_var(self, var, **kwargs) -> None:
        """
        Write a variable to the model output.

        Args:
            var (np.ndarray): The variable to write.
            **kwargs: Keyword arguments for writing the variable.
        """
        pass

    @abstractmethod
    def z_coords(self, **kwargs) -> np.ndarray:
        """
        Get the vertical coordinates of the model.

        Args:
            **kwargs: Keyword arguments for getting the vertical coordinates.

        Returns:
            np.ndarray: The vertical coordinates.
        """
        pass

    @abstractmethod
    def preprocess(self, **kwargs) -> None:
        """
        Preprocess the model data.

        Args:
            **kwargs: Keyword arguments for preprocessing.
        """
        pass

    @abstractmethod
    def postprocess(self, **kwargs) -> None:
        """
        Postprocess the model data.

        Args:
            **kwargs: Keyword arguments for postprocessing.
        """
        pass

    @abstractmethod
    def run(self, **kwargs) -> None:
        """
        Run the model.

        Args:
            **kwargs: Keyword arguments for running the model.
        """
        pass

registry = {
    'lorenz96': 'Lorenz96Model',
    'qg.fortran': 'QGFortranModel',
    'qg.fortran.emulator': 'QGFortranModelEmulator',
    'vort2d': 'Vort2DModel',
    'topaz.v5': 'Topaz5Model',
    'nextsim.v1': 'NextsimModel',
    'nextsim.dg': 'NextsimDGModel',
    'wrf': 'WRFModel',
}

def get_model_class(model_name: str) -> Type["Model"]:
    """
    Factory function to return the correct Model subclass.

    Args:
        model_name (str): Model name

    Returns:
        Type["Model"]: Corresponding Model subclass
    """
    model_name = model_name.lower()

    if model_name not in registry.keys():
        raise NotImplementedError(f"Model class not implemented for '{model_name}'")

    module = importlib.import_module('NEDAS.models.'+model_name)
    ModelClass = getattr(module, registry[model_name])

    return ModelClass