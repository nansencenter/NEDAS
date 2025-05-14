import os
import inspect
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from NEDAS.config import parse_config

class Model(ABC):
    """
    Class for configuring and running a model
    """
    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        ##get a list of values from default.yml and update with kwargs, save to config_dict
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.grid = None
        self.z = None

        self.variables = {}

        self.run_process = None
        self.run_status = 'pending'

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
