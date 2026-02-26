import importlib
from typing import Type
import os
import inspect
from abc import ABC, abstractmethod
from datetime import datetime
from NEDAS.config import parse_config

class Dataset(ABC):
    """
    Dataset class (template for specific dataset sources)
    """
    def  __init__(self, config_file=None, parse_args=False, **kwargs):

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
            kwargs['units'] = self.variables[kwargs['name']]['units']

        ##other args, set default values if not specified
        for key in ['model', 'grid', 'mask']:
            if key not in kwargs:
                kwargs[key] = None
        for key in ['nobs', 'obs_window_min', 'obs_window_max']:
            if key not in kwargs:
                kwargs[key] = 0

        return kwargs

    @abstractmethod
    def read_obs(self, **kwargs) -> dict:
        """
        Return observation sequence matching the given kwargs
        """
        obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }
        return obs_seq

registry = {
    'ecmwf.era5': 'ERA5Data',
    'ecmwf.forecast': 'EcmwfForecastData',
    'ifremer.argo': 'ArgoObs',
    'osisaf.ice_conc': 'OsisafSeaIceConcObs',
    'osisaf.ice_drift': 'OsisafSeaIceDriftObs',
    'amsr2': 'AMSR2Obs',
    'cs2smos': 'Cs2SmosObs',
    'rgps': 'RgpsObs',
    'topaz': 'TopazPrepObs',
    'vort2d': 'Vort2DObs',
    'synthetic': 'SyntheticObs',
}

def get_dataset_class(dataset_name: str) -> Type["Dataset"]:
    """
    Factory function to return the correct Dataset subclass.

    Args:
        dataset_name (str): Dataset name

    Returns:
        Type["Dataset"]: Corresponding Dataset subclass.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in registry.keys():
        raise NotImplementedError(f"Dataset class not implemented for '{dataset_name}'")

    module = importlib.import_module('NEDAS.datasets.'+dataset_name)
    DatasetClass = getattr(module, registry[dataset_name])

    return DatasetClass
