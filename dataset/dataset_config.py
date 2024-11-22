import os
import inspect
from datetime import datetime
from config import parse_config

class DatasetConfig(object):
    """
    Dataset class (template for specific dataset sources)
    """
    def  __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.variables = {}

        for key in ['grid', 'mask', 'z', 'nobs', 'obs_window_min', 'obs_window_max', 'err', 'hroi', 'vroi', 'troi', 'impact_on_state']:
            if not hasattr(self, key):
                setattr(self, key, None)

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

        ##other args used in random_network
        for key in ['grid', 'mask', 'z', 'truth_dir', 'nobs', 'obs_window_min', 'obs_window_max']:
            if key not in kwargs:
                kwargs[key] = None

        return kwargs

    def read_obs(self, **kwargs):
        """
        Return observation sequence matching the given kwargs
        """
        obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }
        return obs_seq

