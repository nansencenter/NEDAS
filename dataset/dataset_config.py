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
            setattr(self, key, None)
            if key in kwargs:
                setattr(self, key, kwargs[key])

    def parse_kwargs(self, **kwargs):
        """
        Parse the input kwargs to pinpoint a specific file/variable...
        """
        ##args to pinpoint a certain observatino record, used by read_obs, etc.
        if 'path' in kwargs:
            self.path = kwargs['path']
        else:
            self.path = self.dataset_dir

        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = list(self.variables.keys())[0]

        if 'time' in kwargs:
            self.time = kwargs['time']
        else:
            self.time = None
        if self.time is not None:
            assert isinstance(self.time, datetime), "kwargs['time'] is not a datetime object"

        ##other args used in random_network
        for key in ['grid', 'mask', 'z', 'nobs', 'obs_window_min', 'obs_window_max']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

    def read_obs(self, **kwargs):
        """
        Return observation sequence matching the given kwargs
        """
        obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }
        return obs_seq

