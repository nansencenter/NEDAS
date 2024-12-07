import os
import inspect
from datetime import datetime
from config import parse_config

class ModelConfig(object):
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
        self.mask = None

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

