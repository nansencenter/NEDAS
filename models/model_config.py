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
        if 'path' in kwargs:
            self.path = kwargs['path']
        else:
            self.path = '.'

        if 'member' in kwargs:
            self.member = kwargs['member']
        else:
            self.member = None

        if 'name' not in kwargs:
            self.name = list(self.variables.keys())[0]  ##if not specified, use first variable listed
        else:
            self.name = kwargs['name']

        if 'time' in kwargs:
            self.time = kwargs['time']
        else:
            self.time = None
        if self.time is not None:
            assert isinstance(kwargs['time'], datetime), 'time is expected to be a datetime object'

        if 'k' in kwargs:
            self.k = kwargs['k']
        else:
            self.k = self.variables[self.name]['levels'][0]  ##get the first level if not specified

        ##now some additional args for runtime utility functions
        for key in ['job_submit_cmd', 'restart_dir', 'forecast_period']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

