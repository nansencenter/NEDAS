import os
import glob
import inspect
import numpy as np
from typing import Callable, Any
from abc import ABC, abstractmethod
from datetime import datetime
from NEDAS.config import parse_config
from .types import VarName, VarDesc, LevelID
from .context import Context

class Dataset(ABC):
    """
    Dataset class (template for specific dataset sources)
    """
    dataset_name: str
    variables: dict[VarName, VarDesc] = {}
    obs_operator: dict[VarName, Callable] = {}
    memory: dict = {}
    _c: Context

    def  __init__(self, context: Context|None=None,
                  config_file: str|None=None,
                  parse_args: bool=False,
                  **kwargs) -> None:
        # prepare context
        if context is not None:
            assert isinstance(context, Context), f"{context} is not a Context object"
            self._c = context
        else:
            self._c = Context()  # use default context if not specified

        # parse dataset config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        self.dataset_name = os.path.basename(code_dir)
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        if not hasattr(self, 'dataset_dir'):
            self.dataset_dir = '.'

        if not hasattr(self, 'obs_window_min'):
            self.obs_window_min = 0
        if not hasattr(self, 'obs_window_max'):
            self.obs_window_max = 0

    @property
    def c(self) -> Context:
        return self._c

    def parse_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Parse the input kwargs to pinpoint a specific file/variable...
        """
        # args to pinpoint a certain observatino record, used by read_obs, etc.
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
            kwargs['units'] = self.variables[kwargs['name']].units   # TODO: potential key error here if variables is not defined.

        # other args, set default values if not specified
        # model (Model): model class instance
        # grid (GridType): target analysis grid
        # mask (np.ndarray): target analysis grid mask (True if grid point is not part of the state)
        # z (dict[LevelID, np.ndarray]): z coordinates at each level k on analysis grid
        #TODO: These should be available from the context
        for key in ['model', 'grid', 'mask', 'z']:
            if key not in kwargs:
                kwargs[key] = None
        # nobs (int): number of observations
        # obs_window_min (int)
        # obs_window_max (int)  # TODO: maybe setting them both to 0 is incorrect? need to iterate from min to max...
        for key in ['nobs', 'obs_window_min', 'obs_window_max']:
            if key not in kwargs:
                kwargs[key] = 0

        return kwargs

    def get_mstr(self, member):
        return f'_mem{member+1:03d}' if member is not None else ''

    def get_tstr(self, time):
        assert time is not None, 'missing time'
        return f"{time:%Y%m%d_%H%M}"

    def generate_obs_network(self, **kwargs) -> dict[str, np.ndarray]:
        """
        Generate a random observing network for use in synthetic observation experiments.

        Args:
            **kwargs
        """
        raise NotImplementedError(f"'generate_obs_network' is not implemented for {self.__class__.__name__}")

    def read_obs(self, **kwargs) -> dict[str, np.ndarray]:
        if self.c.config.io_mode == 'offline':
            return self.read_obs_from_file(**kwargs)
        elif self.c.config.io_mode == 'online':
            return self.read_obs_from_memory(**kwargs)
        else:
            raise ValueError(f"Unknown io_mode: {self.c.config.io_mode}")

    @abstractmethod
    def read_obs_from_file(self, **kwargs) -> dict[str, np.ndarray]:
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

    def read_obs_from_memory(self, **kwargs) -> dict[str, np.ndarray]:
        kwargs = self.parse_kwargs(kwargs)
        tstr = self.get_tstr(kwargs['time'])
        tag = kwargs['tag']
        mstr = self.get_mstr(kwargs['member'])
        key = tag+mstr
        name = kwargs['name']
        if tstr not in self.memory:
            raise KeyError(f"{self.__class__.__name__}: '{tstr}' not found in memory")
        if key not in self.memory[tstr]:
            raise KeyError(f"{self.__class__.__name__}: '{key}' not found in memory['{tstr}']")
        if name not in self.memory[tstr][key]:
            raise KeyError(f"{self.__class__.__name__}: '{name}' not found in memory['{tstr}']['{key}']")
        return self.memory[tstr][key][name]

    def write_obs(self, seq: dict, **kwargs) -> None:
        if self.c.config.io_mode == 'offline':
            self.write_obs_to_file(seq, **kwargs)
        elif self.c.config.io_mode == 'online':
            self.write_obs_to_memory(seq, **kwargs)
        else:
            raise ValueError(f"Unknown io_mode: {self.c.config.io_mode}")

    def write_obs_to_file(self, seq: dict, **kwargs):
        pass

    def write_obs_to_memory(self, seq: dict, **kwargs):
        kwargs = self.parse_kwargs(kwargs)
        tag = kwargs['tag']
        mstr = self.get_mstr(kwargs['member'])
        key = tag+mstr
        tstr = self.get_tstr(kwargs['time'])
        name = kwargs['name']
        # create memory dict entry if not yet
        if tstr not in self.memory:
            self.memory[tstr] = {}
        if key not in self.memory[tstr]:
            self.memory[tstr][key] = {}
        self.memory[tstr][key][name] = seq

    def save_memory(self, tag: str, time: datetime|None=None, path: str|None=None) -> None:
        if self.c.config.io_mode == 'offline':
            return
        if path is None:
            path = self.c.config.work_dir
        times_to_save = [self.get_tstr(time)] if time is not None else self.memory.keys()
        for tstr in times_to_save:
            if tstr not in self.memory:
                continue
            for key in self.memory[tstr]:
                if not key.startswith(tag):
                    continue
                savedir = os.path.join(path, 'memory', 'dataset', self.dataset_name, tstr, key)
                self.c.fs.make_dir(savedir)
                for name in self.memory[tstr][key]:
                    savefile = os.path.join(savedir, f'{name}.npy')
                    np.save(savefile, np.array(self.memory[tstr][key][name], dtype=object))

    def load_memory(self, tag: str, time: datetime|None=None, path: str|None=None) -> None:
        if self.c.config.io_mode == 'offline':
            return
        if path is None:
            path = self.c.config.work_dir
        tstr_pattern = self.get_tstr(time) if time is not None else '????????_????'
        search_path = os.path.join(path, 'memory', 'dataset', self.dataset_name, tstr_pattern, f'{tag}*', '*.npy')
        for savefile in glob.glob(search_path):
            # extract tstr and key
            tstr = os.path.basename(os.path.dirname(os.path.dirname(savefile)))
            key = os.path.basename(os.path.dirname(savefile))
            name = os.path.splitext(os.path.basename(savefile))[0]
            # load data to memory
            if tstr not in self.memory:
                self.memory[tstr] = {}
            if key not in self.memory[tstr]:
                self.memory[tstr][key] = {}
            self.memory[tstr][key][name] = np.load(savefile, allow_pickle=True).item()
