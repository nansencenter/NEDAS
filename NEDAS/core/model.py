import os
import glob
import inspect
from typing import Generic, TypeVar, Any
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from NEDAS.config import parse_config
from NEDAS.grid import GridType
from .types import IOMode, VarName, VarDesc, LevelID, EnsRunStrategy
from .context import Context

GridT = TypeVar("GridT", bound=GridType)

class Model(Generic[GridT], ABC):
    """
    Class for configuring and running a model
    """
    model_name: str
    io_mode: IOMode
    variables: dict[VarName, VarDesc]
    grid: GridT
    z: dict[LevelID, np.ndarray]
    mask: np.ndarray
    ens_init_dir: str|None
    truth_dir: str|None
    ens_run_strategy: EnsRunStrategy
    nproc_per_run: int = 1
    nproc_per_util: int = 1
    walltime: int|None = None
    run_process = None
    run_status: str = 'pending'
    restart_dir: str
    forecast_period: int
    memory: dict
    _c: Context

    def __init__(self, context: Context|None=None,
                 io_mode: IOMode|None=None,
                 config_file: str|None=None,
                 parse_args: bool=False,
                 **kwargs) -> None:
        # prepare context
        if context is not None:
            assert isinstance(context, Context), f"{context} is not a Context object"
            self._c = context
        else:
            self._c = Context()  # use default context if not specified

        # determine io_mode
        if io_mode:
            self.io_mode = io_mode
        else:
            self.io_mode = self._c.config.io_mode

        # parse model config file and obtain a list of attributes
        # get a list of values from default.yml and update with kwargs, save to config_dict
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        self.model_name = os.path.basename(code_dir)
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

    @property
    def c(self) -> Context:
        return self._c

    def parse_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        # args that pinpoints a certain model state variable
        if 'path' not in kwargs:
            kwargs['path'] = self.c.config.work_dir  # default path is work_dir

        if 'member' not in kwargs:
            kwargs['member'] = None
        if kwargs['member'] is not None:
            assert kwargs['member'] >= 0, f"member index should be >= 0, got {kwargs['member']}"

        if 'name' not in kwargs:
            kwargs['name'] = list(self.variables.keys())[0]  # if not specified, use first variable listed
        assert kwargs['name'] in self.variables, f"'{kwargs['name']}' is not defined in model variables"

        if 'time' not in kwargs:
            kwargs['time'] = None
        if kwargs['time'] is not None:
            assert isinstance(kwargs['time'], datetime), "kwargs 'time' is expected to be a datetime object'"

        levels = list(self.variables[kwargs['name']].levels)
        if 'k' not in kwargs:
            kwargs['k'] = levels[0]  # set to the first level if not specified
        assert kwargs['k'] in levels, f"level {kwargs['k']} is not available for variable {kwargs['name']}"

        if 'units' not in kwargs:
            kwargs['units'] = self.variables[kwargs['name']].units

        # some other runtime args need to be initialized for methods
        for key in ['restart_dir', 'forecast_period']:
            if key not in kwargs:
                kwargs[key] = None
        return kwargs

    def get_mstr(self, member):
        return f'_mem{member+1:03d}' if member is not None else ''

    def get_tstr(self, time):
        assert time is not None, 'missing time'
        return f"{time:%Y%m%d_%H%M}"

    @abstractmethod
    def read_grid(self, **kwargs) -> None:
        """
        Read the grid information from the model output.

        Args:
            **kwargs: Keyword arguments for reading the grid.
        """
        ...

    def read_var(self, **kwargs) -> np.ndarray:
        """
        Read a variable from the model output.

        Args:
            **kwargs: Keyword arguments for reading the variable.

        Returns:
            np.ndarray: The read variable.
        """
        if self.io_mode == 'offline':
            return self.read_var_from_file(**kwargs)
        elif self.io_mode == 'online':
            return self.read_var_from_memory(**kwargs)
        else:
            raise ValueError(f"Unknown io_mode {self.io_mode}")

    def read_var_from_file(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def read_var_from_memory(self, **kwargs):
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

    def write_var(self, var, **kwargs) -> None:
        """
        Write a variable to the model output.

        Args:
            var (np.ndarray): The variable to write.
            **kwargs: Keyword arguments for writing the variable.
        """
        if self.io_mode == 'offline':
            self.write_var_to_file(var, **kwargs)
        elif self.io_mode == 'online':
            self.write_var_to_memory(var, **kwargs)
        else:
            raise ValueError(f"Unknown io_mode {self.io_mode}")

    def write_var_to_file(self, var, **kwargs):
        raise NotImplementedError

    def write_var_to_memory(self, var, **kwargs):
        kwargs = self.parse_kwargs(kwargs)
        tag = kwargs['tag']
        mstr = self.get_mstr(kwargs['member'])
        tstr = self.get_tstr(kwargs['time'])
        key = tag+mstr
        name = kwargs['name']
        # create memory dict entry if not yet
        if tstr not in self.memory:
            self.memory[tstr] = {}
        if key not in self.memory[tstr]:
            self.memory[tstr][key] = {}
        self.memory[tstr][key][name] = var

    def save_memory(self, tag: str, time: datetime|None=None, path: str|None=None) -> None:
        if path is None:
            path = self.c.config.work_dir
        times_to_save = [self.get_tstr(time)] if time is not None else self.memory.keys()
        for tstr in times_to_save:
            if tstr not in self.memory:
                continue
            for key in self.memory[tstr]:
                if not key.startswith(tag):
                    continue
                savedir = os.path.join(path, 'memory', 'model', self.model_name, tstr, key)
                self.c.fs.make_dir(savedir)
                for name in self.memory[tstr][key]:
                    savefile = os.path.join(savedir, f'{name}.npy')
                    np.save(savefile, self.memory[tstr][key][name])

    def load_memory(self, tag: str, time: datetime|None=None, path: str|None=None) -> None:
        if path is None:
            path = self.c.config.work_dir
        tstr_pattern = self.get_tstr(time) if time is not None else '????????_????'
        search_path = os.path.join(path, 'memory', 'model', self.model_name, tstr_pattern, f'{tag}*', '*.npy')
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
            self.memory[tstr][key][name] = np.load(savefile)

    @abstractmethod
    def z_coords(self, **kwargs) -> np.ndarray:
        """
        Get the vertical coordinates of the model.

        Args:
            **kwargs: Keyword arguments for getting the vertical coordinates.

        Returns:
            np.ndarray: The vertical coordinates.
        """
        ...

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> None:
        """
        Preprocess the model data.

        Args:
            **kwargs: Keyword arguments for preprocessing.
        """
        ...

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> None:
        """
        Postprocess the model data.

        Args:
            **kwargs: Keyword arguments for postprocessing.
        """
        ...

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """
        Run the model forward in time.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments

        Keyword Args:
            time (datetime): current time when forecast starts
            restart_dir (str): directory where restart files are located
            forecast_period (int): forecast period in hours

        If self.ens_run_strategy == 'batch', the method will run all ensemble members in one go,
        expect additional kwargs['nens'] to be the ensemble size.
        If self.ens_run_strategy == 'scheduler', the method runs a single member indexed by kwargs['member'],
        and kwargs['worker_id'] is the pid assigned by the scheduler to run this method.
        """
        ...

    def generate_truth(self, *args, **kwargs) -> None:
        """
        Generate truth (nature run) model states. Use for running synthetic observation experiments.

        """
        raise NotImplementedError(f"'generate_truth' is not implemented for {self.__class__.__name__}")


    def generate_init_ensemble(self, *args, **kwargs) -> None:
        """
        Generate initial perturbed model states for ensemble forecasts.

        Args:
            nens (int): ensemble size
            **kwargs
        """
        raise NotImplementedError(f"'generate_init_ensemble' is not implemented for {self.__class__.__name__}")
