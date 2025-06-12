import os
import inspect
from typing import Optional, Any
import yaml
import dateutil.parser
import numpy as np
from datetime import datetime, timedelta, timezone
from pyproj import Proj
import NEDAS
from .parse_config import parse_config
from NEDAS.utils import parallel

class Config:
    """
    Class to manage the configuration for running the NEDAS analysis.
    Configuration entries are described in details in :doc:`config_file`.

    Args:
        config_file (str, optional): Path to the configuration file.
        parse_args (bool, optional): If true, parse command line arguments to collect configuration. Default is False.
        **kwargs: Additional key-value pairs to be passed to parse_config. Can be used to override values in the config file.

    Attributes:
        nproc (int): Number of processors to use for the analysis step.
        comm (Comm): MPI communicator, set by :meth:`set_comm`.
        grid (Grid): Analysis grid, set by :meth:`set_analysis_grid`.
        model_config (dict): A dictionary where keys are model names and values are the corresponding Model instances, set by :meth:`set_model_config`.
        dataset_config (dict): A dictionary where keys are dataset names and values are the corresponding Dataset instances, set by :meth:`set_dataset_config`.
    """
    work_dir: str
    time: datetime
    time_start: datetime
    time_end: datetime
    time_analysis_start: datetime
    time_analysis_end: datetime
    cycle_period: float
    nproc: int
    nproc_mem: int
    nproc_rec: int
    nproc_util: int
    pid: int
    pid_mem: int
    pid_rec: int
    grid_def: dict
    model_def: dict
    dataset_def: dict
    comm: parallel.Comm

    def __init__(self, config_file: Optional[str]=None, parse_args: bool=False, **kwargs):
        # parse the yaml config file to obtain the values
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        self.config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)

        # replace placeholders in dir paths with actual values
        self.work_dir = os.path.abspath(self.work_dir)
        self.nedas_root = NEDAS.__path__[0]
        self.config_dict = self.parse_directories(self.config_dict)

        # set additional attributes
        self.set_time()
        self.set_comm()
        self.set_analysis_grid()
        self.set_models()
        self.set_datasets()

        if 'pid_show' not in self.config_dict or self.config_dict['pid_show'] is None:
            self.config_dict['pid_show'] = 0

        self.directories: dict[str, str] = self.config_dict['directories']

    def __getattr__(self, key):
        # get values from config_dict if defined, otherwise will get the attr from the instance directly.
        if 'config_dict' in self.__dict__ and key in self.config_dict:
            return self.config_dict[key]

    def __setattr__(self, key, value):
        if key != 'config_dict' and 'config_dict' in self.__dict__ and key in self.config_dict:
            if key == 'time' and not isinstance(value, datetime):
                raise TypeError("Time must be a datetime object.")
            self.config_dict[key] = value
        else:
            super().__setattr__(key, value)

    @property
    def prev_time(self) -> datetime:
        """
        Previous analysis time. Automatically updated when self.time changes.

        Returns:
            datetime: Previous analysis time.
        """
        if self.time > self.time_start:
            return self.time - self.cycle_period * timedelta(hours=1)
        else:
            return self.time

    @property
    def next_time(self) -> datetime:
        """
        Next analysis time. Automatically updated when self.time changes.

        Returns:
            datetime: Next analysis time.
        """
        return self.time + self.cycle_period * timedelta(hours=1)

    @property
    def print_1p(self):
        """
        Customized print function for showing runtime message.

        Only the processor with ID = self.pid_show will show the message,
        this avoids the redundancy if all processors are showing the same message.
        """
        decorator = parallel.by_rank(self.comm, self.pid_show)
        return decorator(parallel.print_with_cache)

    def cycle_dir(self, time: datetime) -> str:
        """
        Directory path for an analysis cycle.

        Args:
            time (datetime): Time of the analysis cycle.

        Returns:
            str: Directory path for the analysis cycle.
        """
        return self.directories['cycle_dir'].format(time=time)

    def forecast_dir(self, time: datetime, model_name: str):
        """
        Directory path for a model forecast step.

        Args:
            time (datetime): Time of the analysis cycle.
            model_name (str): Name of the model.

        Returns:
            str: Directory path for the model forecast.
        """
        return self.directories['forecast_dir'].format(time=time, model_name=model_name)

    def analysis_dir(self, time: datetime, iter: int=0):
        """
        Directory path for an analysis step.

        Args:
            time (datetime): Time of the analysis cycle.
            iter (int): If niter > 1, an outer iteration loop exists, step is the index in the loop.

        Returns:
            str: Directory path for the analysis step.
        """
        if self.niter == 1:
            iter_dir= ''
        else:
            iter_dir = f"iter{iter}"
        return self.directories['analysis_dir'].format(time=time, step=iter_dir)

    def parse_directories(self, data: Any) -> Any:
        """
        Parse the directories or file names defined in :code:`data`
        and replace the placeholders {work_dir} and {nedas_root} with the actual values.
        """
        if isinstance(data, dict):
            return {key: self.parse_directories(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.parse_directories(element) for element in data]
        elif isinstance(data, str):
            return data.replace('{work_dir}', self.work_dir).replace('{nedas_root}', self.nedas_root)
        else:
            return data

    def set_time(self):
        """
        Initialize the time variables for the analysis.

        Checks if the mandatory :code:`time_*` entries are defined in the config file.
        If :code:`time` is not set, set it to :code:`time_start` by default.
        YAML file recognizes 2001-01-01T00:00:00 format and convert directly to datetime object.
        If time is a formatted string, will try to parse it using dateutil.parser.
        """
        # check if mandatory time keys are defined in config file
        for key in ['time', 'time_start', 'time_end', 'time_analysis_start', 'time_analysis_end']:
            if key not in self.config_dict:
                raise KeyError(f"'{key}' is missing in config file")
            if isinstance(self.config_dict[key], str):
                try:
                    self.config_dict[key] = dateutil.parser.parse(self.config_dict[key])
                except Exception:
                    raise ValueError(f"Failed to convert string {key}={self.config_dict[key]} to datetime")
            # add default tzinfo
            if self.config_dict[key] and self.config_dict[key].tzinfo is None:
                self.config_dict[key] = self.config_dict[key].replace(tzinfo=timezone.utc)

        if self.time is None:
            ##initialize current time to start time, if not available
            self.config_dict['time'] = self.config_dict['time_start'].replace()

    def set_comm(self):
        """
        Initialize the MPI communicator, check the number of processors, split the communicator if necessary.

        For serial program, use a dummy communicator, set :code:`nproc` to the number of available processors on the machine;
        for MPI program, use :code:`MPI.COMM_WORLD` and check if size matchs with :code:`nproc`.

        Split the communicator into member and record groups, according to :code:`nproc` and :code:`nproc_mem`.
        See :mod:`NEDAS.utils.parallel` module for more details.
        """
        ##initialize mpi communicator (could be size 1 for serial program)
        self.comm = parallel.Comm()
        comm_size = self.comm.Get_size()
        self.pid = self.comm.Get_rank()  ##current processor id

        ##self.nproc is the total number of processors
        if hasattr(self, 'nproc') and self.nproc is not None:
            ##if it is set in config file, check if it matches the actual comm size
            if comm_size > 1 and self.nproc != comm_size:
                raise ValueError(f"Config nproc={self.nproc} does not match MPI comm size={comm_size}.")
        else:
            ##if not set, figure out the available number of processors
            if comm_size == 1:
                ##serial program, set nproc to available processors (scheduler can spawn tasks to them)
                nproc_avail = os.cpu_count()
                if nproc_avail:
                    self.nproc = nproc_avail
            else:
                ##mpi program, set nproc to comm size
                self.nproc = comm_size

        ##divide processors into mem/rec groups
        if not hasattr(self, 'nproc_mem') or self.nproc_mem is None:
            self.nproc_mem = self.nproc
        if self.nproc % self.nproc_mem != 0:
            raise ValueError(f"nproc={self.nproc} is not evenly divided by nproc_mem={self.nproc_mem}")
        self.nproc_rec = int(self.nproc/self.nproc_mem)
        self.pid_mem = self.pid % self.nproc_mem
        self.pid_rec = self.pid // self.nproc_mem

        if not hasattr(self, 'nproc_util') or self.nproc_util is None:
            self.nproc_util = self.nproc

        ##split comm if in mpi program
        if comm_size > 1:
            self.comm_mem = self.comm.Split(self.pid_rec, self.pid_mem)
            self.comm_rec = self.comm.Split(self.pid_mem, self.pid_rec)
        else:
            self.comm_mem = self.comm
            self.comm_rec = self.comm

    def set_analysis_grid(self):
        """
        Initialize the analysis grid based on the configuration.

        If :code:`grid_def['type']` is 'custom', will create a analysis grid based on provided parameters.
        If :code:`grid_def['type']` is a model name, will load the grid from the specified model class.
        """
        assert isinstance(self.grid_def, dict)
        if self.grid_def['type'] == 'custom':
            if 'proj' in self.grid_def and self.grid_def['proj'] is not None:
                proj = Proj(self.grid_def['proj'])
            else:
                proj = None
            xmin, xmax = self.grid_def['xmin'], self.grid_def['xmax']
            ymin, ymax = self.grid_def['ymin'], self.grid_def['ymax']
            dx = self.grid_def['dx']
            known_keys = {'type', 'proj', 'xmin', 'xmax', 'ymin', 'ymax', 'dx', 'mask'}
            other_opts = {k: v for k, v in self.grid_def.items() if k not in known_keys}
            self.grid = NEDAS.grid.Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, **other_opts)

            ##mask for invalid grid points (none for now, add option later)
            self.grid.mask = np.full((self.grid.ny, self.grid.nx), False, dtype=bool)
            if 'mask' in self.grid_def and self.grid_def['mask'] is not None:
                model_name = self.grid_def['mask']
                Model = NEDAS.models.get_model_class(model_name)
                model = Model()
                if hasattr(model, 'prepare_mask'):
                    self.grid.mask = model.prepare_mask(self.grid)

        else:
            ##get analysis grid from model module
            model_name = self.grid_def['type']
            kwargs = self.model_def[model_name]
            Model = NEDAS.models.get_model_class(model_name)
            model = Model(**kwargs)
            self.grid = model.grid

    def set_models(self):
        """
        Initialize model instances based on :code:`model_def[model_name]` settings.
        Store the model instances in :code:`models[model_name]`.
        """
        self.models = {}
        for model_name, kwargs in self.model_def.items():
            Model = NEDAS.models.get_model_class(model_name)
            if not isinstance(kwargs, dict):
                kwargs = {}
            self.models[model_name] = Model(**kwargs)

    def set_datasets(self):
        """
        Initialize dataset instances based on :code:`dataset_def[dataset_name]` settings.
        Store the dataset instances in :code:`datasets[dataset_name]`.
        """
        self.datasets = {}
        for dataset_name, kwargs in self.dataset_def.items():
            Dataset = NEDAS.datasets.get_dataset_class(dataset_name)
            if not isinstance(kwargs, dict):
                kwargs = {}
            self.datasets[dataset_name] = Dataset(grid=self.grid, mask=self.grid.mask, **kwargs)

    def show_summary(self):
        """
        Print a summary of the configuration.
        """
        print(f"""Initializing config...
 working directory: {self.work_dir}
 parallel scheme: nproc = {self.nproc}, nproc_mem = {self.nproc_mem}
 cycling from {self.time_start} to {self.time_end}
 analysis start at {self.time_analysis_start}
 cycle_period = {self.cycle_period} hours
 current time: {self.time}
 """, flush=True)

    def dump_yaml(self, config_file: str):
        """
        Dump the current configuration to a YAML file.

        Args:
            config_file (str): Path to the output configuration file.
        """
        with open(config_file, 'w') as f:
            yaml.dump(self.config_dict, f)
