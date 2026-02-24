from typing import Callable, get_args
import numpy as np
from datetime import datetime, timedelta
from pyproj import Proj
from NEDAS.utils.parallel import Comm, by_rank, print_with_cache
from NEDAS.grid import Grid, GridType
from NEDAS.config import Config
from NEDAS.models import get_model_class
from NEDAS.datasets import get_dataset_class
from .io_backend import IOBackend

class Coordinator:
    """
    Coordinator manages the generation of dynamic objects in runtime
    """
    comm: Comm
    comm_rec: Comm
    comm_mem: Comm
    pid_show: int
    #grid: GridType
    time: datetime
    iter: int
    #models: dict[str, Model]
    #datasets: dict[str, Dataset]
    io: IOBackend
    transform_funcs: list #[Transform]
    #localization_funcs: list[Callable]
    #inflation_func: Inflation

    def __init__(self, config: Config):
        self.config = config

        ##initialize the current time pointer
        ##prev_time and next_time properties provide the time for previous/next analysis cycle 
        self.time = self.config.time

        ##initialize the current iteration
        self.iter = self.config.iter

        ##initialize the pid that shows progress (default to the root process pid=0)
        self.pid_show = 0

        ##setup a few living objects
        self.set_comm()
        self.set_analysis_grid()
        # self.set_models()
        # self.set_datasets()
#         self.set_io_backend()

    def __getattr__(self, key):
        # get values from config if defined, otherwise will get the attr from the instance directly.
        if hasattr(self.config, key):
            return getattr(self.config, key)

    @property
    def prev_time(self) -> datetime:
        """
        Previous analysis time. Automatically updated when self.time changes.

        Returns:
            datetime: Previous analysis time.
        """
        if self.time > self.config.time_start:
            return self.time - self.config.cycle_period * timedelta(hours=1)
        else:
            return self.time

    @property
    def next_time(self) -> datetime:
        """
        Next analysis time. Automatically updated when self.time changes.

        Returns:
            datetime: Next analysis time.
        """
        return self.time + self.config.cycle_period * timedelta(hours=1)

    def set_comm(self):
        """
        Initialize the MPI communicator, split the communicator if necessary.

        For serial program, use a dummy communicator, set :code:`nproc` to the number of available processors on the machine;
        for MPI program, use :code:`MPI.COMM_WORLD` and check if size matchs with :code:`nproc`.

        Split the communicator into member and record groups, according to :code:`nproc` and :code:`nproc_mem`.
        See :mod:`NEDAS.utils.parallel` module for more details.
        """
        ##initialize mpi communicator (could be size 1 for serial program)
        self.comm = Comm()
        comm_size = self.comm.Get_size()

        if comm_size != self.nproc:
            raise RuntimeError(f"Config nproc={self.nproc} does not match with MPI COMM size={comm_size}.")

        self.pid = self.comm.Get_rank()  ##current processor id

        self.pid_mem = self.pid % self.nproc_mem
        self.pid_rec = self.pid // self.nproc_mem

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
            self.grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, **other_opts)

            ##mask for invalid grid points (none for now, add option later)
            self.grid.mask = np.full((self.grid.ny, self.grid.nx), False, dtype=bool)
            if 'mask' in self.grid_def and self.grid_def['mask'] is not None:
                model_name = self.grid_def['mask']
                Model = get_model_class(model_name)
                model = Model()
                prepare_mask = getattr(model, 'prepare_mask', None)
                if prepare_mask is not None:
                    self.grid.mask = prepare_mask(self.grid)

        else:
            ##get analysis grid from model module
            model_name = self.grid_def['type']
            if self.model_def is None or model_name not in self.model_def:
                raise KeyError(f"'{model_name}' not defined in config file model_def section")
            kwargs = self.model_def[model_name]
            Model = get_model_class(model_name)
            model = Model(**kwargs)
            model_grid = getattr(model, 'grid')
            if not isinstance(model_grid, get_args(GridType)):
                raise TypeError(f"Model {model_name} does not have a valid grid attribute.")
            self.grid = model_grid

    def set_models(self):
        """
        Initialize model instances based on :code:`model_def[model_name]` settings.
        Store the model instances in :code:`models[model_name]`.
        """
        self.models = {}
        if self.model_def is None:
            return
        for model_name, kwargs in self.model_def.items():
            #instantiate the model class
            ModelClass = get_model_class(model_name)
            if not isinstance(kwargs, dict):
                kwargs = {}
            kwargs['io_mode'] = self.io_mode
            model = ModelClass(**kwargs)

            self.models[model_name] = model

    def set_datasets(self):
        """
        Initialize dataset instances based on :code:`dataset_def[dataset_name]` settings.
        Store the dataset instances in :code:`datasets[dataset_name]`.
        """
        self.datasets = {}
        if self.dataset_def is None:
            return
        for dataset_name, kwargs in self.dataset_def.items():
            DatasetClass = get_dataset_class(dataset_name)
            if not isinstance(kwargs, dict):
                kwargs = {}
            self.datasets[dataset_name] = DatasetClass(grid=self.grid, mask=self.grid.mask, **kwargs)


#     def set_io_backend(self):
#         ##validate io_mode
#         if self.io_mode not in ['offline', 'online']:
#             raise ValueError(f"Unsupported io_mode '{self.io_mode}', only 'online' or 'offline'.")

#         # define directory structure for file storage (offline mode)
#         if self.io_mode == 'offline':
#             self.directories: dict[str, str] = self.config_dict['directories']

    @property
    def print_1p(self):
        """
        Customized print function for showing runtime message.

        Only the processor with ID = self.pid_show will show the message,
        this avoids the redundancy if all processors are showing the same message.
        """
        decorator = by_rank(self.comm, self.pid_show)
        return decorator(print_with_cache)

#     def show_summary(self):
#         """
#         Print a summary of the configuration.
#         """
#         print(f"""Initializing config...
#  working directory: {self.work_dir}
#  parallel scheme: nproc = {self.nproc}, nproc_mem = {self.nproc_mem}
#  cycling from {self.time_start} to {self.time_end}
#  analysis start at {self.time_analysis_start}
#  cycle_period = {self.cycle_period} hours
#  current time: {self.time}
#  nens: {self.nens}
#  Assimilation scheme:
#  """, flush=True)
