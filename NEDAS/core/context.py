from __future__ import annotations
from typing import get_args, Callable, TYPE_CHECKING
import numpy as np
from datetime import datetime, timedelta
from pyproj import Proj
from NEDAS.utils import parallel, progress
from NEDAS import grid, config, models, datasets, assim_tools
if TYPE_CHECKING:
    from . import Model, Dataset, IOBackend, State, Obs, Transform, Inflation, Assimilator, Updator

class Context:
    """
    Runtime context manages the generation and interaction of dynamic objects in runtime
    """
    comm: parallel.Comm
    comm_rec: parallel.Comm
    comm_mem: parallel.Comm
    pid_show: int
    grid: grid.GridType
    grid_orig: grid.GridType
    time: datetime
    iter: int
    models: dict[str, Model]
    datasets: dict[str, Dataset]
    assimilator: Assimilator
    updator: Updator
    transform_funcs: list[Transform]
    localization_funcs: dict[str, Callable]
    inflation_func: Inflation
    io: IOBackend
    state: State
    obs: Obs

    def __init__(self, config: config.Config):
        self.config = config

        # initialize the current time pointer
        # prev_time and next_time properties provide the time for previous/next analysis cycle 
        self.time = self.config.time

        # initialize the current iteration
        self.iter = self.config.iter

        # initialize the pid that shows progress (default to the root process pid=0)
        self.pid_show = 0

        # setup a few living objects
        self.set_comm()
        self.set_grid()
        self.set_models()
        self.set_datasets()

        # more living objects (io, state, obs, other components)
        # will be created by scheme class methods at runtime
    
    def update_assim_tools(self):
        """ Update the assimilation tool components based on runtime configuration """
        # update grid with current iteration settings
        res_lev = self.config.resolution_level[self.iter]
        self.grid = self.grid_orig.change_resolution_level(res_lev)

        # initialize a few func components in the assimilation algorithm
        self.assimilator = assim_tools.assimilators.get_assimilator(self)
        self.updator = assim_tools.updators.get_updator(self)
        self.localization_funcs = assim_tools.localization.get_localization_funcs(self)
        self.inflation_func = assim_tools.inflation.get_inflation_func(self)
        self.transform_funcs = assim_tools.transforms.get_transform_funcs(self)

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
        self.comm = parallel.Comm()
        comm_size = self.comm.Get_size()

        if comm_size != self.config.nproc:
            raise RuntimeError(f"Config nproc={self.config.nproc} does not match with MPI COMM size={comm_size}.")

        self.pid = self.comm.Get_rank()  ##current processor id

        self.pid_mem = self.pid % self.config.nproc_mem
        self.pid_rec = self.pid // self.config.nproc_mem

        ##split comm if in mpi program
        if comm_size > 1:
            self.comm_mem = self.comm.Split(self.pid_rec, self.pid_mem)
            self.comm_rec = self.comm.Split(self.pid_mem, self.pid_rec)
        else:
            self.comm_mem = self.comm
            self.comm_rec = self.comm

    def set_grid(self):
        """
        Initialize the analysis grid based on the configuration.

        If :code:`grid_def['type']` is 'custom', will create a analysis grid based on provided parameters.
        If :code:`grid_def['type']` is a model name, will load the grid from the specified model class.
        """
        grid_def = self.config.grid_def
        if grid_def['type'] == 'custom':
            if 'proj' in grid_def and grid_def['proj'] is not None:
                proj = Proj(grid_def['proj'])
            else:
                proj = None
            xmin, xmax = grid_def['xmin'], grid_def['xmax']
            ymin, ymax = grid_def['ymin'], grid_def['ymax']
            dx = grid_def['dx']
            known_keys = {'type', 'proj', 'xmin', 'xmax', 'ymin', 'ymax', 'dx', 'mask'}
            other_opts = {k: v for k, v in grid_def.items() if k not in known_keys}
            self.grid = grid.Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, **other_opts)

            ##mask for invalid grid points (none for now, add option later)
            self.grid.mask = np.full((self.grid.ny, self.grid.nx), False, dtype=bool)
            if 'mask' in grid_def and grid_def['mask'] is not None:
                model_name = grid_def['mask']
                Model = models.get_model_class(model_name)
                model = Model()
                prepare_mask = getattr(model, 'prepare_mask', None)
                if prepare_mask is not None:
                    self.grid.mask = prepare_mask(self.grid)

        else:
            ##get analysis grid from model module
            model_def = self.config.model_def
            model_name = grid_def['type']
            if model_def is None or model_name not in model_def:
                raise KeyError(f"'{model_name}' not defined in config file model_def section")
            kwargs = model_def[model_name]
            Model = models.get_model_class(model_name)
            model = Model(**kwargs)
            model_grid = getattr(model, 'grid')
            if not isinstance(model_grid, get_args(grid.GridType)):
                raise TypeError(f"Model {model_name} does not have a valid grid attribute.")
            self.grid = model_grid

        # make a copy of the original analysis grid
        self.grid_orig = self.grid

    def set_models(self):
        """
        Initialize model instances based on :code:`model_def[model_name]` settings.
        Store the model instances in :code:`models[model_name]`.
        """
        self.models = {}
        for model_name, kwargs in self.config.model_def.items():
            #instantiate the model class
            ModelClass = models.get_model_class(model_name)
            if not isinstance(kwargs, dict):
                kwargs = {}
            kwargs['io_mode'] = self.config.io_mode
            model = ModelClass(**kwargs)

            self.models[model_name] = model

    def set_datasets(self):
        """
        Initialize dataset instances based on :code:`dataset_def[dataset_name]` settings.
        Store the dataset instances in :code:`datasets[dataset_name]`.
        """
        self.datasets = {}
        for dataset_name, kwargs in self.config.dataset_def.items():
            DatasetClass = datasets.get_dataset_class(dataset_name)
            if not isinstance(kwargs, dict):
                kwargs = {}
            kwargs['io_mode'] = self.config.io_mode
            self.datasets[dataset_name] = DatasetClass(grid=self.grid, mask=self.grid.mask, **kwargs)

    @property
    def print_1p(self):
        """
        Customized print function for showing runtime message.

        Only the processor with ID = self.pid_show will show the message,
        this avoids the redundancy if all processors are showing the same message.
        """
        decorator = parallel.by_rank(self.comm, self.pid_show)
        return decorator(parallel.print_with_cache)

    def show_progress(self, debug_message: str, task: int, total_ntask: int) -> None:
        """
        Show progress

        If debug=True, print the debug_message with flush=True
        Otherwise, show a progress bar, indicating current task/total_ntask percentage.
        """
        if self.config.debug:
            print(debug_message, flush=True)
        else:
            self.print_1p(progress.progress_bar(task, total_ntask))

    def show_summary(self):
        conf_summary = self.config.summary()
        summary_text = f"""Configuration:\n{conf_summary}
current time: {self.time}
...
"""
        self.print_1p(summary_text)
