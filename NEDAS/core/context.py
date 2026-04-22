from __future__ import annotations
import os
import sys
import shutil
import copy
import time
from typing import get_args, Callable, TYPE_CHECKING
from functools import wraps
import numpy as np
from datetime import datetime, timedelta
from pyproj import Proj
from NEDAS import __version__
from NEDAS.utils import parallel, progress
from NEDAS.config import Config
from NEDAS import grid, models, datasets, assim_tools, io_backends, job_submitters
from .file_system import FileSystem
from .types import ProcIDMem, MemID, ParallelMode
if TYPE_CHECKING:
    from . import Model, Dataset, IOBackend, JobSubmitter, State, Obs, Transform, Inflation, Assimilator, Updator

class Context:
    """
    Runtime context manages the generation and interaction of dynamic objects in runtime
    """
    interactive: bool
    is_notebook: bool
    debug: bool
    comm: parallel.Comm
    comm_rec: parallel.Comm
    comm_mem: parallel.Comm
    pid_show: int
    progress: progress.Progress
    fs: FileSystem
    io: IOBackend
    jsub: JobSubmitter
    nens: int
    mem_list: dict[ProcIDMem, list[MemID]]
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
    state: State
    obs: Obs

    def __init__(self, config: Config|None=None,
                 config_file: str|None=None,
                 parse_args: bool=False,
                 **kwargs) -> None:
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config(config_file=config_file, parse_args=parse_args, **kwargs)

        # initialize logging
        self.debug = self.config.debug
        self.interactive = self.check_interactive()
        self.is_notebook = self.check_notebook()
        self.set_logging()

        # initialize the current time pointer
        # prev_time and next_time properties provide the time for previous/next analysis cycle
        self.time = self.config.time
        # initialize the current iteration
        self.iter = self.config.iter
        # initialize the pid that shows progress (default to the root process pid=0)
        self.pid_show = 0
        self._prev_msg = ''

        # ensemble size
        self.nens = self.config.nens

        # setup the parallel (serial or MPI program) communicator
        self.set_comm()
        self.mem_list = parallel.bcast_by_root(self.comm)(self.distribute_mem_tasks)()

        # initialize a few helper class instances
        self.fs = FileSystem(self.config)
        self.io = io_backends.get_io_backend(self.config.io_mode)
        self.jsub = job_submitters.get_job_submitter(**(self.config.job_submit or {}))

        # setup the analysis grid object
        self.set_grid()

        # setup the model and obs dataset objects
        self.set_models()
        self.set_datasets()

        # more living objects (io, state, obs, other components)
        # will be created by scheme class __init__ and methods at runtime

    def check_interactive(self) -> bool:
        """
        If the runtime environment supports interactive output (with ANSI escape code).
        """
        # if debug mode is on, disable interactive output
        # since we need to show a lot of debug messages
        if self.config.debug:
            return False
        # if interactive option is explicitly set in config, use it
        if self.config.interactive is not None:
            return self.config.interactive
        # otherwise, check if a tty is present, if so, should support interactive output.
        return os.isatty(sys.stdout.fileno())

    def check_notebook(self) -> bool:
        """
        If the runtime environment is a jupyter notebook
        """
        if self.config.is_notebook is not None:
            return self.config.is_notebook
        return "ipykernel" in sys.modules

    def get_cols(self) -> int:
        # if cols is explicitly set in config, use it
        if self.config.cols is not None:
            return self.config.cols
        # in jupyter notebook environment, just use a large number
        if self.is_notebook:
            return 800
        # otherwise, get real time terminal size
        cols, _ = shutil.get_terminal_size()
        return cols

    def set_logging(self) -> None:
        progress_opts = {
            'interactive': self.interactive,
            'is_notebook': self.is_notebook,
            'cols': self.get_cols(),
            'debug': self.config.debug,
            'call_stack': self.config.call_stack,
            'call_stack_max_level': self.config.call_stack_max_level,
            'anchor': self.config.anchor,
            'tabspace': self.config.tabspace,
            'progress_bar_width': self.config.progress_bar_width,
        }
        self.progress = progress.Progress(**progress_opts)

    def distribute_mem_tasks(self) -> dict[int, list[int]]:
        """
        Distribute mem_id across processors
        """
        # list of mem_id as tasks
        mem_list_full = [m for m in range(self.nens)]
        mem_list = parallel.distribute_tasks(self.comm_mem, mem_list_full)
        return mem_list

    def update_assim_tools(self):
        """
        Update the assimilation tool components based on runtime configuration
        """
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

    def set_comm(self) -> None:
        """
        Initialize the MPI communicator, split the communicator if necessary.

        For serial program, use a dummy communicator, set :code:`nproc` to the number of available processors on the machine;
        for MPI program, use :code:`MPI.COMM_WORLD` and check if size matchs with :code:`nproc`.

        Split the communicator into member and record groups, according to :code:`nproc` and :code:`nproc_mem`.
        See :mod:`NEDAS.utils.parallel` module for more details.
        """
        # initialize mpi communicator (could be size 1 for serial program)
        self.comm = parallel.Comm()
        comm_size = self.comm.Get_size()

        self.pid = self.comm.Get_rank()  # current processor id

        # stop early if mpi environment is not ready (program is not called from mpirun)
        if not self.comm.mpi_ready:
            self.pid_mem, self.pid_rec = 0, 0
            self.comm_mem, self.comm_rec = self.comm, self.comm
            return

        # validate mpi environment
        if comm_size != self.config.nproc:
            raise RuntimeError(f"Config nproc={self.config.nproc} does not match with MPI COMM size={comm_size}.")

        # split comm so that nproc_mem * nproc_rec == nproc
        self.pid_mem = self.pid % self.config.nproc_mem
        self.pid_rec = self.pid // self.config.nproc_mem
        self.comm_mem = self.comm.Split(self.pid_rec, self.pid_mem)
        self.comm_rec = self.comm.Split(self.pid_mem, self.pid_rec)

    def set_grid(self) -> None:
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

            # mask for invalid grid points (none for now, add option later)
            self.grid.mask = np.full((self.grid.ny, self.grid.nx), False, dtype=bool)
            if 'mask' in grid_def and grid_def['mask'] is not None:
                model_name = grid_def['mask']
                Model = models.get_model_class(model_name)
                model = Model(context=self)
                prepare_mask = getattr(model, 'prepare_mask', None)
                if prepare_mask is not None:
                    self.grid.mask = prepare_mask(self.grid)

        else:
            # get analysis grid from model module
            model_def = self.config.model_def
            model_name = grid_def['type']
            if model_def is None or model_name not in model_def:
                raise KeyError(f"'{model_name}' not defined in config file model_def section")
            kwargs = model_def[model_name]
            Model = models.get_model_class(model_name)
            model = Model(context=self, **kwargs)
            model_grid = getattr(model, 'grid')
            if not isinstance(model_grid, get_args(grid.GridType)):
                raise TypeError(f"Model {model_name} does not have a valid grid attribute.")
            self.grid = model_grid

        # make a copy of the original analysis grid
        self.grid_orig = self.grid

    def set_models(self) -> None:
        """
        Initialize model instances based on :code:`model_def[model_name]` settings.
        Store the model instances in :code:`models[model_name]`.
        """
        self.models = {}
        if self.config.model_def is None:
            return
        for model_name, kwargs in self.config.model_def.items():
            #instantiate the model class
            ModelClass = models.get_model_class(model_name)
            self.models[model_name] = ModelClass(context=self, **(kwargs or {}))

    def set_datasets(self) -> None:
        """
        Initialize dataset instances based on :code:`dataset_def[dataset_name]` settings.
        Store the dataset instances in :code:`datasets[dataset_name]`.
        """
        self.datasets = {}
        if self.config.dataset_def is None:
            return
        for dataset_name, kwargs in self.config.dataset_def.items():
            DatasetClass = datasets.get_dataset_class(dataset_name)
            self.datasets[dataset_name] = DatasetClass(context=self, **(kwargs or {}))

    @property
    def total_tasks(self):
        return self.progress.node['total_tasks']

    @total_tasks.setter
    def total_tasks(self, value: int):
        self.progress.node['total_tasks'] = value
        self.debug_message = f"total tasks: {value}"

    @property
    def current_task(self):
        return self.progress.node['current_task']

    @current_task.setter
    def current_task(self, value: int):
        self.progress.node['current_task'] = value
        self.progress.set_flag('running')
        status = self.progress.update()
        self.print_1p(status)

    @property
    def message(self):
        return ''

    @message.setter
    def message(self, msg: str):
        self.progress.node['message'] = msg

    @property
    def debug_message(self):
        return ''

    @debug_message.setter
    def debug_message(self, msg: str):
        if self.config.quiet or not self.debug:
            return
        # show the debug message with PID info.
        # this uses the print since all PID ranks shall print its own message
        print(f"PID {self.pid:>4}: {msg}", flush=True)

    def timer(self, func: Callable):
        """
        Decorator to count the elapsed time for a function at runtime.
        """
        if not self.config.timer:
            return func
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                t1 = time.time()
                self.progress.node['elapsed_time'] = t1 - t0
        return wrapper

    def logger(self, func_name: str):
        """
        Decorator to register the func in call stack and show runtime messages.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # register the function in call stack
                    status = self.progress.push(func_name)
                    self.print_1p(status)
                    self.progress.set_flag('waiting')

                    # execute the function with timer
                    result = self.timer(func)(*args, **kwargs)
                    self.progress.set_flag('done')
                    return result

                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    self.progress.set_flag('error')
                    raise
                finally:
                    status = self.progress.pop()
                    self.print_1p(status)

            return wrapper
        return decorator

    def print_1p(self, msg: str):
        """
        Customized print function for showing runtime message.

        Only the processor with PID = self.pid_show will show the message,
        this avoids the redundancy if all processors are showing the same message.
        """
        if self.config.quiet or not msg:
            return
        if self.comm.Get_rank() != self.pid_show:
            return
        self._prev_msg = progress.print_with_cache(msg, self._prev_msg)

    def log_event(self, msg: str, flag=''):
        self.print_1p(self.progress.log(msg, flag))

    def show_greeting(self) -> None:
        greeting_msg = f"""
█▄  █ █▀▀▀ █▀▀▄ ▄▀▀▄ ▄▀▀▀
█ ▀▄█ █▀▀  █  █ █▀▀█ ▀▀▀█
▀   ▀ ▀▀▀▀ ▀▀▀  ▀  ▀ ▀▀▀  version {__version__}
"""
        self.print_1p(greeting_msg)

    def show_summary(self) -> None:
        summary_text = self.config.summary()
        self.print_1p(summary_text)

    def dump_config(self, config_file: str) -> None:
        """
        Dumps a snapshot of the current state to a yaml config file.
        The original config object remains unchanged in memory.
        """
        # make a copy of the config object for dumping
        tmp_config = copy.copy(self.config)

        # inject runtime state to the temporary config
        for rt_state in ['time', 'iter', 'pid_show', 'interactive', 'is_notebook']:
            val = getattr(self, rt_state)
            setattr(tmp_config, rt_state, val)
        setattr(tmp_config, 'call_stack', self.progress.call_stack)
        setattr(tmp_config, 'cols', self.progress.fmt.cols)

        # save the config to yaml file
        tmp_config.dump_yaml(config_file)

    def run_job(self, commands: str,
                parallel_mode: ParallelMode='serial',
                nproc: int=1,
                offset: int=0,
                **kwargs) -> None:
        """
        The user-facing method for running command on a computer.
        It re-configures the existing job submitter with runtime arguments and execute the command.

        Args:
            commands (str): Shell commands to be dispatched by job submitter
            parallel_mode (ParallelMode, optional): parallel mode ('serial', 'mpi', 'openmp'), default is 'serial'
            nproc (int, optional): number of processors (default is 1)
            offset (int, optional): offset in full list of processors (default is 0)
            **kwargs: other keyword arguments to update the job submitter configuration
        """
        # update the state of the job submitter for this specific task
        self.jsub.parallel_mode = parallel_mode
        self.jsub.nproc = nproc
        self.jsub.offset = offset

        for key, value in kwargs.items():
            if value and hasattr(self.jsub, key):
                setattr(self.jsub, key, value)

        # dispatch the command
        self.jsub.run(commands)
