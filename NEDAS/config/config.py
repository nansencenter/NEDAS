import numpy as np
import os
import inspect
import yaml
import importlib
from datetime import datetime
from pyproj import Proj
from NEDAS.grid import Grid
from NEDAS.utils.parallel import Comm, by_rank
from NEDAS.utils.progress import print_with_cache
from NEDAS.utils.conversion import s2t, t2s, dt1h
from NEDAS.config.parse_config import parse_config

class Config:
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        self._time = None
        self._pid_show = 0

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)

        self.keys = []
        for key, value in config_dict.items():
            setattr(self, key, value)
            self.keys.append(key)

        ##some attributes are useful in runtime
        self.set_time()
        self.set_comm()
        self.set_analysis_grid()
        self.set_model_config()
        self.set_dataset_config()

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        if value:
            if isinstance(value, str):
                self._time = s2t(value)
            elif isinstance(value, datetime):
                self._time = value
            else:
                raise TypeError(f"Error: time must be a string or datetime object, not {type(value)}")

    @property
    def prev_time(self):
        if self.time > self.time_start:
            return self.time - self.cycle_period * dt1h
        else:
            return self.time

    @property
    def next_time(self):
        return self.time + self.cycle_period * dt1h

    @property
    def pid_show(self):
        return self._pid_show

    @pid_show.setter
    def pid_show(self, value):
        self._pid_show = value

    @property
    def print_1p(self):
        return by_rank(self.comm, self.pid_show)(print_with_cache)

    def cycle_dir(self, time):
        return self.directories['cycle_dir'].format(work_dir=self.work_dir, time=time)

    def forecast_dir(self, time, model_name):
        return self.directories['forecast_dir'].format(work_dir=self.work_dir, time=time, model_name=model_name)

    def analysis_dir(self, time, scale_id):
        if self.nscale == 1:
            scale_dir = ''
        else:
            scale_dir = f"scale{scale_id}"
        return self.directories['analysis_dir'].format(work_dir=self.work_dir, time=time, scale_dir=scale_dir)

    def set_time(self):
        for key in ['time_start', 'time_end', 'time_analysis_start', 'time_analysis_end']:
            ##make sure key is defined in config file
            assert key in self.keys, f"'{key}' is missing in config file"
            ##convert string to datetime obj
            setattr(self, key, s2t(getattr(self, key)))

        if self.time is None:
            ##initialize current time to start time, if not available
            self.time = self.time_start
            if 'time' not in self.keys:
                self.keys.append('time')

    def set_comm(self):
        ##initialize mpi communicator (could be size 1 for serial program)
        self.comm = Comm()
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
                self.nproc = os.cpu_count()
            else:
                ##mpi program, set nproc to comm size
                self.nproc = comm_size

        ##divide processors into mem/rec groups
        if not hasattr(self, 'nproc_mem') or self.nproc_mem is None:
            self.nproc_mem = self.nproc
        assert self.nproc % self.nproc_mem == 0, f"nproc={self.nproc} is not evenly divided by nproc_mem={self.nproc_mem}"
        self.nproc_rec = int(self.nproc/self.nproc_mem)
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
        ##initialize analysis grid
        if self.grid_def['type'] == 'custom':
            if 'proj' in self.grid_def and self.grid_def['proj'] is not None:
                proj = Proj(self.grid_def['proj'])
            else:
                proj = None
            xmin, xmax = self.grid_def['xmin'], self.grid_def['xmax']
            ymin, ymax = self.grid_def['ymin'], self.grid_def['ymax']
            dx = self.grid_def['dx']
            known_keys = {'type', 'proj', 'xmin', 'xmax', 'ymin', 'ymax', 'dx'}
            other_opts = {k: v for k, v in self.grid_def.items() if k not in known_keys}
            self.grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, **other_opts)

            ##mask for invalid grid points (none for now, add option later)
            if 'mask' in self.grid_def:
                model_name = self.grid_def['mask']
                module = importlib.import_module('NEDAS.models.'+model_name)
                model = getattr(module, 'Model')()
                self.grid.mask = model.prepare_mask(self.grid)
            else:
                self.grid.mask = np.full((self.grid.ny, self.grid.nx), False, dtype=bool)

        else:
            ##get analysis grid from model module
            model_name = self.grid_def['type']
            kwargs = self.model_def[model_name]
            module = importlib.import_module('NEDAS.models.'+model_name)
            model = getattr(module, 'Model')(**kwargs)
            self.grid = model.grid

    def set_model_config(self):
        ##initialize model config dict
        self.model_config = {}
        for model_name, kwargs in self.model_def.items():
            ##load model class instance
            module = importlib.import_module('NEDAS.models.'+model_name)
            if not isinstance(kwargs, dict):
                kwargs = {}
            self.model_config[model_name] = getattr(module, 'Model')(**kwargs)

    def set_dataset_config(self):
        ##initialize dataset config dict
        self.dataset_config = {}
        for dataset_name, kwargs in self.dataset_def.items():
            ##load dataset module
            module = importlib.import_module('NEDAS.dataset.'+dataset_name)
            if not isinstance(kwargs, dict):
                kwargs = {}
            self.dataset_config[dataset_name] = getattr(module, 'Dataset')(grid=self.grid, mask=self.grid.mask, **kwargs)

    def show_summary(self):
        ##print a summary
        print(f"""Initializing config...
 working directory: {self.work_dir}
 parallel scheme: nproc = {self.nproc}, nproc_mem = {self.nproc_mem}
 cycling from {self.time_start} to {self.time_end}
 analysis start at {self.time_analysis_start}
 cycle_period = {self.cycle_period} hours
 current time: {self.time}
 """, flush=True)

    def dump_yaml(self, config_file):
        config_dict = {}
        with open(config_file, 'w') as f:
            for key in self.keys:
                value = getattr(self, key)
                if key in ['time_start', 'time_end', 'time_analysis_start', 'time_analysis_end', 'time']:
                    value = t2s(value)
                config_dict[key] = value
            yaml.dump(config_dict, f)
