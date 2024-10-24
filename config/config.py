import numpy as np
import os
import inspect
import yaml
import importlib
import pyproj
from grid import Grid
from utils.parallel import Comm
from utils.conversion import s2t, t2s
from .parse_config import parse_config

class Config(object):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
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

        ##these attributes will also be useful during runtime
        for key in ['state_info','mem_list','rec_list','partitions','obs_info','obs_rec_list','obs_inds','par_list']:
            setattr(self, key, None)

    def set_time(self):
        for key in ['time_start', 'time_end', 'time_assim_start', 'time_assim_end']:
            assert key in self.keys, f"'{key}' is missing in config file"

        ##these keys become available at runtime
        for key in ['time', 'prev_time', 'next_time']:
            if key not in self.keys:
                setattr(self, key, self.time_start)
                self.keys.append(key)

        ##convert string to datetime obj
        for key in ['time_start', 'time_end', 'time_assim_start', 'time_assim_end', 'time', 'prev_time', 'next_time']:
            setattr(self, key, s2t(getattr(self, key)))

    def set_comm(self):
        ##initialize mpi communicator
        self.comm = Comm()
        if not hasattr(self, 'nproc'):
            self.nproc = self.comm.Get_size()
        self.pid = self.comm.Get_rank()  ##current processor id

        ##divide processors into mem/rec groups
        if not hasattr(self, 'nproc_mem'):
            self.nproc_mem = self.nproc
        assert self.nproc % self.nproc_mem == 0, f"nproc {self.nproc} is not evenly divided by nproc_mem {self.nproc_mem}"
        self.nproc_rec = int(self.nproc/self.nproc_mem)
        self.pid_mem = self.pid % self.nproc_mem
        self.pid_rec = self.pid // self.nproc_mem
        self.comm_mem = self.comm.Split(self.pid_rec, self.pid_mem)
        self.comm_rec = self.comm.Split(self.pid_mem, self.pid_rec)

        self.pid_show = 0  ##which pid is showing progress messages, default to root=0

    def set_analysis_grid(self):
        ##initialize analysis grid
        if self.grid_def['type'] == 'custom':
            proj = pyproj.Proj(self.grid_def['proj'])
            xmin, xmax = self.grid_def['xmin'], self.grid_def['xmax']
            ymin, ymax = self.grid_def['ymin'], self.grid_def['ymax']
            dx = self.grid_def['dx']
            centered = self.grid_def.get('centered', False)
            self.grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, centered=centered)

            ##mask for invalid grid points (none for now, add option later)
            self.mask = np.full((self.grid.ny, self.grid.nx), False, dtype=bool)

        else:
            ##get analysis grid from model module
            model_name = self.grid_def['type']
            kwargs = self.model_def[model_name]
            module = importlib.import_module('models.'+model_name)
            model = getattr(module, 'Model')(**kwargs)
            self.grid = model.grid
            self.mask = model.mask

        if self.grid.regular:
            self.ny, self.nx = self.grid.x.shape
        else:
            self.npoints = self.grid.x.size

    def set_model_config(self):
        ##initialize model config dict
        self.model_config = {}
        for model_name, kwargs in self.model_def.items():
            ##load model class instance
            module = importlib.import_module('models.'+model_name)
            self.model_config[model_name] = getattr(module, 'Model')(**kwargs)

    def show_summary(self):
        ##print a summary
        print(f"""Initializing config...
 working directory: {self.work_dir}
 parallel scheme: nproc = {self.nproc}, nproc_mem = {self.nproc_mem}
 cycling from {self.time_start} to {self.time_end}
 assimilation start at {self.time_assim_start}
 cycle_period = {self.cycle_period} hours
 current time: {self.time}
 """, flush=True)

    def dump_yaml(self, config_file):
        config_dict = {}
        with open(config_file, 'w') as f:
            for key in self.keys:
                value = getattr(self, key)
                if key in ['time_start', 'time_end', 'time_assim_start', 'time_assim_end', 'time', 'prev_time', 'next_time']:
                    value = t2s(value)
                config_dict[key] = value
            yaml.dump(config_dict, f)

