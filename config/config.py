import numpy as np
import os
import inspect
import importlib
import pyproj

from .parse_config import parse_config
from grid import Grid
from utils.parallel import Comm
from utils.conversion import s2t
from utils.log import message

class Config(object):

    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        ##create work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        ##convert time string to datetime object
        self.time_start = s2t(self.time_start)
        self.time_end = s2t(self.time_end)
        self.time_assim_start = s2t(self.time_assim_start)
        self.time_assim_end = s2t(self.time_assim_end)
        if hasattr(self, 'time'):
            self.time = s2t(self.time)
        else:
            self.time = self.time_start

        ##initialize mpi communicator
        self.comm = Comm()

        if hasattr(self, 'nproc'):
            assert self.nproc == self.comm.Get_size(), f"nproc {self.comm.Get_size()} is not the same as defined in config {self.nproc}"
        else:
            self.nproc = self.comm.Get_size()

        self.pid = self.comm.Get_rank()  ##current processor id

        ##divide processors into mem/rec groups
        if not hasattr(self, 'nproc_mem'):
            self.nproc_mem = self.nproc
        assert self.nproc % self.nproc_mem == 0, "nproc {self.nproc} is not evenly divided by nproc_mem {self.nproc_mem}"
        self.nproc_rec = int(self.nproc/self.nproc_mem)

        self.pid_mem = self.pid % self.nproc_mem
        self.pid_rec = self.pid // self.nproc_mem
        self.comm_mem = self.comm.Split(self.pid_rec, self.pid_mem)
        self.comm_rec = self.comm.Split(self.pid_mem, self.pid_rec)

        self.pid_show = 0  ##which pid is showing progress messages, default to root=0


        ##initialize analysis grid
        if self.grid_def['type'] == 'custom':
            proj = pyproj.Proj(self.grid_def['proj'])
            xmin, xmax = self.grid_def['xmin'], self.grid_def['xmax']
            ymin, ymax = self.grid_def['ymin'], self.grid_def['ymax']
            dx = self.grid_def['dx']
            self.grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, centered=True)

        else:
            ##get analysis grid from model module
            model_name = self.grid_def['type']
            module = importlib.import_module('models.'+model_name)
            model_dir = os.path.join(self.data_dir, model_name)
            m = getattr(module, 'Model')()
            self.grid = m.read_grid(model_dir)

        ##mask for invalid grid points
        # if self.mask
        self.mask = np.full((self.grid.ny, self.grid.nx), False, dtype=bool)

        ##print a summary
        message(self.comm, f"Initializing config...\n\n working directory: {self.work_dir}\n\n parallel scheme: nproc = {self.nproc}, nproc_mem = {self.nproc_mem}\n\n analysis grid: type = {self.grid_def['type']}, shape = {self.grid.x.shape}\n\n cycling from {self.time_start} to {self.time_end}\n assimilation start at {self.time_assim_start}, cycle_period = {self.cycle_period} hours\n\n", self.pid_show)


