import numpy as np
import os
import inspect
import time
import subprocess

from pyproj import Proj
from grid import Grid
from config import parse_config
from utils.conversion import t2s, s2t, dt1h
from utils.netcdf_lib import nc_read_var, nc_write_var

from .util import initial_condition, advance_time

class Model(object):
    def __init__(self, config_file=None, parse_args=False, **kwargs):

        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        ##define the model grid
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        x = ii*self.dx
        y = jj*self.dx
        self.grid = Grid(Proj('+proj=stere'), x, y, cyclic_dim='xy')
        self.mask = np.full(self.grid.x.shape, False)  ##no mask

        levels = np.array([0])  ##there is no vertical levels

        self.variables = {'velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'restart_dt':self.restart_dt, 'levels':levels, 'units':'m/s'}, }

        self.z_units = '*'

        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs):
        """parse kwargs and find matching filename"""
        path = kwargs.get('path', '.')

        if 'member' in kwargs and kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        assert 'time' in kwargs, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')

        return os.path.join(path, tstr+mstr+'.nc')

    def read_grid(self, **kwargs):
        return self.grid

    def read_mask(self, **kwargs):
        return self.mask

    def read_var(self, **kwargs):
        ##check name in kwargs and read the variables from file
        name = kwargs.get('name', 'velocity')
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = self.filename(**kwargs)

        is_vector = self.variables[name]['is_vector']
        if is_vector:
            var1 = nc_read_var(fname, self.variables[name]['name'][0])[0, ...]
            var2 = nc_read_var(fname, self.variables[name]['name'][1])[0, ...]
            var = np.array([var1, var2])
        else:
            var = nc_read_var(fname, self.variables[name]['name'])[0, ...]
        return var

    def write_var(self, var, **kwargs):
        ##TODO: nc_write_var doesn't support parallel=True
        ##      can only work with nproc=1 now
        ##check kwargs
        name = kwargs.get('name', 'velocity')
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = self.filename(**kwargs)

        is_vector = self.variables[name]['is_vector']
        if is_vector:
            for i in range(2):
                nc_write_var(fname, {'t':None, 'y':self.ny, 'x':self.nx}, self.variables[name]['name'][i], var[i,...], recno={'t':0})
        else:
            nc_write_var(fname, {'t':None, 'y':self.ny, 'x':self.nx}, var, self.variables[name]['name'], var, recno={'t':0})

    def z_coords(self, **kwargs):
        return np.zeros(self.grid.x.shape)

    def generate_initial_condition(self):
        state = initial_condition(self.grid, self.Vmax, self.Rmw, self.Vbg, self.Vslope, self.loc_sprd)
        return state

    def preprocess(self, task_id=0, **kwargs):
        restart_dir = kwargs['restart_dir']
        path = kwargs['path']
        os.system("mkdir -p "+path)
        file1 = self.filename(**{**kwargs, 'path':restart_dir})
        file2 = self.filename(**kwargs)
        os.system(f"cp -fL {file1} {file2}")

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, **kwargs):
        self.run_status = 'running'
        state = self.read_var(**kwargs)

        path = kwargs['path']
        subprocess.run("mkdir -p "+path, shell=True)

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        next_state = advance_time(state, self.dx, forecast_period, self.dt, self.gen, self.diss)

        kwargs_out = {**kwargs, 'time':next_time}
        output_file = self.filename(**kwargs_out)
        self.write_var(next_state, **kwargs_out)

