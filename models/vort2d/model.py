import numpy as np
import os
import inspect
import time
import multiprocessing

from pyproj import Proj
from grid import Grid
from config import parse_config
from utils.conversion import t2s
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

        self.uniq_grid_key = ()
        self.uniq_z_key = ()
        self.z_units = '*'

        self.run_process = None
        self.run_status = 'pending'


    def filename(self, **kwargs):
        """parse kwargs and find matching filename"""
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'

        if 'member' in kwargs and kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        assert 'time' in kwargs, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')

        return path+'/'+tstr+mstr+'.nc'


    def read_grid(self, **kwargs):
        return self.grid


    def write_grid(self, grid, **kwargs):
        pass


    def read_mask(self, **kwargs):
        return self.mask


    def read_var(self, **kwargs):
        ##check name in kwargs and read the variables from file
        assert 'name' in kwargs, 'please specify which variable to get, name=?'
        name = kwargs['name']
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = filename(**kwargs)

        is_vector = self.variables[name]['is_vector']
        if is_vector:
            var1 = nc_read_var(fname, self.variables[name]['name'][0])[0, ...]
            var2 = nc_read_var(fname, self.variables[name]['name'][1])[0, ...]
            var = np.array([var1, var2])
        else:
            var = nc_read_var(fname, self.variables[name]['name'])[0, ...]
        return var


    def write_var(self, var, **kwargs):
        ##check kwargs
        assert 'name' in kwargs, 'missing name in kwargs'
        name = kwargs['name']
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = filename(**kwargs)

        is_vector = self.variables[name]['is_vector']
        if is_vector:
            for i in range(2):
                nc_write_var(fname, {'t':None, 'y':self.ny, 'x':self.nx}, self.variables[name]['name'][i], var[i,...], recno={'t':0})
        else:
            nc_write_var(fname, {'t':None, 'y':self.ny, 'x':self.nx}, var, self.variables[name]['name'], var, recno={'t':0})


    def z_coords(**kwargs):
        return np.zeros(self.grid.x.shape)


    def run(self, task_id=0, task_nproc=1, **kwargs):
        assert task_nproc==1, f"vort2d model only support serial runs (got task_nproc={task_nproc})"
        self.run_status = 'running'

##model.run()
##p = multiprocess.Process(target=func, args)
##p.run()
##p.kill()

