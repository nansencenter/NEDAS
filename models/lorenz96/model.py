import numpy as np
import os
import inspect
from grid import Grid1D
from config import parse_config
from utils.conversion import t2s, s2t, dt1h
from utils.shell_utils import run_command
from utils.netcdf_lib import nc_read_var, nc_write_var
from .core import M_nl

class L96Model(object):
    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.grid = Grid1D.regular_grid(0, self.nx, 1, cyclic=True)
        self.mask = np.full(self.grid.x.shape, False)

        self.variables = {'state': {'name':'state', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'*'}, }
        self.z_units = '*'

        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs):
        path = kwargs.get('path', '.')

        if 'member' in kwargs and kwargs['member'] is not None:
            mstr = '_mem{:04d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        assert 'time' in kwargs, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')

        return os.path.join(path, tstr+mstr+'.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var(self, **kwargs):
        name = kwargs.get('name', 'state')
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = self.filename(**kwargs)
        var = nc_read_var(fname, self.variables[name]['name'])[0, ...]
        return var

    def write_var(self, var, **kwargs):
        name = kwargs.get('name', 'state')
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = self.filename(**kwargs)
        nc_write_var(fname, {'t':None, 'x':self.nx}, self.variables[name]['name'], var, recno={'t':0})

    def z_coords(self, **kwargs):
        return np.zeros(self.nx)

    def generate_initial_condition(self):
        state = np.random.normal(0, 1, self.nx)
        return state

    def preprocess(self, task_id=0, **kwargs):
        restart_dir = kwargs['restart_dir']
        path = kwargs['path']
        run_command("mkdir -p "+path)
        file1 = self.filename(**{**kwargs, 'path':restart_dir})
        file2 = self.filename(**kwargs)
        run_command(f"cp -fL {file1} {file2}")

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, **kwargs):
        self.run_status = 'running'

        state = self.read_var(**kwargs)

        path = kwargs['path']
        run_command("mkdir -p "+path)

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        next_state = M_nl(state, self.F, forecast_period/24, self.dt)

        self.write_var(next_state, **{**kwargs, 'time':next_time})

        self.run_status = 'complete'

