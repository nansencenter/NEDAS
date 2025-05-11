import os
import numpy as np
from NEDAS.grid import Grid1D
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.shell_utils import run_command, makedir
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from .core import M_nl
from NEDAS.models import Model

class Lorenz96Model(Model):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.grid = Grid1D.regular_grid(0, self.nx, 1, cyclic=True)
        self.grid.mask = np.full(self.grid.x.shape, False)

        self.variables = {'state': {'name':'state', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'*'}, }
        self.z_units = '*'

        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        if kwargs['member'] is not None:
            mstr = '_mem{:04d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        assert kwargs['time'] is not None, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')

        return os.path.join(kwargs['path'], tstr+mstr+'.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        var = nc_read_var(fname, self.variables[name]['name'])[0, ...]
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        nc_write_var(fname, {'t':None, 'x':self.nx}, self.variables[name]['name'], var, recno={'t':0})

    def z_coords(self, **kwargs):
        return np.zeros(self.nx)

    def generate_initial_condition(self):
        state = np.random.normal(0, 1, self.nx)
        return state

    def preprocess(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        makedir(kwargs['path'])
        file1 = self.filename(**{**kwargs, 'path':kwargs['restart_dir']})
        file2 = self.filename(**kwargs)
        run_command(f"cp -fL {file1} {file2}")

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        state = self.read_var(**kwargs)

        makedir(kwargs['path'])

        next_time = kwargs['time'] + kwargs['forecast_period'] * dt1h

        next_state = M_nl(state, self.F, kwargs['forecast_period']/24, self.dt)

        self.write_var(next_state, **{**kwargs, 'time':next_time})

        self.run_status = 'complete'

