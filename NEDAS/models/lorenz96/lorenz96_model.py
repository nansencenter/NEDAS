import os
from typing import Literal
import numpy as np
from NEDAS.grid import Grid1D
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.shell_utils import run_command, makedir
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from .core import M_nl
from NEDAS.models import Model

class Lorenz96Model(Model):
    io_mode: Literal['online', 'offline']
    nx: int
    F: float
    dt: float
    restart_dt: float

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.grid = Grid1D.regular_grid(0, self.nx, 1, cyclic=True)
        self.grid.mask = np.full(self.grid.x.shape, False)

        self.variables = {'state': {'name':'state', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'*'}, }
        self.z_units = '*'

        self.run_process = None
        self.run_status = 'pending'

        if self.io_mode == 'online':
            self.memory = {}
            self.read_var = self._read_var_from_memory
            self.write_var = self._write_var_to_memory
            self.preprocess = self._preprocess_in_memory

        elif self.io_mode == 'offline':
            self.read_var = self._read_var_from_file
            self.write_var = self._write_var_to_file
            self.preprocess = self._preprocess_restartfiles

        else:
            raise ValueError(f"Unknown io_mode {self.io_mode}")

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

    def _read_var_from_memory(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        member = kwargs['member']
        time = kwargs['time']
        if name not in self.memory:
            raise RuntimeError('lorenz96 model online state memory not allocated yet.')
        key = (member, time)
        if key not in self.memory[name]:
            raise RuntimeError(f'lorenz96 model online state: {key} not found in memory for {name}')
        return self.memory[name][key]

    def _read_var_from_file(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        var = nc_read_var(fname, self.variables[name]['name'])[0, ...]
        return var

    def _write_var_to_memory(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        member = kwargs['member']
        time = kwargs['time']
        ##create memory dict entry if not yet
        if name not in self.memory:
            self.memory[name] = {}
        self.memory[name][member, time] = var

    def _write_var_to_file(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        nc_write_var(fname, {'t':None, 'x':self.nx}, self.variables[name]['name'], var, recno={'t':0})

    def z_coords(self, **kwargs):
        return np.zeros(self.nx)

    def generate_initial_condition(self):
        state = np.random.normal(0, 1, self.nx)
        return state

    def _preprocess_restartfiles(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        makedir(kwargs['path'])
        file1 = self.filename(**{**kwargs, 'path':kwargs['restart_dir']})
        file2 = self.filename(**kwargs)
        run_command(f"cp -fL {file1} {file2}")

    def _preprocess_in_memory(self, task_id=0, **kwargs):
        pass

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        state = self.read_var(**kwargs)
        next_time = kwargs['time'] + kwargs['forecast_period'] * dt1h
        next_state = M_nl(state, self.F, kwargs['forecast_period']/24, self.dt)
        self.write_var(next_state, **{**kwargs, 'time':next_time})

        self.run_status = 'complete'
