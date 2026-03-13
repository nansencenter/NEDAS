import os
import numpy as np
from NEDAS.grid import Grid1D
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.core import Model
from NEDAS.core.types import VarDesc, IOMode

def M_nl(x, F, T, dt):
    """
    Lorenz 1996 model with 40 variables, nonlinear advance_time function
    Input:
    -x: np.array, the model state
    -F: parameter, default is 8
    -T: duration of the simulation
    -dt: model time step
    Output:
    -x: np.array, the updated model state after simulation
    """
    for _ in range(int(T/dt)):
        x += ((np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F) * dt
    return x

class Lorenz96Model(Model[Grid1D]):
    io_mode: IOMode = 'online'  # both online and offline supported, default to online
    nx: int
    F: float
    dt: float
    restart_dt: float
    memory: dict = {}

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.grid = Grid1D.regular_grid(0, self.nx, 1, cyclic=True)
        self.grid.mask = np.full(self.grid.x.shape, False)

        self.variables = {
            'state': VarDesc(name='state', dtype='float', is_vector=False, dt=self.restart_dt, levels=np.array([0]), units='*', z_units='*'),
        }
        self.z = {0: np.zeros(self.nx)}

        self.io_methods = {
            'offline': {
                'read_var': self._read_var_from_file,
                'write_var': self._write_var_to_file,
            },
            'online': {
                'read_var': self._read_var_from_memory,
                'write_var': self._write_var_to_memory,
            }
        }
        if self.io_mode not in self.io_methods.keys():
            raise ValueError(f"Unknown io_mode {self.io_mode}")

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)

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
        return self.io_methods[self.io_mode]['read_var'](**kwargs)

    def _read_var_from_memory(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        tag = kwargs['tag']
        name = kwargs['name']
        member = kwargs['member']
        time = kwargs['time']
        key = (member, time)
        if key not in self.memory[tag][name]:
            raise KeyError(f'lorenz96 model online state: {key} not found in memory[{tag}][{name}]')
        return self.memory[tag][name][key]

    def _read_var_from_file(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        var_name = self.variables[name].name
        assert isinstance(var_name, str)
        var = nc_read_var(fname, var_name)[0, ...]
        return var

    def write_var(self, var, **kwargs):
        self.io_methods[self.io_mode]['write_var'](var, **kwargs)

    def _write_var_to_memory(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        tag = kwargs['tag']
        name = kwargs['name']
        member = kwargs['member']
        time = kwargs['time']
        ##create memory dict entry if not yet
        if tag not in self.memory:
            self.memory[tag] = {}
        if name not in self.memory[tag]:
            self.memory[tag][name] = {}
        self.memory[tag][name][member, time] = var

    def _write_var_to_file(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        var_name = self.variables[name].name
        assert isinstance(var_name, str)
        nc_write_var(fname, {'t':None, 'x':self.nx}, var_name, var, recno={'t':0})

    def z_coords(self, **kwargs):
        return self.z[kwargs['k']]

    def generate_initial_condition(self):
        state = np.random.normal(0, 1, self.nx)
        return state

    def preprocess(self, *args, **kwargs):
        if self.io_mode == 'offline':
            kwargs = super().parse_kwargs(kwargs)
            c = self.get_context(kwargs)

            c.io.make_dir(kwargs['path'])
            file1 = self.filename(**{**kwargs, 'path':kwargs['restart_dir']})
            file2 = self.filename(**kwargs)
            c.io.copy_file(file1, file2)

    def postprocess(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        self.run_status = 'running'

        state = self.read_var(**kwargs)
        next_time = kwargs['time'] + kwargs['forecast_period'] * dt1h
        next_state = M_nl(state, self.F, kwargs['forecast_period']/24, self.dt)
        self.write_var(next_state, **{**kwargs, 'time':next_time})

        self.run_status = 'complete'

    def generate_truth(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            c = self.get_context(kwargs)
            c.io.make_dir(self.truth_dir)
        state = self.generate_initial_condition()
        kwargs['time'] = kwargs['time_start']
        kwargs['member'] = None
        while kwargs['time'] <= kwargs['time_end']:
            self.write_var(state, name='state', **kwargs)
            state = M_nl(state, self.F, kwargs['forecast_period']/24, self.dt)
            kwargs['time'] += kwargs['forecast_period'] * dt1h

    def generate_init_ensemble(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            c = self.get_context(kwargs)
            c.io.make_dir(self.ens_init_dir)

        state = self.generate_initial_condition()
        kwargs['time'] = kwargs['time_start']
        self.write_var(state, name='state', **kwargs)
