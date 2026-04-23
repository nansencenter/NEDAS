import os
import numpy as np
from NEDAS.grid import Grid1D
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.core import Model
from NEDAS.core.types import VarDesc, IOMode

def M_nl(x_in, F, T, dt):
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
    x = x_in.copy()
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.grid = Grid1D.regular_grid(0, self.nx, 1, cyclic=True)
        self.grid.mask = np.full(self.grid.x.shape, False)

        self.variables = {
            'state': VarDesc(name='state', dtype='float', is_vector=False, dt=self.restart_dt, levels=np.array([0]), units='*', z_units='*'),
        }
        self.z = {0: np.zeros(self.nx)}

        # convention to real time for the nondimensional model
        # 6 h in meteorological models is representative by t = 0.05
        # so 120 hours per unit model time
        self.hours_per_unit_time = 120.

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        mstr = self.get_mstr(kwargs['member'])
        tstr = self.get_tstr(kwargs['time'])
        return os.path.join(kwargs['path'], tstr+mstr+'.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var_from_file(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        var_name = self.variables[name].name
        assert isinstance(var_name, str)
        var = nc_read_var(fname, var_name)[0, ...]
        return var

    def write_var_to_file(self, var, **kwargs):
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
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            self.c.fs.make_dir(kwargs['path'])
            file1 = self.filename(**{**kwargs, 'path':kwargs['restart_dir']})
            file2 = self.filename(**kwargs)
            self.c.fs.copy_file(file1, file2)
        elif self.io_mode == 'online':
            # save a copy of the current state (forecast) as prior
            var = self.read_var_from_memory(**kwargs)
            self.write_var_to_memory(var.copy(), **{**kwargs, 'tag':'prior'})

    def postprocess(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        # if offline mode, the current files are just posterior states
        # do nothing
        if self.io_mode == 'online':
            # save a copy of the current state (analysis) as posterior
            # don't need to copy since current state is no longer updated
            var = self.read_var_from_memory(**kwargs)
            self.write_var_to_memory(var, **{**kwargs, 'tag':'post'})

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
        self.c.fs.make_dir(self.truth_dir)
        state = self.generate_initial_condition()
        kwargs['time'] = self.c.config.time_start
        kwargs['member'] = None
        while kwargs['time'] <= self.c.config.time_end:
            self.write_var(state, **kwargs)
            state = M_nl(state, self.F, kwargs['forecast_period']/self.hours_per_unit_time, self.dt)
            kwargs['time'] += kwargs['forecast_period'] * dt1h

    def generate_init_ensemble(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        self.c.fs.make_dir(self.ens_init_dir)
        state = self.generate_initial_condition()
        kwargs['time'] = self.c.config.time_start
        kwargs['path'] = self.ens_init_dir
        self.write_var(state, **kwargs)
