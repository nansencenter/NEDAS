import os
import numpy as np
from NEDAS.grid import Grid1D
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.core import Model
from NEDAS.core.types import VarDesc, IOMode

class Lorenz96Model(Model[Grid1D]):
    """
    Lorenz 1996 model with 40 variables

    Args:
        nx (int): dimension of the model, default is 40.
        F (float): forcing parameter, default is 8
        F_std (float): std for per-member F perturbation in generate_init_ensemble; 0 disables SSPE
        dt (float): model time step, default is 0.05
        numeric_opt (str): numeric option, default is 'rk4'
    """
    io_mode: IOMode = 'online'  # both online and offline supported, default to online
    nx: int
    F: float
    F_std: float
    dt: float
    restart_dt: float
    numeric_opt: str
    memory: dict = {}

    # scalar parameters available for SSPE (name → metadata dict)
    params = {
        'F': {'dtype': 'float', 'units': '*'},
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.grid = Grid1D.regular_grid(0, self.nx, 1, cyclic=True)
        self.grid.mask = np.full(self.grid.x.shape, False)

        self.variables = {
            'state': VarDesc(name='state', dtype='float', is_vector=False, dt=self.restart_dt, levels=np.array([0]), units='*', z_units='*'),
        }
        self.z = {0: np.zeros(self.nx)}

        self.run_1step_funcs = {
            'euler_forward': self.run_1step_euler_forward,
            'rk4': self.run_1step_rk4,
        }
        assert self.numeric_opt in self.run_1step_funcs, f"{self.__class__.__name__}: unknown numerics '{self.numeric_opt}'."
        self.run_1step = self.run_1step_funcs[self.numeric_opt]  # signature: (x, F=None)

        # convention to real time for the nondimensional model
        # 6 h in meteorological models is representative by t = 0.05
        # so 120 hours per unit model time
        self.hours_per_unit_time = 120.

    def dxdt(self, x, F=None):
        F = self.F if F is None else F
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    def run_1step_euler_forward(self, x, F=None):
        return x + self.dt * self.dxdt(x, F)

    def run_1step_rk4(self, x, F=None):
        dx1 = self.dt * self.dxdt(x, F)
        dx2 = self.dt * self.dxdt(x + dx1/2.0, F)
        dx3 = self.dt * self.dxdt(x + dx2/2.0, F)
        dx4 = self.dt * self.dxdt(x + dx3, F)
        return x + (dx1 + 2.0*dx2 + 2.0*dx3 + dx4)/6.0

    def advance_time(self, x_in, T, F=None):
        """
        Nonlinear advance_time function

        Args:
            x_in (np.ndarray): the initial condition
            T (float): duration of the simulation
            F (float, optional): forcing parameter; uses self.F if not given

        Return:
            np.ndarray: the updated model state after simulation
        """
        if np.isnan(x_in).any():
            raise RuntimeError('NaN detected in lorenz96 model state. Aborting...')
        if np.isinf(x_in).any():
            raise RuntimeError('Inf detected in lorenz96 model state. Aborting...')
        x = x_in.copy()
        for _ in range(int(T/self.dt)):
            x = self.run_1step(x, F)
        return x

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

        # use per-member forcing parameter if available (falls back to self.F)
        F_m = self.read_param(name='F', member=kwargs['member'])

        state = self.read_var(**kwargs)
        next_time = kwargs['time'] + kwargs['forecast_period'] * dt1h
        next_state = self.advance_time(state, kwargs['forecast_period']/self.hours_per_unit_time, F=F_m)
        self.write_var(next_state, **{**kwargs, 'time':next_time})

        self.run_status = 'complete'

    def generate_truth(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            self.c.fs.make_dir(self.truth_dir)
        state = self.generate_initial_condition()
        kwargs['time'] = self.c.config.time_start
        kwargs['member'] = None
        while kwargs['time'] <= self.c.config.time_end:
            self.write_var(state, **kwargs)
            state = self.advance_time(state, kwargs['forecast_period']/self.hours_per_unit_time)
            kwargs['time'] += kwargs['forecast_period'] * dt1h

    def generate_init_ensemble(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            self.c.fs.make_dir(self.ens_init_dir)
        state = self.generate_initial_condition()
        kwargs['time'] = self.c.config.time_start
        kwargs['path'] = self.ens_init_dir
        self.write_var(state, **kwargs)

        # perturb forcing parameter for this member if F_std > 0
        m = kwargs.get('member')
        if self.F_std > 0 and m is not None:
            rng = np.random.default_rng(seed=m)  # per-member seed
            self.write_param(self.F + rng.normal(0, self.F_std), name='F', member=m)
