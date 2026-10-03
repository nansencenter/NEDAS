"""
Base class for small dynamical systems (ODEs, maps and 1D PDEs) used to test DA methods.

These are the toy models of DART (models/lorenz_63, 9var, ikeda, ...) and DAPPER
(dapper/mods): their whole state is a short vector, so a subclass only defines the dynamics
(`dxdt`, or `step` for a map or a custom scheme), a reference state `x0` and default settings
in its own default.yml. Everything else, the io and the truth and ensemble generation, is here.

The state vector is the model variable 'state' on a 1D grid of its indices (Grid1D, x = 0..n-1),
so synthetic observations are placed by `obs_x` index, and localization distances are in index
units. A subclass with more than one kind of variable (lorenz96_2scale) overrides
`get_field` and `set_field` to map the state vector to its variables on the grid.

Both io modes are supported: online (default) keeps the state vector of each member in
self.memory, offline in a netCDF file per member and time (variable 'state').

Model time is nondimensional; `hours_per_unit_time` relates it to the hours NEDAS cycles in,
following the convention of Lorenz (1996) that 0.05 time units is 6 hours (120 h per unit).
"""
import os
import numpy as np
from NEDAS.grid import Grid1D
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.core import Model
from NEDAS.core.types import VarDesc, IOMode


class OdeModel(Model[Grid1D]):
    """
    Base class of the small models: a state vector advanced by `step`.

    Settings common to all of them (in each model's default.yml):

    Args:
        dt (float): time step in model time units (1 for a map)
        hours_per_unit_time (float): hours per model time unit
        restart_dt (float): restart interval in hours
        spinup_time (float): model time to run from a perturbed `x0` to get a state on the attractor
        init_sd (float): standard deviation of the random perturbation added to `x0` before the spinup
        init_ens_mode (str): 'climatology' for members that are independently spun up states, or
            'truth' for the truth at the start time plus noise of standard deviation `init_ens_sd`
        init_ens_sd (float): see `init_ens_mode`
        seed (int): random seed for the truth; member m uses seed + 1 + m
    """
    io_mode: IOMode = 'online'
    dt: float
    hours_per_unit_time: float
    restart_dt: float
    spinup_time: float
    init_sd: float
    init_ens_mode: str
    init_ens_sd: float
    seed: int
    memory: dict = {}
    cyclic: bool = False  # whether the state vector is periodic (for distances on the grid)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory = {}  # not the class-level dict shared by all instances
        assert self.init_ens_mode in ('climatology', 'truth'), f"unknown init_ens_mode '{self.init_ens_mode}'"

        n = self.grid_size()
        self.grid = Grid1D.regular_grid(0, n, 1, cyclic=self.cyclic)
        self.grid.mask = np.full(self.grid.x.shape, False)
        self.z = {0: np.zeros(n)}
        self.variables = {
            name: VarDesc(name=name, dtype='float', is_vector=False, dt=self.restart_dt,
                          levels=np.array([0]), units='*', z_units='*')
            for name in self.field_names()
        }

    # --- what a subclass defines --------------------------------------------------------------

    @property
    def state_size(self) -> int:
        """length of the state vector"""
        raise NotImplementedError

    def x0(self) -> np.ndarray:
        """a reference state, perturbed and spun up to start the truth and the ensemble"""
        raise NotImplementedError

    def dxdt(self, x: np.ndarray) -> np.ndarray:
        """the time derivative of the state (for an ODE stepped by `step`)"""
        raise NotImplementedError

    def step(self, x: np.ndarray) -> np.ndarray:
        """one time step of length dt; RK4 of dxdt unless a subclass has its own scheme"""
        k1 = self.dxdt(x)
        k2 = self.dxdt(x + 0.5 * self.dt * k1)
        k3 = self.dxdt(x + 0.5 * self.dt * k2)
        k4 = self.dxdt(x + self.dt * k3)
        return x + self.dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.

    def grid_size(self) -> int:
        """number of grid points; the state vector itself unless a subclass maps it otherwise"""
        return self.state_size

    def field_names(self) -> list[str]:
        """the NEDAS variables of the model"""
        return ['state']

    def get_field(self, x: np.ndarray, name: str) -> np.ndarray:
        """the variable `name` on the grid, from a state vector"""
        assert name == 'state', f"unknown variable '{name}'"
        return x.copy()

    def set_field(self, x: np.ndarray, name: str, fld: np.ndarray) -> None:
        """set the variable `name` in a state vector, in place"""
        assert name == 'state', f"unknown variable '{name}'"
        x[:] = fld

    # --- time integration ---------------------------------------------------------------------

    def nsteps(self, hours: float) -> int:
        """number of time steps in an interval of the given hours"""
        n = hours / self.hours_per_unit_time / self.dt
        assert abs(n - round(n)) < 1e-6, \
            f"{hours} h is not a multiple of the model time step ({self.dt * self.hours_per_unit_time} h)"
        return int(round(n))

    def advance(self, x: np.ndarray, nsteps: int) -> np.ndarray:
        """the state after nsteps time steps from x (x is not changed)"""
        x = np.array(x, dtype=float)
        for _ in range(nsteps):
            x = self.step(x)
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"{self.__class__.__name__}: non-finite model state, the time step may be too long")
        return x

    def initial_condition(self, seed: int) -> np.ndarray:
        """a state on the attractor: x0 plus noise of init_sd, spun up for spinup_time"""
        rng = np.random.default_rng(seed)
        x = self.x0() + rng.normal(0, self.init_sd, self.state_size)
        return self.advance(x, int(round(self.spinup_time / self.dt)))

    # --- io -----------------------------------------------------------------------------------

    def filename(self, **kwargs) -> str:
        kwargs = super().parse_kwargs(kwargs)
        return os.path.join(kwargs['path'], self.get_tstr(kwargs['time']) + self.get_mstr(kwargs['member']) + '.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def z_coords(self, **kwargs) -> np.ndarray:
        return self.z[0]

    def get_state(self, **kwargs) -> np.ndarray:
        """the state vector at kwargs time and member (and tag, in memory)"""
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            return nc_read_var(self.filename(**kwargs), 'state')[0, ...]
        tstr, key = self.get_tstr(kwargs['time']), kwargs['tag'] + self.get_mstr(kwargs['member'])
        try:
            return self.memory[tstr][key]['_state']
        except KeyError:
            raise KeyError(f"{self.__class__.__name__}: no state in memory['{tstr}']['{key}']") from None

    def set_state(self, x: np.ndarray, **kwargs) -> None:
        """store the state vector at kwargs time and member (and tag, in memory)"""
        kwargs = super().parse_kwargs(kwargs)
        assert x.shape == (self.state_size,), f"state vector shape {x.shape}, expected ({self.state_size},)"
        if self.io_mode == 'offline':
            self.c.fs.make_dir(kwargs['path'])
            nc_write_var(self.filename(**kwargs), {'t': None, 'n': self.state_size}, 'state', x, recno={'t': 0})
            return
        tstr, key = self.get_tstr(kwargs['time']), kwargs['tag'] + self.get_mstr(kwargs['member'])
        self.memory.setdefault(tstr, {}).setdefault(key, {})['_state'] = x

    def read_var_from_file(self, **kwargs) -> np.ndarray:
        kwargs = super().parse_kwargs(kwargs)
        return self.get_field(self.get_state(**kwargs), kwargs['name'])

    def read_var_from_memory(self, **kwargs) -> np.ndarray:
        return self.read_var_from_file(**kwargs)

    def write_var_to_file(self, var, **kwargs) -> None:
        kwargs = super().parse_kwargs(kwargs)
        x = np.array(self.get_state(**kwargs))
        self.set_field(x, kwargs['name'], np.asarray(var))
        self.set_state(x, **kwargs)

    def write_var_to_memory(self, var, **kwargs) -> None:
        self.write_var_to_file(var, **kwargs)

    # --- the NEDAS workflow -------------------------------------------------------------------

    def preprocess(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            self.c.fs.make_dir(kwargs['path'])
            self.c.fs.copy_file(self.filename(**{**kwargs, 'path': kwargs['restart_dir']}), self.filename(**kwargs))
        else:
            # keep a copy of the forecast as the prior
            self.set_state(self.get_state(**kwargs).copy(), **{**kwargs, 'tag': 'prior'})

    def postprocess(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        # offline, the files are the posterior states already
        if self.io_mode == 'online':
            self.set_state(self.get_state(**kwargs), **{**kwargs, 'tag': 'post'})

    def run(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        self.run_status = 'running'
        next_time = kwargs['time'] + kwargs['forecast_period'] * dt1h
        x = self.advance(self.get_state(**kwargs), self.nsteps(kwargs['forecast_period']))
        self.set_state(x, **{**kwargs, 'time': next_time})
        self.run_status = 'complete'

    def generate_truth(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        opts = {**kwargs, 'member': None}
        if self.io_mode == 'offline':
            assert self.truth_dir is not None
            opts['path'] = self.truth_dir
        time = self.c.config.time_start
        x = self.initial_condition(self.seed)
        nsteps = self.nsteps(kwargs['forecast_period'])
        while time <= self.c.config.time_end:
            self.set_state(x, **{**opts, 'time': time})
            x = self.advance(x, nsteps)
            time += kwargs['forecast_period'] * dt1h

    def generate_init_ensemble(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        member = kwargs['member'] if kwargs['member'] is not None else 0
        opts = {**kwargs, 'time': self.c.config.time_start}
        if self.io_mode == 'offline':
            assert self.ens_init_dir is not None
            opts['path'] = self.ens_init_dir
        seed = self.seed + 1 + member
        if self.init_ens_mode == 'truth':
            truth_opts = {**opts, 'member': None, 'tag': 'truth'}
            if self.io_mode == 'offline':
                assert self.truth_dir is not None
                truth_opts['path'] = self.truth_dir
            rng = np.random.default_rng(seed)
            x = self.get_state(**truth_opts) + rng.normal(0, self.init_ens_sd, self.state_size)
        else:
            x = self.initial_condition(seed)
        self.set_state(x, **opts)
