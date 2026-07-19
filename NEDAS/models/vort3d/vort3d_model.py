import os
import numpy as np
from NEDAS.grid import RegularGrid
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.core import Model
from NEDAS.core.types import VarDesc
from .core import SOUNDING_P
from .util import LAYER_NAMES, initial_condition, advance_time


class Vort3DModel(Model[RegularGrid]):
    """
    Zhu, Smith & Ulrich (2001) minimal 3D tropical cyclone model: sigma-
    coordinate primitive equations on an f/beta-plane, 3 layers (upper
    troposphere, lower/mid troposphere, boundary layer) + surface fluxes,
    radiative cooling, explicit condensation, and the Ooyama (1969)
    convective closure. See ~/Google_Drive/papers/2024.NEDAS.Introduction/
    vort3d/ for the standalone prototype and validation this was ported
    from (dev log: techNotes/models/vort3d.md).

    State is 3 layers x 4 fields (u,v,theta,q) + one 2D field (p*, column
    mass) -- represented here as separate NEDAS variables per layer
    ('wind_1','theta_1','q_1', etc, suffixes 1/3/b for
    upper-troposphere/lower-mid-troposphere/boundary-layer) plus 'pstar',
    rather than a single multi-level variable -- the three layers are
    physically distinct (different typical error/localization behavior),
    and this keeps read/write as simple single-array-per-name operations
    matching vort2d/lorenz96's pattern instead of needing a
    read-modify-write per level within one native netCDF variable.

    Args:
        nx, ny (int): grid dimensions (paper: 200x200)
        dx (float): grid spacing, m (paper: 20000)
        dt (float): internal model integration time step, s (paper: 15)
        restart_dt (float): restart/output interval, hours
        beta (float): df/dy, Coriolis beta parameter, /m/s (0 = pure
            f-plane, matching the paper's own experiments; >0 enables
            beta-drift)
        moist (bool): if False, runs the dry dynamical core only (no
            surface fluxes, radiative cooling, condensation, or convection)
        Vbg (float): random background-flow wind speed amplitude, m/s
            (0 = calm, matching the paper's own experiments)
        Vslope (float): background-flow kinetic-energy spectrum power law
        bg_seed (int|None): RNG seed for the background flow; if None,
            generate_init_ensemble uses the member index as the seed (so
            each ensemble member gets an independent background-flow
            realization -- the only source of initial-condition spread
            currently implemented; the vortex itself is not yet
            randomized in position/intensity, unlike vort2d's loc_sprd).
    """
    nx: int
    ny: int
    dx: float
    dt: float
    restart_dt: float
    beta: float
    moist: bool
    Vbg: float
    Vslope: float
    bg_seed: int | None
    memory: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # `memory: dict = {}` above is a class-level attribute; Model.__init__
        # never assigns self.memory, so without this line every instance in
        # the process would share the SAME dict object (same pitfall fixed
        # in vort2d_model.py/lorenz96_model.py -- see those for the history).
        self.memory = {}

        # define the model grid: periodic in x, rigid wall in y (the
        # paper's zonal-channel domain, section 2e) -- NOT doubly periodic
        # like vort2d's cyclic_dim='xy'.
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        x = ii * self.dx
        y = jj * self.dx
        self.grid = RegularGrid(None, x, y, cyclic_dim='x')
        self.grid.mask = np.full(self.grid.x.shape, False)  # no mask

        levels = np.array([0])
        self.variables = {}
        for name in LAYER_NAMES:
            self.variables[f'wind_{name}'] = VarDesc(
                name=(f'u{name}', f'v{name}'), dtype='float', is_vector=True,
                dt=self.restart_dt, levels=levels, units='m/s', z_units='hPa')
            self.variables[f'theta_{name}'] = VarDesc(
                name=f'theta{name}', dtype='float', is_vector=False,
                dt=self.restart_dt, levels=levels, units='K', z_units='hPa')
            self.variables[f'q_{name}'] = VarDesc(
                name=f'q{name}', dtype='float', is_vector=False,
                dt=self.restart_dt, levels=levels, units='kg/kg', z_units='hPa')
        self.variables['pstar'] = VarDesc(
            name='pstar', dtype='float', is_vector=False,
            dt=self.restart_dt, levels=levels, units='Pa', z_units='hPa')

        # nominal pressure of each layer, from the paper's Appendix A far-
        # field sounding (Table A1) -- a static z-coordinate, documented
        # simplification since the layers are actually sigma surfaces
        # (their true pressure varies with p* and hence with location/time
        # as the vortex evolves; see the dev log's phase-1 discussion of
        # this same simplification in the standalone prototype).
        self._layer_pressure = {'1': SOUNDING_P[0], '3': SOUNDING_P[1], 'b': SOUNDING_P[2]}

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        mstr = self.get_mstr(kwargs['member'])
        tstr = self.get_tstr(kwargs['time'])
        return os.path.join(kwargs['path'], tstr + mstr + '.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var_from_file(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        rec = self.variables[kwargs['name']].asdict()
        comm = None  # reading files doesn't require collective io (file locks)
        if rec['is_vector']:
            u = nc_read_var(fname, rec['name'][0], comm=comm)[0, ...]
            v = nc_read_var(fname, rec['name'][1], comm=comm)[0, ...]
            var = np.array([u, v])
        else:
            var = nc_read_var(fname, rec['name'], comm=comm)[0, ...]
        return var

    def write_var_to_file(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        rec = self.variables[kwargs['name']].asdict()
        comm = self.c.comm  # for async file io (netcdf without parallel support)
        if rec['is_vector']:
            for i in range(2):
                nc_write_var(fname, {'t': None, 'y': self.ny, 'x': self.nx},
                              rec['name'][i], var[i, ...], recno={'t': 0}, comm=comm)
        else:
            nc_write_var(fname, {'t': None, 'y': self.ny, 'x': self.nx},
                          rec['name'], var, recno={'t': 0}, comm=comm)

    def z_coords(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        layer = kwargs['name'].rsplit('_', 1)[-1]
        p = self._layer_pressure.get(layer, 0.0)  # pstar itself has no layer pressure
        return np.full(self.grid.x.shape, p)

    def generate_initial_condition(self):
        bg_seed = self.bg_seed
        return initial_condition(self.nx, self.ny, self.dx, beta=self.beta, moist=self.moist,
                                  Vbg=self.Vbg, Vslope=self.Vslope, bg_seed=bg_seed)

    def _read_full_state(self, **kwargs) -> dict:
        """Read all layer/pstar fields at kwargs['time'] into one flat
        state dict, keyed by native variable name (see util.pack_state) --
        needed because the dynamical core integrates all fields together
        as one coupled system, unlike vort2d's single-variable state."""
        state = {}
        for var_name, rec in self.variables.items():
            val = self.read_var(**{**kwargs, 'name': var_name})
            if rec.is_vector:
                layer = var_name.rsplit('_', 1)[-1]
                state[f'u{layer}'] = val[0]
                state[f'v{layer}'] = val[1]
            else:
                state[rec.name] = val
        return state

    def _write_full_state(self, state: dict, **kwargs) -> None:
        for var_name, rec in self.variables.items():
            if rec.is_vector:
                layer = var_name.rsplit('_', 1)[-1]
                val = np.array([state[f'u{layer}'], state[f'v{layer}']])
            else:
                val = state[rec.name]
            self.write_var(val, **{**kwargs, 'name': var_name})

    def preprocess(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            self.c.fs.make_dir(kwargs['path'])
            file1 = self.filename(**{**kwargs, 'path': kwargs['restart_dir']})
            file2 = self.filename(**kwargs)
            self.c.run_job(f"cp -fL {file1} {file2}")
        elif self.io_mode == 'online':
            # save a copy of the current state (forecast) as prior
            var = self.read_var_from_memory(**kwargs)
            self.write_var_to_memory(var.copy(), **{**kwargs, 'tag': 'prior'})

    def postprocess(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        # if offline mode, the current files are just posterior states; do nothing
        if self.io_mode == 'online':
            # save a copy of the current state (analysis) as posterior
            var = self.read_var_from_memory(**kwargs)
            self.write_var_to_memory(var, **{**kwargs, 'tag': 'post'})

    def run(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        self.run_status = 'running'

        state = self._read_full_state(**kwargs)
        forecast_period = kwargs['forecast_period']
        next_time = kwargs['time'] + forecast_period * dt1h

        new_state = advance_time(state, self.nx, self.ny, self.dx, self.beta,
                                  self.moist, self.dt, forecast_period)
        if any(np.any(np.isnan(v)) for v in new_state.values()):
            raise RuntimeError(f"{self.__class__.__name__}: NaN detected in model run")

        self._write_full_state(new_state, **{**kwargs, 'time': next_time})
        self.run_status = 'complete'

    def generate_truth(self, *args, **kwargs) -> None:
        assert self.truth_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        debug = kwargs.get('debug', False)
        self.c.fs.make_dir(self.truth_dir)

        self.c.total_tasks = int((self.c.config.time_end - self.c.config.time_start)
                                  / (dt1h * self.c.config.cycle_period))

        t = self.c.config.time_start
        self.c.current_task = 0
        while t < self.c.config.time_end:
            opts = {**kwargs, 'path': self.truth_dir, 'time': t}

            if t == self.c.config.time_start:
                state = self.generate_initial_condition()
                if debug:
                    print(f"generating initial condition in {self.truth_dir}")
                self._write_full_state(state, **opts)

            next_t = t + kwargs['forecast_period'] * dt1h
            self.c.debug_message = f"running model, saving output at {next_t}"
            self.run(**{**kwargs, 'path': self.truth_dir, 'time': t})
            t = next_t
            self.c.current_task += 1

    def generate_init_ensemble(self, *args, **kwargs) -> None:
        assert self.ens_init_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        debug = kwargs.get('debug', False)
        self.c.fs.make_dir(self.ens_init_dir)

        member = kwargs.get('member')
        # per-member background-flow seed, if not explicitly fixed by config
        bg_seed = self.bg_seed if self.bg_seed is not None else member
        if debug:
            print(f"generating initial condition for member {member+1 if member is not None else 1}")

        state = initial_condition(self.nx, self.ny, self.dx, beta=self.beta, moist=self.moist,
                                   Vbg=self.Vbg, Vslope=self.Vslope, bg_seed=bg_seed)
        self._write_full_state(state, **{**kwargs, 'path': self.ens_init_dir})
