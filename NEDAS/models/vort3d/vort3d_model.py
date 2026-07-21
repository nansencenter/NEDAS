import os
import numpy as np
from NEDAS.grid import RegularGrid
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.core import Model
from NEDAS.core.types import VarDesc
from .core import make_sigma_levels, PSTAR_FAR, p_top
from .util import layer_names, initial_condition, advance_time


class Vort3DModel(Model[RegularGrid]):
    """
    Zhu, Smith & Ulrich (2001) minimal 3D tropical cyclone model: sigma-
    coordinate primitive equations on an f/beta-plane, `nz` free-
    atmosphere layers (top-to-bottom) plus one boundary layer at the
    bottom, + surface fluxes, radiative cooling, explicit condensation,
    and a convective closure. See ~/Google_Drive/papers/
    2024.NEDAS.Introduction/vort3d/ for the standalone prototype and
    validation this was ported from (dev log: techNotes/models/vort3d.md).

    State is (nz+1) layers x 4 fields (u,v,theta,q) + one 2D field (p*,
    column mass) -- represented as 3 multi-level NEDAS variables ('wind',
    'theta', 'q', each with `levels` spanning all nz+1 layers, level index
    k=0..nz-1 = free-atmosphere layers top-to-bottom, k=nz = boundary
    layer) plus one single-level 'pstar', following the same
    VarDesc(levels=...) + per-level `read_var(k=...)` pattern NEDAS's qg
    model uses -- NOT the earlier (pre-2026-07-21) design of one separate
    NEDAS variable per layer ('wind_0', 'wind_1', ..., 'wind_b', ...),
    which was simpler to read/write but made state_def/variable-list
    length scale with nz (3*(nz+1)+1 entries), an increasingly bad
    tradeoff as nz grows. Unlike qg's own per-level file I/O (which
    reloads and rewrites the ENTIRE multi-level array on every
    single-level write, since it stores each level in a plain .npy file),
    this uses a second "unlimited" netCDF dimension (`z`, alongside `t`)
    so a single-level write only touches that level's slice -- verified
    directly that netCDF4/HDF5 supports multiple unlimited dimensions per
    variable and handles out-of-order/sparse writes correctly (unwritten
    slices read back as NaN, not garbage).

    Args:
        nx, ny (int): grid dimensions (paper: 200x200)
        dx (float): grid spacing, m (paper: 20000)
        nz (int): number of free-atmosphere layers (paper: 2 -- upper
            troposphere + lower/mid troposphere). Total prognostic layers
            = nz+1 (always +1 boundary layer at the bottom). For nz=2, the
            vertical sigma levels match the paper's own Fig. 1/Table A1
            exactly; other nz use equal-sigma-thickness free-tropospheric
            layers (see core.make_sigma_levels).
        dt (float): internal model integration time step, s (paper: 15)
        restart_dt (float): restart/output interval, hours
        beta (float): df/dy, Coriolis beta parameter, /m/s (0 = pure
            f-plane, matching the paper's own experiments; >0 enables
            beta-drift)
        moist (bool): if False, runs the dry dynamical core only (no
            surface fluxes, radiative cooling, condensation, or convection)
        convection_scheme (str): 'ooyama' (the paper's own closure, only
            valid for nz=2) or 'betts_miller' (a simplified Betts/
            Betts-Miller-style column relaxation, valid for any nz -- see
            core.py's module docstring and the dev log for why these are
            the two supported options)
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
    nz: int
    dt: float
    restart_dt: float
    beta: float
    moist: bool
    convection_scheme: str
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

        self.n = self.nz + 1  # total prognostic layers (free-atmosphere + boundary), matches Core's own convention
        self.layer_names = layer_names(self.nz)  # ['0','1',...,'nz-1','b'], index k -> layer name

        multi_levels = np.arange(self.n, dtype=float)
        single_level = np.array([0.])
        self.variables = {
            'wind': VarDesc(name=('u', 'v'), dtype='float', is_vector=True,
                            dt=self.restart_dt, levels=multi_levels, units='m/s', z_units='hPa'),
            'theta': VarDesc(name='theta', dtype='float', is_vector=False,
                             dt=self.restart_dt, levels=multi_levels, units='K', z_units='hPa'),
            'q': VarDesc(name='q', dtype='float', is_vector=False,
                        dt=self.restart_dt, levels=multi_levels, units='kg/kg', z_units='hPa'),
            'pstar': VarDesc(name='pstar', dtype='float', is_vector=False,
                             dt=self.restart_dt, levels=single_level, units='Pa', z_units='hPa'),
        }

        # nominal far-field pressure of each layer -- a static z-coordinate,
        # documented simplification since the layers are actually sigma
        # surfaces (their true pressure varies with p* and hence with
        # location/time as the vortex evolves; see the dev log's phase-1
        # discussion of this same simplification in the standalone prototype).
        sigma_mid, _ = make_sigma_levels(self.nz)
        layer_p = sigma_mid*PSTAR_FAR + p_top
        self._layer_pressure = {name: float(layer_p[k]) for k, name in enumerate(self.layer_names)}

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        mstr = self.get_mstr(kwargs['member'])
        tstr = self.get_tstr(kwargs['time'])
        return os.path.join(kwargs['path'], tstr + mstr + '.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def _has_levels(self, name):
        return len(self.variables[name].levels) > 1

    def read_var_from_file(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name].asdict()
        comm = None  # reading files doesn't require collective io (file locks)
        has_z = self._has_levels(name)
        iz = int(kwargs['k']) if has_z else None

        def _read_one(varname):
            arr = nc_read_var(fname, varname, comm=comm)
            return arr[0, iz, ...] if has_z else arr[0, ...]

        if rec['is_vector']:
            u = _read_one(rec['name'][0])
            v = _read_one(rec['name'][1])
            var = np.array([u, v])
        else:
            var = _read_one(rec['name'])
        return var

    def write_var_to_file(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name].asdict()
        comm = self.c.comm  # for async file io (netcdf without parallel support)
        has_z = self._has_levels(name)
        if has_z:
            dims = {'t': None, 'z': None, 'y': self.ny, 'x': self.nx}
            recno = {'t': 0, 'z': int(kwargs['k'])}
        else:
            dims = {'t': None, 'y': self.ny, 'x': self.nx}
            recno = {'t': 0}

        if rec['is_vector']:
            for i in range(2):
                nc_write_var(fname, dims, rec['name'][i], var[i, ...], recno=recno, comm=comm)
        else:
            nc_write_var(fname, dims, rec['name'], var, recno=recno, comm=comm)

    def z_coords(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if kwargs['name'] == 'pstar':
            p = 0.0  # pstar itself has no layer pressure
        else:
            p = self._layer_pressure[self.layer_names[int(kwargs['k'])]]
        return np.full(self.grid.x.shape, p)

    def generate_initial_condition(self):
        bg_seed = self.bg_seed
        return initial_condition(self.nx, self.ny, self.dx, nz=self.nz, beta=self.beta,
                                  moist=self.moist, convection_scheme=self.convection_scheme,
                                  Vbg=self.Vbg, Vslope=self.Vslope, bg_seed=bg_seed)

    def _read_full_state(self, **kwargs) -> dict:
        """Read all layer/pstar fields at kwargs['time'] into one flat
        state dict, keyed by native variable name (see util.pack_state) --
        needed because the dynamical core integrates all fields together
        as one coupled system, unlike vort2d's single-variable state."""
        state = {}
        for k, layer in enumerate(self.layer_names):
            u, v = self.read_var(**{**kwargs, 'name': 'wind', 'k': k})
            state[f'u{layer}'] = u
            state[f'v{layer}'] = v
            state[f'theta{layer}'] = self.read_var(**{**kwargs, 'name': 'theta', 'k': k})
            state[f'q{layer}'] = self.read_var(**{**kwargs, 'name': 'q', 'k': k})
        state['pstar'] = self.read_var(**{**kwargs, 'name': 'pstar'})
        return state

    def _write_full_state(self, state: dict, **kwargs) -> None:
        for k, layer in enumerate(self.layer_names):
            val = np.array([state[f'u{layer}'], state[f'v{layer}']])
            self.write_var(val, **{**kwargs, 'name': 'wind', 'k': k})
            self.write_var(state[f'theta{layer}'], **{**kwargs, 'name': 'theta', 'k': k})
            self.write_var(state[f'q{layer}'], **{**kwargs, 'name': 'q', 'k': k})
        self.write_var(state['pstar'], **{**kwargs, 'name': 'pstar'})

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

        new_state = advance_time(state, self.nx, self.ny, self.dx, self.nz, self.beta,
                                  self.moist, self.convection_scheme, self.dt, forecast_period)
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

        state = initial_condition(self.nx, self.ny, self.dx, nz=self.nz, beta=self.beta,
                                   moist=self.moist, convection_scheme=self.convection_scheme,
                                   Vbg=self.Vbg, Vslope=self.Vslope, bg_seed=bg_seed)
        self._write_full_state(state, **{**kwargs, 'path': self.ens_init_dir})
