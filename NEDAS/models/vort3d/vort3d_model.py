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
        Vmax (float): initial vortex peak tangential wind speed, m/s
            (smith_vortex's own default: 15.0)
        Rmw (float): initial vortex radius of maximum wind, m
            (smith_vortex's own default: 120e3)
        vortex_x0, vortex_y0 (float): initial vortex center, m, relative to
            the domain center (Core's own default: displaced toward the
            southern boundary, vortex_y0=-700e3, to give the vortex room to
            drift over a long integration -- see vort3d.md dev log,
            2026-07-22). This is the truth/reference center; ensemble
            members perturb around it (see pos_sprd).
        pos_sprd (float): initial vortex position spread, m -- each
            ensemble member's vortex center is drawn from a Gaussian with
            std=pos_sprd around (vortex_x0, vortex_y0), matching vort2d's
            loc_sprd exactly (np.random.normal(0, loc_sprd), NOT a uniform
            draw), seeded per member the same way as the background flow
            (vort2d itself reseeds from system entropy every call instead,
            which vort3d deliberately does not do -- reproducibility across
            runs is needed for the model-comparison work this was built
            for). The truth run always uses the exact configured center
            (pos_sprd=0), matching vort2d's convention.
        u_bkg, v_bkg (float): uniform (spatially-constant) steering flow,
            m/s, added on top of Vbg's turbulent background flow -- see
            Core's own docstring for why this is a distinct mechanism from
            Vbg (persists/steers rather than getting sheared apart by the
            vortex).
        f0 (float): reference Coriolis parameter, /s, at beta=0/y=0 (default:
            20N, matching the paper's own fixed-latitude assumption).
        theta_sprd, q_sprd (float): ensemble spread (Gaussian std, K and
            kg/kg) for a domain-uniform boundary-layer-only theta/q
            perturbation per member -- same per-member-seeded Gaussian
            mechanism as pos_sprd, applied to Core's theta_offset/q_offset
            (see Core's own docstring for why only the boundary layer, and
            why no dynamical adjustment is needed).
        Vmax_sprd, Rmw_sprd (float): ensemble spread (Gaussian std, m/s and
            m) for the initial vortex's own peak wind / radius of maximum
            wind, drawn the same per-member-seeded way as pos_sprd; floored
            (Vmax >= 1 m/s, Rmw >= 10 km) to avoid a degenerate/negative
            vortex from an unlucky large negative draw.

        All spread parameters (pos_sprd, theta_sprd, q_sprd, Vmax_sprd,
        Rmw_sprd) draw from ONE shared per-member RNG stream (see
        generate_init_ensemble/_perturb_ic), not independently re-seeded
        streams -- re-seeding fresh for each quantity would make them draw
        the same underlying random sample (just rescaled), spuriously
        correlating position/thermodynamic/intensity perturbations across
        the ensemble instead of sampling them independently.
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
    Vmax: float
    Rmw: float
    vortex_x0: float
    vortex_y0: float
    pos_sprd: float
    u_bkg: float
    v_bkg: float
    f0: float
    theta_sprd: float
    q_sprd: float
    Vmax_sprd: float
    Rmw_sprd: float
    output_dt: float | None = None  # sub-cycle output interval, hours -- if set and shorter
    # than a given run() call's forecast_period, run() chunks the integration into output_dt-
    # sized segments and writes the full state to file after each one, giving intermediate
    # snapshots within a single DA cycle without changing cycle_period (which the scheme's own
    # checkpointing/next-cycle bookkeeping still depends on). None (default) reproduces the
    # original single-shot-per-cycle behavior exactly.
    dt_reduction_factor: float = 0.5  # adaptive-dt retry on NaN blowup, see util.advance_time
    max_dt_retries: int = 3
    min_dt: float | None = None  # None uses dt * dt_reduction_factor**max_dt_retries
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

    def _perturb_ic(self, bg_seed, pos_sprd, theta_sprd, q_sprd, Vmax_sprd, Rmw_sprd):
        """Draw per-member IC perturbations from ONE shared RNG stream (seeded per
        member, same as the background flow) -- sharing one stream, rather than
        re-seeding fresh for each quantity, avoids spuriously correlating the
        different perturbed quantities (see class docstring)."""
        vortex_x0, vortex_y0 = self.vortex_x0, self.vortex_y0
        theta_offset, q_offset = 0.0, 0.0
        Vmax, Rmw = self.Vmax, self.Rmw
        if any(s > 0 for s in (pos_sprd, theta_sprd, q_sprd, Vmax_sprd, Rmw_sprd)):
            rng = np.random.default_rng(bg_seed)
            if pos_sprd > 0:
                vortex_x0 = vortex_x0 + rng.normal(0, pos_sprd)
                vortex_y0 = vortex_y0 + rng.normal(0, pos_sprd)
            if theta_sprd > 0:
                theta_offset = rng.normal(0, theta_sprd)
            if q_sprd > 0:
                q_offset = rng.normal(0, q_sprd)
            if Vmax_sprd > 0:
                Vmax = max(Vmax + rng.normal(0, Vmax_sprd), 1.0)
            if Rmw_sprd > 0:
                Rmw = max(Rmw + rng.normal(0, Rmw_sprd), 10.0e3)
        return vortex_x0, vortex_y0, theta_offset, q_offset, Vmax, Rmw

    def generate_initial_condition(self, pos_sprd=None, theta_sprd=None, q_sprd=None,
                                    Vmax_sprd=None, Rmw_sprd=None):
        bg_seed = self.bg_seed
        if pos_sprd is None:
            pos_sprd = self.pos_sprd
        if theta_sprd is None:
            theta_sprd = self.theta_sprd
        if q_sprd is None:
            q_sprd = self.q_sprd
        if Vmax_sprd is None:
            Vmax_sprd = self.Vmax_sprd
        if Rmw_sprd is None:
            Rmw_sprd = self.Rmw_sprd
        vortex_x0, vortex_y0, theta_offset, q_offset, Vmax, Rmw = self._perturb_ic(
            bg_seed, pos_sprd, theta_sprd, q_sprd, Vmax_sprd, Rmw_sprd)
        return initial_condition(self.nx, self.ny, self.dx, nz=self.nz, beta=self.beta,
                                  moist=self.moist, convection_scheme=self.convection_scheme,
                                  Vbg=self.Vbg, Vslope=self.Vslope, bg_seed=bg_seed,
                                  Vmax=Vmax, Rmw=Rmw,
                                  vortex_x0=vortex_x0, vortex_y0=vortex_y0,
                                  u_bkg=self.u_bkg, v_bkg=self.v_bkg, f0=self.f0,
                                  theta_offset=theta_offset, q_offset=q_offset)

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

        # chunk into output_dt-sized segments if requested (see output_dt field docstring) --
        # each chunk is its own advance_time call (cold-restarted AB3 tendency history, same
        # documented simplification advance_time's own docstring already accepts for
        # forecast_period-sized calls); with output_dt unset or >= forecast_period this reduces
        # to exactly one chunk of length forecast_period, i.e. the original behavior.
        chunk_h = self.output_dt if (self.output_dt and self.output_dt < forecast_period) else forecast_period
        n_chunks = int(round(forecast_period / chunk_h))
        t = kwargs['time']
        for _ in range(n_chunks):
            state = advance_time(state, self.nx, self.ny, self.dx, self.nz, self.beta,
                                  self.moist, self.convection_scheme, self.dt, chunk_h,
                                  dt_reduction_factor=self.dt_reduction_factor,
                                  max_dt_retries=self.max_dt_retries, min_dt=self.min_dt)
            if any(np.any(np.isnan(v)) for v in state.values()):
                raise RuntimeError(f"{self.__class__.__name__}: NaN detected in model run")
            t = t + chunk_h * dt1h
            self._write_full_state(state, **{**kwargs, 'time': t})

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
                # truth uses the exact configured vortex/thermodynamic state, no
                # ensemble spread applied (all sprd params 0), same convention as
                # vort2d's loc_sprd
                state = self.generate_initial_condition(
                    pos_sprd=0, theta_sprd=0, q_sprd=0, Vmax_sprd=0, Rmw_sprd=0)
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

        # perturb the initial vortex position/intensity/size and boundary-layer
        # thermodynamics per member (mirrors vort2d's loc_sprd) -- previously the only
        # IC-spread source was the background flow's own per-member realization, with
        # the vortex itself always exactly the same across members (see vort3d.md dev
        # log, 2026-07-22)
        vortex_x0, vortex_y0, theta_offset, q_offset, Vmax, Rmw = self._perturb_ic(
            bg_seed, self.pos_sprd, self.theta_sprd, self.q_sprd, self.Vmax_sprd, self.Rmw_sprd)

        state = initial_condition(self.nx, self.ny, self.dx, nz=self.nz, beta=self.beta,
                                   moist=self.moist, convection_scheme=self.convection_scheme,
                                   Vbg=self.Vbg, Vslope=self.Vslope, bg_seed=bg_seed,
                                   Vmax=Vmax, Rmw=Rmw,
                                   vortex_x0=vortex_x0, vortex_y0=vortex_y0,
                                   u_bkg=self.u_bkg, v_bkg=self.v_bkg, f0=self.f0,
                                   theta_offset=theta_offset, q_offset=q_offset)
        self._write_full_state(state, **{**kwargs, 'path': self.ens_init_dir})
