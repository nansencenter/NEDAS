"""
NEDAS Model interface for the Python QG spectral model.

Mirrors QGFortranModel in structure; runs QGModel in-process (no external
executable).  Supports both offline (file-based .npy) and online (in-memory)
IO modes.

File format
-----------
Each snapshot is stored as a single NumPy .npy file containing the full
spectral streamfunction psi of shape (nz, nky, nkx), complex128.
Filename: <path>/<member_str>/output_<YYYYMMDD_HH>.npy

Array conventions
-----------------
Physical variables returned to / received from NEDAS have shape (ny, nx)
for scalars or (2, ny, nx) for vector (u, v), in NEDAS grid convention where
axis-0 = y direction and axis-1 = x direction.

The QG spectral model internally uses spec2grid() which returns (nx, ny) with
axis-0 = x direction, so _derive_var() and _psi_from_var() apply .T when
crossing the boundary between the two conventions.
"""

import os
from typing import Any
import numpy as np
from datetime import datetime

from NEDAS.utils.conversion import dt1h
from NEDAS.grid import Grid
from NEDAS.core import Model
from NEDAS.core.types import VarDesc

from .model import QGModel
from .spectral import setup_spectral_grid, spec2grid


# ---------------------------------------------------------------------------
# Utility: plain forward FFT (exact inverse of spec2grid)
# ---------------------------------------------------------------------------

def _grid2spec_simple(field_xy, g):
    """Forward FFT from physical to spectral space.

    field_xy : real (..., nx, ny)  with axis-(-2)=x, axis-(-1)=y
    Returns complex (..., nky, nkx).

    Exact inverse of spec2grid: _grid2spec_simple(spec2grid(wf, g), g) == wf
    on active (filter_mask > 0) modes.
    """
    sgn = g['sgn']       # (nx, ny)
    kxup = g['kxup']     # (nkx,)  x-wavenumber positions in full FFT array
    kyup = g['kyup']     # (nky,)  y-wavenumber positions
    nx, ny = g['nx'], g['ny']
    N2 = nx * ny
    F = np.fft.fft2(sgn * field_xy, axes=(-2, -1)) / N2
    wf_t = F[..., kxup[:, np.newaxis], kyup[np.newaxis, :]]  # (..., nkx, nky)
    return wf_t.swapaxes(-2, -1)                              # (..., nky, nkx)


# ---------------------------------------------------------------------------
# NEDAS Model interface
# ---------------------------------------------------------------------------

class QGPythonModel(Model):
    """NEDAS Model interface for the pure-Python QG spectral model.

    Configuration mirrors QGFortranModel and the QGModel dataclass.
    All physics parameters (kmax, nz, F, beta, bot_drag, …) are read from
    default.yml and overridden by the user YAML / CLI args via parse_config().
    """

    # Dynamic config attributes set by parse_config() in the base Model class
    kmax: int
    nz: int
    restart_dt: float
    F: float
    beta: float
    _g: dict[str, Any]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        n = 2 * (self.kmax + 1)
        self.ny, self.nx = n, n
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        self.grid = Grid(None, x, y, cyclic_dim='xy')
        self.grid.mask = np.full(self.grid.x.shape, False)

        # Cache the spectral grid (expensive to recompute each IO call)
        self._g = setup_spectral_grid(self.kmax)

        levels = np.arange(self.nz, dtype=float)
        self.variables = {
            'velocity':    VarDesc(name=('u', 'v'), dtype='float', is_vector=True,
                                   dt=self.restart_dt, levels=levels, units=1, z_units=1),
            'streamfunc':  VarDesc(name='psi',     dtype='float', is_vector=False,
                                   dt=self.restart_dt, levels=levels, units=1, z_units=1),
            'vorticity':   VarDesc(name='zeta',    dtype='float', is_vector=False,
                                   dt=self.restart_dt, levels=levels, units=1, z_units=1),
            'temperature': VarDesc(name='temp',    dtype='float', is_vector=False,
                                   dt=self.restart_dt, levels=levels, units=1, z_units=1),
        }

        assert self.nproc_per_run == 1, \
            f'{self.__class__.__name__} only supports serial runs'

        self.memory = {}

    # -------------------------------------------------------------------
    # File naming
    # -------------------------------------------------------------------

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        mstr = f'{kwargs["member"]+1:04d}' if kwargs['member'] is not None else ''
        assert kwargs['time'] is not None, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')
        return os.path.join(kwargs['path'], mstr, f'output_{tstr}.npy')

    # -------------------------------------------------------------------
    # Variable derivation: spectral psi ↔ NEDAS physical variables
    # -------------------------------------------------------------------

    def _derive_var(self, psi_layer, name):
        """Compute a NEDAS physical variable from one spectral layer (nky, nkx).

        Returns array in NEDAS grid convention: shape (ny, nx) for scalars,
        (2, ny, nx) for velocity.

        spec2grid output is (nx, ny) with axis-0 = x; .T gives (ny, nx).
        """
        g = self._g
        kx_ = g['kx_']
        ky_ = g['ky_']
        ksqd_ = g['ksqd_']
        wf = psi_layer[np.newaxis]   # (1, nky, nkx) for spec2grid batch dim

        if name == 'streamfunc':
            return spec2grid(wf, g)[0].T                       # (ny, nx)
        elif name == 'velocity':
            u = spec2grid(-1j * ky_ * wf, g)[0].T             # (ny, nx)
            v = spec2grid( 1j * kx_ * wf, g)[0].T             # (ny, nx)
            return np.array([u, v])                            # (2, ny, nx)
        elif name == 'vorticity':
            return spec2grid(-ksqd_ * wf, g)[0].T             # (ny, nx)
        elif name == 'temperature':
            return spec2grid(-np.sqrt(ksqd_) * wf, g)[0].T    # (ny, nx)
        else:
            raise ValueError(f'unknown variable: {name}')

    def _psi_from_var(self, var, name, iz, psi_all):
        """Convert a NEDAS physical variable to spectral and update psi_all[iz].

        var: (ny, nx) for scalars, (2, ny, nx) for velocity — NEDAS grid convention.
        .T converts to (nx, ny) = model convention expected by _grid2spec_simple.
        """
        g = self._g
        kx_ = g['kx_']
        ky_ = g['ky_']
        ksqd_ = g['ksqd_']
        kmax = g['kmax']

        if name == 'streamfunc':
            psi_all[iz] = _grid2spec_simple(var.T, g)          # var.T: (nx, ny)

        elif name == 'velocity':
            # u = var[0] shape (ny, nx), v = var[1]
            uk = _grid2spec_simple(var[0].T, g)
            vk = _grid2spec_simple(var[1].T, g)
            zetak = 1j * kx_ * vk - 1j * ky_ * uk
            psi_all[iz] = -zetak / ksqd_
            psi_all[iz, 0, kmax] = 0.0                         # zero DC component

        elif name == 'vorticity':
            zetak = _grid2spec_simple(var.T, g)
            psi_all[iz] = -zetak / ksqd_
            psi_all[iz, 0, kmax] = 0.0

        elif name == 'temperature':
            tempk = _grid2spec_simple(var.T, g)
            k_ = np.sqrt(ksqd_)
            psi_all[iz] = np.where(k_ > 0.0, -tempk / k_, 0.0)

        else:
            raise ValueError(f'unknown variable: {name}')

    def _var_at_level(self, psi_all, k, name):
        """Extract variable at (possibly fractional) vertical level k."""
        k1 = int(k)
        var1 = self._derive_var(psi_all[k1], name)
        if k1 < self.nz - 1 and k != k1:
            k2 = k1 + 1
            var2 = self._derive_var(psi_all[k2], name)
            return (var1 * (k2 - k) + var2 * (k - k1)) / (k2 - k1)
        return var1

    # -------------------------------------------------------------------
    # IO: offline (file-based .npy)
    # -------------------------------------------------------------------

    def read_var_from_file(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        psi_all = np.load(fname)                # (nz, nky, nkx)
        return self._var_at_level(psi_all, kwargs['k'], kwargs['name'])

    def write_var_to_file(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        k = kwargs['k']
        if k != int(k):
            return    # only write at integer levels

        iz = int(k)
        fname = self.filename(**kwargs)

        if os.path.exists(fname):
            psi_all = np.load(fname)
        else:
            nky, nkx = int(self._g['nky']), int(self._g['nkx'])
            psi_all = np.zeros((self.nz, nky, nkx), dtype=complex)

        self._psi_from_var(var, kwargs['name'], iz, psi_all)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.save(fname, psi_all)

    # -------------------------------------------------------------------
    # IO: online (in-memory)
    # Stores spectral psi under memory[tstr][key]['_psi_spec'] rather than
    # individual physical-space variables.
    # -------------------------------------------------------------------

    def _mem_key(self, kwargs):
        tag = kwargs.get('tag', 'forecast')
        mstr = self.get_mstr(kwargs.get('member', None))
        return tag + mstr

    def _get_psi_mem(self, tstr, key):
        return self.memory.get(tstr, {}).get(key, {}).get('_psi_spec', None)

    def _set_psi_mem(self, tstr, key, psi_all):
        if tstr not in self.memory:
            self.memory[tstr] = {}
        if key not in self.memory[tstr]:
            self.memory[tstr][key] = {}
        self.memory[tstr][key]['_psi_spec'] = psi_all

    def read_var_from_memory(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        tstr = self.get_tstr(kwargs['time'])
        key = self._mem_key(kwargs)
        psi_all = self._get_psi_mem(tstr, key)
        if psi_all is None:
            raise KeyError(
                f"{self.__class__.__name__}: no psi in memory['{tstr}']['{key}']")
        return self._var_at_level(psi_all, kwargs['k'], kwargs['name'])

    def write_var_to_memory(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        k = kwargs['k']
        if k != int(k):
            return
        iz = int(k)
        tstr = self.get_tstr(kwargs['time'])
        key = self._mem_key(kwargs)
        psi_all = self._get_psi_mem(tstr, key)
        if psi_all is None:
            nky, nkx = int(self._g['nky']), int(self._g['nkx'])
            psi_all = np.zeros((self.nz, nky, nkx), dtype=complex)
        self._psi_from_var(var, kwargs['name'], iz, psi_all)
        self._set_psi_mem(tstr, key, psi_all)

    # -------------------------------------------------------------------
    # Required abstract methods
    # -------------------------------------------------------------------

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def z_coords(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        return np.full(self.grid.x.shape, kwargs['k'])

    def preprocess(self, *args, **kwargs):
        if self.io_mode == 'online':
            return
        kwargs = super().parse_kwargs(kwargs)
        restart_dir = kwargs.get('restart_dir')
        if restart_dir is None:
            return
        restart_file = self.filename(**{**kwargs, 'path': restart_dir})
        input_file = self.filename(**kwargs)
        self.c.fs.make_dir(os.path.dirname(input_file))
        self.c.fs.copy_file(restart_file, input_file)

    def postprocess(self, *args, **kwargs):
        pass

    # -------------------------------------------------------------------
    # Model construction helpers
    # -------------------------------------------------------------------

    def _make_dz_rho(self):
        """Return (dz, rho) arrays from config or sensible defaults."""
        nz = self.nz
        dz_cfg = getattr(self, 'dz', None)
        rho_cfg = getattr(self, 'rho', None)

        if dz_cfg is not None:
            dz = np.asarray(dz_cfg, dtype=float)
        else:
            dz = np.ones(nz) / nz

        if rho_cfg is not None:
            rho = np.asarray(rho_cfg, dtype=float)
        else:
            # Small linear density increase for stratification
            drho_total = 0.03 * max(nz - 1, 1)
            rho = 1.0 + drho_total * np.arange(nz) / max(nz - 1, 1)

        return dz, rho

    def _make_ubar(self, dz):
        """Mean zonal velocity profile from config."""
        nz = self.nz
        ubar_type = getattr(self, 'ubar_type', 'uniform')
        uscale = getattr(self, 'uscale', 0.0)
        delu = getattr(self, 'delu', 0.0)

        if nz == 1 or ubar_type == 'uniform':
            return np.full(nz, uscale)
        elif ubar_type == 'linear':
            # Linearly decreasing from surface to bottom
            z = np.linspace(0.0, 1.0, nz)
            return uscale - delu * z
        else:
            return np.zeros(nz)

    def _build_qgmodel(self):
        """Construct a QGModel instance from this interface's config."""
        ga = getattr   # shorthand

        m = QGModel(
            kmax=self.kmax,
            nz=self.nz,
            F=self.F,
            Fe=ga(self, 'Fe', 0.0),
            beta=self.beta,
            uscale=ga(self, 'uscale', 0.0),
            vscale=ga(self, 'vscale', 0.0),
            strat_type=ga(self, 'strat_type', 'linear'),
            deltc=ga(self, 'deltc', 0.2),
            surface_bc=ga(self, 'surface_bc', 'rigid_lid'),
            dt=ga(self, 'dt', 0.0),
            dt_max=ga(self, 'dt_max', 0.0),
            adapt_dt=ga(self, 'adapt_dt', True),
            dt_tune=ga(self, 'dt_tune', 1.5),
            dt_step=ga(self, 'dt_step', 10),
            robert=ga(self, 'robert', 0.01),
            filter_type=ga(self, 'filter_type', 'hyperviscous'),
            filter_exp=ga(self, 'filter_exp', 8.0),
            k_cut=ga(self, 'k_cut', 0.0),
            dealiasing=ga(self, 'dealiasing', 'isotropic'),
            filt_tune=ga(self, 'filt_tune', 1.0),
            bot_drag=ga(self, 'bot_drag', 0.0),
            top_drag=ga(self, 'top_drag', 0.0),
            therm_drag=ga(self, 'therm_drag', 0.0),
            quad_drag=ga(self, 'quad_drag', 0.0),
            qd_angle=ga(self, 'qd_angle', 0.0),
            use_forcing=ga(self, 'use_forcing', False),
            norm_forcing=ga(self, 'norm_forcing', False),
            forc_coef=ga(self, 'forc_coef', 0.0),
            forc_corr=ga(self, 'forc_corr', 0.0),
            kf_min=ga(self, 'kf_min', 0.0),
            kf_max=ga(self, 'kf_max', 0.0),
            use_topo=ga(self, 'use_topo', False),
            linear=ga(self, 'linear', False),
            idum=ga(self, 'idum', -7),
        )
        return m

    def _make_init_psi(self, member=None, time=None):
        """Generate initial spectral psi from psi_init_type config."""
        g = self._g
        nky, nkx = int(g['nky']), int(g['nkx'])
        ksqd_ = g['ksqd_']
        filt = g['filter_mask']
        nz = self.nz

        seed = 0
        if member is not None:
            seed += (member + 1) * 7919
        if time is not None:
            seed ^= int(time.strftime('%Y%m%d%H')) % (2**31)
        rng = np.random.default_rng(abs(seed) % (2**31))

        e_o = getattr(self, 'e_o', 0.01)
        k_o = float(getattr(self, 'k_o', 3))
        delk = float(getattr(self, 'delk', 5.0))
        psi_init_type = getattr(self, 'psi_init_type', 'spectral_m')

        psi0 = np.zeros((nz, nky, nkx), dtype=complex)

        if psi_init_type in ('spectral_m', 'spectral_z', 'spectral'):
            kr = np.sqrt(ksqd_)
            ring = np.exp(-0.5 * ((kr - k_o) / max(delk, 0.5)) ** 2) * filt
            for iz in range(nz):
                noise = (rng.standard_normal((nky, nkx))
                         + 1j * rng.standard_normal((nky, nkx)))
                psi0[iz] = noise * ring
        else:
            # white-spectrum fallback
            for iz in range(nz):
                noise = (rng.standard_normal((nky, nkx))
                         + 1j * rng.standard_normal((nky, nkx)))
                psi0[iz] = noise * filt

        # Scale to target spectral kinetic energy sum(k²|ψ|²) = e_o
        energy = float(np.sum(ksqd_[np.newaxis] * np.abs(psi0) ** 2))
        if energy > 0.0:
            psi0 *= np.sqrt(e_o / energy)

        return psi0

    # -------------------------------------------------------------------
    # run() — in-process forecast
    # -------------------------------------------------------------------

    def run(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h
        member = kwargs['member']
        mstr = self.get_mstr(member)
        tag = kwargs.get('tag', 'forecast')

        self.run_status = 'running'

        # ---- Load or generate initial psi ----
        if self.io_mode == 'offline':
            input_file = self.filename(**kwargs)
            if os.path.exists(input_file):
                psi_init = np.load(input_file)
            else:
                psi_init = self._make_init_psi(member, time)
        else:
            tstr_in = self.get_tstr(time)
            key_in = tag + mstr
            psi_init = self._get_psi_mem(tstr_in, key_in)
            if psi_init is None:
                psi_init = self._make_init_psi(member, time)

        # ---- Build and initialise the physics model ----
        m = self._build_qgmodel()
        dz, rho = self._make_dz_rho()
        ubar = self._make_ubar(dz)
        m.initialize(psi_init=psi_init, dz=dz, rho=rho, ubar=ubar)

        # ---- Integrate for forecast_period hours ----
        tscale = getattr(self, 'tscale', 0.1)
        t_end = float(forecast_period) / 24.0 * tscale
        while m.time < t_end - 0.5 * m.dt:
            m.step()

        # ---- Save output ----
        psi_out = m.psi
        assert psi_out is not None
        tstr_out = self.get_tstr(next_time)
        if self.io_mode == 'offline':
            out_file = self.filename(**{**kwargs, 'time': next_time})
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.save(out_file, psi_out)
        else:
            out_key = tag + mstr
            self._set_psi_mem(tstr_out, out_key, psi_out.copy())

        self.run_status = 'done'

    # -------------------------------------------------------------------
    # Truth and ensemble generation
    # -------------------------------------------------------------------

    def generate_truth(self, *args, **kwargs) -> None:
        assert self.truth_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        kwargs['member'] = None
        self.c.fs.make_dir(self.truth_dir)

        spinup_hours = getattr(self, 'spinup_hours', 168)
        run_dir = os.path.join(self.truth_dir, 'run')

        # Spinup from psi_init_type IC to establish turbulent state
        init_time = self.c.config.time_start - spinup_hours * dt1h
        self.run(**{**kwargs, 'path': run_dir, 'member': 0,
                    'time': init_time, 'forecast_period': spinup_hours})

        # Cycling run through the experiment window
        current_time = self.c.config.time_start
        fp = kwargs['forecast_period']
        while current_time < self.c.config.time_end:
            self.run(**{**kwargs, 'path': run_dir, 'member': 0,
                        'time': current_time, 'forecast_period': fp})
            current_time += fp * dt1h

        # Move outputs to truth_dir and clean up
        self.c.fs.move_files_to_dir(
            os.path.join(run_dir, '0001', 'output*.npy'), self.truth_dir)
        self.c.debug_message = f'removing temporary run directory: {run_dir}'
        self.c.fs.remove_dir(run_dir)

    def generate_init_ensemble(self, *args, **kwargs) -> None:
        assert self.ens_init_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        spinup_hours = getattr(self, 'spinup_hours', 168)
        member = kwargs['member']
        assert member is not None, 'generate_init_ensemble requires a member index'
        mstr = f'{member+1:04d}'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')
        init_file = os.path.join(self.ens_init_dir, mstr, f'output_{tstr}.npy')

        if os.path.exists(init_file):
            return

        init_time = kwargs['time'] - spinup_hours * dt1h
        run_dir = os.path.join(self.ens_init_dir, 'spinup')
        self.run(**{**kwargs, 'path': run_dir,
                    'time': init_time, 'forecast_period': spinup_hours})

        src_file = os.path.join(run_dir, mstr, f'output_{tstr}.npy')
        self.c.fs.make_dir(os.path.dirname(init_file))
        self.c.fs.move_file(src_file, init_file)
