"""
QGModel: Python implementation of the multi-layer quasi-geostrophic spectral model.

Translates qg_driver.f90 / qg_run_tools.f90 / qg_init_tools.f90.

Physics
-------
PV equation (spectral):
  dq/dt = J(ψ, q) + β·∂ψ/∂x + (mean-flow terms) + (dissipation) + (forcing)

PV-streamfunction relation:
  q = -\\|k\\|²ψ + S·ψ   (S = tridiagonal stratification operator, \\|k\\| = wavenumber magnitude)

Time integration: leapfrog with Robert (Asselin) filter, adaptive timestep.

Array conventions
-----------------
Spectral: shape (nz, nky, nkx)  — nky = kmax+1, nkx = 2*kmax+1
Physical: shape (nz, ny, nx)    — nx = ny = 2*(kmax+1)
Wavenumber grids: shape (nky, nkx)

Usage
-----
::

    m = QGModel(kmax=63, nz=4, F=50.0, beta=1.5, ...)
    m.initialize()
    for _ in range(1000):
        m.step()
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

from .spectral import (
    setup_spectral_grid, make_filter,
    spec2grid_cc, grid2spec, spec2grid,
    ir_prod, ir_pwr, jacob,
)
from .strat import (
    get_psiq, get_vmodes, get_layer_depths, get_tripint,
    layer2mode, mode2layer,
)
from .numerics import march, ran, ran_reset, ring_integral


@dataclass
class QGModel:
    """Multi-layer QG spectral model (Python / NumPy / JAX).

    Parameters mirror the Fortran input namelist (qg_params.f90).
    """

    # Resolution
    kmax: int = 63
    nz:   int = 1

    # Fundamental scales
    beta: float = 1.5
    F:    float = 0.0      # f²L²/[(2π)²g'H₀]
    Fe:   float = 0.0      # free-surface parameter

    # Mean flow
    uscale: float = 0.0
    vscale: float = 0.0

    # Stratification
    strat_type:  str   = 'linear'
    deltc:       float = 0.2      # thermocline thickness (exp/stc profile)
    surface_bc:  str   = 'rigid_lid'

    # Time stepping
    dt:       float = 0.0
    dt_max:   float = 0.0   # hard ceiling on adapt_dt; 0 = uncapped
    adapt_dt: bool  = True
    dt_tune:  float = 1.5
    dt_step:  int   = 10
    robert:   float = 0.01

    # Filters / de-aliasing
    filter_type:  str   = 'hyperviscous'
    filter_exp:   float = 8.0
    k_cut:        float = 0.0
    dealiasing:   str   = 'isotropic'
    filt_tune:    float = 1.0

    # Dissipation
    bot_drag:   float = 0.0
    top_drag:   float = 0.0
    therm_drag: float = 0.0
    quad_drag:  float = 0.0
    qd_angle:   float = 0.0

    # Markovian forcing
    use_forcing: bool  = False
    norm_forcing: bool = False
    forc_coef:   float = 0.0
    forc_corr:   float = 0.0
    kf_min:      float = 0.0
    kf_max:      float = 0.0

    # Topography
    use_topo:  bool  = False

    # Tracer
    use_tracer: bool  = False

    # Control
    linear:   bool  = False
    idum:     int   = -7
    pi: float = field(default=np.pi, init=False, repr=False)

    # ---- Internal state (populated by initialize()) ----
    _g:      Optional[dict[str, Any]] = field(default=None, init=False, repr=False)
    _psiq:   Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _filt:   Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _ubar:   Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _vbar:   Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _qbarx:  Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _qbary:  Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _shearu: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _shearv: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _dz:     Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _rho:    Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _hb:     Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _force_o:    Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _toposhift:  Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # Prognostic fields
    q:      Optional[np.ndarray] = field(default=None, init=False, repr=False)
    q_o:    Optional[np.ndarray] = field(default=None, init=False, repr=False)
    rhs:    Optional[np.ndarray] = field(default=None, init=False, repr=False)
    psi:    Optional[np.ndarray] = field(default=None, init=False, repr=False)
    psi_o:  Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # Counters
    cntr:   int   = field(default=0, init=False)
    time:   float = field(default=0.0, init=False)
    _call_q: int  = field(default=0, init=False)

    # ---------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------

    def initialize(self, psi_init=None, dz=None, rho=None,
                   ubar=None, vbar=None, hb=None):
        """Set up spectral grid, stratification, and initial fields.

        Parameters
        ----------
        psi_init : ndarray (nz, nky, nkx) complex, optional
            Initial streamfunction in spectral space.
        dz  : (nz,) layer thicknesses (should sum to 1)
        rho : (nz,) layer densities
        ubar, vbar : (nz,) mean zonal / meridional velocity profiles
        hb  : (nky, nkx) complex bottom topography in spectral space
        """
        g = setup_spectral_grid(self.kmax)
        self._g = g
        nz  = self.nz
        nky = int(g['nky'])
        nkx = int(g['nkx'])

        # Default vertical grid
        if dz is None:
            dz = np.ones(nz) / nz
        if rho is None:
            rho = np.ones(nz)
        self._dz  = np.asarray(dz,  dtype=np.float64)
        self._rho = np.asarray(rho, dtype=np.float64)

        # Stratification operator
        drho = np.diff(rho)
        if nz > 1 and len(drho) > 0:
            self._psiq = get_psiq(self._dz, drho, self.F, self.Fe, self.surface_bc)
        else:
            self._psiq = None

        # Mean flow
        self._ubar  = np.asarray(ubar,  dtype=np.float64) if ubar is not None else np.zeros(nz)
        self._vbar  = np.asarray(vbar,  dtype=np.float64) if vbar is not None else np.zeros(nz)
        self._shearu = np.zeros(nz)
        self._shearv = np.zeros(nz)
        if nz > 1 and self._psiq is not None:
            # shearu = tri2mat(psiq) * ubar  (Fortran qg_driver line 61)
            psiq_mat = _build_trimat(self._psiq, nz)
            self._shearu = psiq_mat @ self._ubar
            self._shearv = psiq_mat @ self._vbar

        # PV gradient from mean flow + beta
        self._qbarx = self._shearv[:nz]
        self._qbary = -self._shearu[:nz] + self.beta

        # Filter
        self._filt = make_filter(
            g,
            filter_type=self.filter_type,
            filter_exp=self.filter_exp,
            k_cut=self.k_cut if self.k_cut > 0 else None,
            dealiasing=self.dealiasing,
            filt_tune=self.filt_tune,
        )

        # Topography
        self._hb = hb if hb is not None else None
        if self._hb is not None and self._ubar is not None:
            kx_ = g['kx_']
            ky_ = g['ky_']
            self._toposhift = (
                -1j * self._ubar[nz-1] * (kx_ * self._hb)
                -1j * self._vbar[nz-1] * (ky_ * self._hb)
            )
        else:
            self._toposhift = None

        # Forcing state
        if self.use_forcing:
            self._force_o = np.zeros((nky, nkx), dtype=complex)
        else:
            self._force_o = None

        ran_reset(self.idum)

        # Initialise prognostic fields
        shape_spec = (nz, nky, nkx)
        psi_arr = (np.asarray(psi_init, dtype=complex) if psi_init is not None
                   else np.zeros(shape_spec, dtype=complex))
        self.psi   = psi_arr
        self.psi_o = psi_arr.copy()
        q_arr      = self.get_pv(psi_arr)
        self.q     = q_arr
        self.q_o   = q_arr.copy()
        self.rhs   = self._get_rhs()

        if self.adapt_dt and self.dt == 0.0:
            self._update_dt()

        self._call_q = 0

    # ---------------------------------------------------------------
    # PV operations
    # ---------------------------------------------------------------

    def get_pv(self, psi):
        """Compute PV q from streamfunction ψ (spectral).

        q = -\\|k\\|²ψ + S·ψ

        Translates Fortran Get_pv in qg_run_tools.f90.
        psi : (nz, nky, nkx)  or  (nv, nky, nkx) for surf_buoy
        Returns (nz, nky, nkx)
        """
        assert self._g is not None, 'call initialize() before get_pv()'
        g     = self._g
        nz    = self.nz
        ksqd_ = g['ksqd_']   # (nky, nkx)
        psiq  = self._psiq    # (nv, 3)

        q = np.zeros((nz,) + ksqd_.shape, dtype=complex)

        if nz > 1 and psiq is not None:
            # Vertical stretching term
            # top boundary index depends on surface_bc
            top = nz - 1 if self.surface_bc == 'periodic' else 0

            # q[0] = psiq[0,-1]*psi[top] + psiq[0,0]*psi[0] + psiq[0,1]*psi[1]
            q[0] = (psiq[0, 0] * psi[top]
                    + psiq[0, 1] * psi[0]
                    + psiq[0, 2] * psi[1])
            # interior
            for iz in range(1, nz - 1):
                q[iz] = (psiq[iz, 0] * psi[iz - 1]
                         + psiq[iz, 1] * psi[iz]
                         + psiq[iz, 2] * psi[iz + 1])
            # bottom
            q[nz-1] = (psiq[nz-1, 0] * psi[nz - 2]
                       + psiq[nz-1, 1] * psi[nz - 1]
                       + psiq[nz-1, 2] * psi[0])   # wrap for periodic; 0 for rigid_lid
        else:
            q = -self.F * psi

        q = -ksqd_[np.newaxis, :, :] * psi[:nz] + q
        return q

    def invert_pv(self):
        """Invert PV q → ψ by solving the tridiagonal system per wavenumber.

        Translates Fortran Invert_pv in qg_run_tools.f90.
        Returns psi of shape (nz, nky, nkx).
        """
        from .numerics import tridiag_vec, tridiag_cyc_vec

        assert self._g is not None, 'call initialize() before invert_pv()'
        assert self.q is not None

        g     = self._g
        nz    = self.nz
        ksqd_ = g['ksqd_']
        lin2kx = g['lin2kx']
        lin2ky = g['lin2ky']
        nmask  = g['nmask']
        psiq   = self._psiq

        psi = np.zeros((nz,) + ksqd_.shape, dtype=complex)

        if nz == 1:
            # Barotropic: psi = -q / (F + k²)
            denom = self.F + ksqd_
            psi[0] = np.where(ksqd_ != 0, -self.q[0] / denom, 0.0)
            return psi

        # Build per-wavenumber diagonal: psiq[:, 1] - k²
        # Shape: (nmask, nz)  — k² varies per wavenumber
        assert psiq is not None
        ksqd_flat = ksqd_[lin2ky, lin2kx]   # (nmask,)
        diag_base = psiq[:nz, 1]              # (nz,) shared part
        diag = diag_base[np.newaxis, :] - ksqd_flat[:, np.newaxis]   # (nmask, nz)

        sub = psiq[1:nz, 0]      # (nz-1,)
        sup = psiq[0:nz-1, 2]    # (nz-1,)

        # RHS: q at active wavenumbers, shape (nmask, nz)
        q_flat = self.q[:, lin2ky, lin2kx].T    # (nmask, nz)

        if self.surface_bc == 'periodic':
            corner_lo = psiq[0, 0]
            corner_hi = psiq[nz-1, 2]
            psi_flat = tridiag_cyc_vec(q_flat, sub, diag, sup, corner_lo, corner_hi)
        else:
            psi_flat = tridiag_vec(q_flat, sub, diag, sup)

        # Scatter back
        for iz in range(nz):
            psi[iz, lin2ky, lin2kx] = psi_flat[:, iz]

        return psi

    # ---------------------------------------------------------------
    # RHS (advection + dissipation + forcing)
    # ---------------------------------------------------------------

    def _get_rhs(self):
        """Compute the right-hand side of the PV equation.

        Translates Fortran Get_rhs subroutine in qg_driver.f90.
        Returns rhs of shape (nz, nky, nkx).
        """
        assert self._g is not None
        assert self._filt is not None
        assert self._dz is not None
        assert self.psi is not None and self.q is not None and self.psi_o is not None
        assert self._qbarx is not None and self._qbary is not None
        assert self._ubar is not None and self._vbar is not None

        g     = self._g
        nz    = self.nz
        kx_   = g['kx_']
        ky_   = g['ky_']
        ksqd_ = g['ksqd_']
        filt  = self._filt   # (nky, nkx)

        rhs = np.zeros((nz,) + kx_.shape, dtype=complex)

        if not self.linear:
            # Velocities and PV gradients in physical space (staggered-packed)
            ug  = spec2grid_cc(-1j * ky_[np.newaxis] * self.psi, g)
            vg  = spec2grid_cc( 1j * kx_[np.newaxis] * self.psi, g)

            q_work = self.q.copy()
            if self.use_topo and self._hb is not None:
                q_work[nz-1] += self._hb

            qxg = spec2grid_cc(1j * kx_[np.newaxis] * q_work, g)
            qyg = spec2grid_cc(1j * ky_[np.newaxis] * q_work, g)

            # Advection: -J(ψ, q) = -(u·∇q) = -(ug*dq/dx + vg*dq/dy)
            rhs = -grid2spec(ir_prod(ug, qxg) + ir_prod(vg, qyg), g)

            # Quadratic bottom drag
            if self.quad_drag != 0.0:
                speed = ir_pwr(ir_pwr(ug[nz-1], 2.0) + ir_pwr(vg[nz-1], 2.0), 0.5)
                qdrag = self.quad_drag * filt * (
                    1j * kx_ * grid2spec(speed * (
                        ug[nz-1] * np.sin(self.qd_angle)
                        + vg[nz-1] * np.cos(self.qd_angle)
                    ), g)
                    - 1j * ky_ * grid2spec(speed * (
                        ug[nz-1] * np.cos(self.qd_angle)
                        - vg[nz-1] * np.sin(self.qd_angle)
                    ), g)
                )
                rhs[nz-1] -= qdrag

        # Mean-flow / beta-plane linear terms
        # rhs -= ik_y*(-qbarx*ψ + vbar*q) + ik_x*(qbary*ψ + ubar*q)
        if np.any(self._qbarx != 0):
            rhs -= 1j * (ky_[np.newaxis] * (
                -self._qbarx[:, np.newaxis, np.newaxis] * self.psi
                + self._vbar[:, np.newaxis, np.newaxis] * self.q
            ))
        if np.any(self._qbary != 0):
            rhs -= 1j * (kx_[np.newaxis] * (
                 self._qbary[:, np.newaxis, np.newaxis] * self.psi
                + self._ubar[:, np.newaxis, np.newaxis] * self.q
            ))

        # Topographic phase shifting
        if self.use_topo and self._toposhift is not None:
            rhs[nz-1] += self._toposhift

        # Bottom / top Ekman drag.
        # Use current psi (not time-lagged psi_o) so the drag acts as a
        # semi-implicit term: ψ_{n+1} = ψ_{n-1}/(1+2*dt*drag) — unconditionally
        # stable, avoiding the leapfrog computational-mode growth that occurs
        # when drag*k²*dt >> 1 near the de-aliasing cutoff.
        if self.bot_drag != 0.0:
            rhs[nz-1] += self.bot_drag * ksqd_ * self.psi[nz-1]
        if self.top_drag != 0.0:
            rhs[0]    += self.top_drag * ksqd_ * self.psi[0]

        # Markovian stochastic forcing on top layer
        if self.use_forcing:
            rhs[0] += self._markovian(
                self.kf_min, self.kf_max, self.forc_coef, self.forc_corr,
                self.norm_forcing, self.psi[0]
            )

        # Thermal drag
        if self.therm_drag != 0.0 and self.F != 0.0:
            if nz == 1:
                rhs += self.therm_drag * self.F * self.psi_o
            elif self.Fe != 0.0:
                rhs -= self.therm_drag * (self.q + ksqd_[np.newaxis] * self.psi)
            else:
                rhs[0]  -= self.therm_drag * self.F * (self.psi_o[1] - self.psi_o[0]) / self._dz[0]
                rhs[1]  -= self.therm_drag * self.F * (self.psi_o[0] - self.psi_o[1]) / self._dz[1]

        rhs = filt[np.newaxis] * rhs
        return rhs

    # ---------------------------------------------------------------
    # Markovian forcing
    # ---------------------------------------------------------------

    def _markovian(self, kf_min, kf_max, amp, lam, normalize, field):
        """Random Markovian (red-noise) forcing in spectral space.

        Translates Fortran Markovian in qg_run_tools.f90.
        Returns forcing array (nky, nkx).
        """
        assert self._g is not None
        g     = self._g
        ksqd_ = g['ksqd_']
        nkx   = int(g['nkx'])
        nky   = int(g['nky'])

        assert self._force_o is not None

        mask = (ksqd_ > kf_min**2) & (ksqd_ <= kf_max**2)
        noise_phase = 2.0 * np.pi * ran(nkx, nky).T   # (nky, nkx)
        noise = amp * np.sqrt(1.0 - lam**2) * np.exp(1j * noise_phase)

        frc_o = self._force_o
        frc_o = np.where(mask, lam * frc_o + noise, frc_o)
        forc = frc_o.copy()

        if normalize:
            gamma = -2.0 * np.sum(np.conj(field) * forc) / amp
            if abs(gamma) > 1e-5:
                forc = forc / gamma

        self._force_o = frc_o
        return forc

    # ---------------------------------------------------------------
    # Adaptive timestep
    # ---------------------------------------------------------------

    def _update_dt(self):
        """Set dt based on CFL condition.

        Translates Fortran adapt_dt logic in qg_driver.f90:
          dt = dt_tune * 2π / (kmax * sqrt(max(zsq(psi), beta, 1)))
        where  zsq = 2 * sum(dz * k⁴ * |ψ|²)  (twice the enstrophy).
        """
        if self.psi is not None and self._g is not None and self._dz is not None:
            ksqd_ = self._g['ksqd_']
            zsq = 2.0 * float(
                np.sum(self._dz[:, np.newaxis, np.newaxis]
                       * ksqd_[np.newaxis] ** 2
                       * np.abs(self.psi) ** 2)
            )
        else:
            zsq = 0.0
        denom = np.sqrt(max(zsq, self.beta, 1.0))
        self.dt = self.dt_tune * 2.0 * self.pi / (self.kmax * denom)
        if self.dt_max > 0.0:
            self.dt = min(self.dt, self.dt_max)

    # ---------------------------------------------------------------
    # Time integration
    # ---------------------------------------------------------------

    def step(self):
        """Advance the model one time step.

        Translates the main time loop body in qg_driver.f90.
        """
        if self._g is None:
            raise RuntimeError('call initialize() before step()')
        assert self.q is not None and self.q_o is not None
        assert self.rhs is not None and self._filt is not None
        assert self.psi is not None

        g = self._g

        # Adaptive timestep update
        if self.adapt_dt and (self.cntr % self.dt_step == 0 or self.cntr == 0):
            self._update_dt()

        # March PV forward
        q_new, q_o_new, self._call_q = march(
            self.q, self.q_o, self.rhs, self.dt, self.robert, self._call_q
        )
        self.q   = self._filt[np.newaxis] * q_new
        self.q_o = q_o_new

        # Save psi for time-lagged drag, then invert
        self.psi_o = self.psi.copy()
        self.psi   = self.invert_pv()

        # New RHS
        self.rhs = self._get_rhs()

        self.time  += self.dt
        self.cntr  += 1

    def run(self, n_steps):
        """Run the model for n_steps time steps."""
        for _ in range(n_steps):
            self.step()

    # ---------------------------------------------------------------
    # Convenience: physical-space fields
    # ---------------------------------------------------------------

    def get_psi_grid(self):
        """Return streamfunction in physical space, shape (nz, ny, nx)."""
        assert self.psi is not None and self._g is not None
        return spec2grid(self.psi, self._g)

    def get_q_grid(self):
        """Return PV in physical space, shape (nz, ny, nx)."""
        assert self.q is not None and self._g is not None
        return spec2grid(self.q, self._g)

    def get_u_grid(self):
        """Return zonal velocity, shape (nz, ny, nx)."""
        assert self.psi is not None and self._g is not None
        return spec2grid(-1j * self._g['ky_'][np.newaxis] * self.psi, self._g)

    def get_v_grid(self):
        """Return meridional velocity, shape (nz, ny, nx)."""
        assert self.psi is not None and self._g is not None
        return spec2grid(1j * self._g['kx_'][np.newaxis] * self.psi, self._g)

    @property
    def nx(self):
        return self._g['nx'] if self._g else 2 * (self.kmax + 1)

    @property
    def ny(self):
        return self.nx

    @property
    def nkx(self):
        return self._g['nkx'] if self._g else 2 * self.kmax + 1

    @property
    def nky(self):
        return self._g['nky'] if self._g else self.kmax + 1


# ---------------------------------------------------------------------------
# Helper referenced in strat.py (avoids circular import)
# ---------------------------------------------------------------------------

def _build_trimat(psiq, nz):
    """Build full nz×nz matrix from tridiagonal psiq coefficients."""
    mat = np.diag(psiq[:nz, 1])
    if nz > 1:
        mat += np.diag(psiq[1:nz, 0], -1)
        mat += np.diag(psiq[0:nz-1, 2], 1)
    return mat


