"""
Zhu, Smith & Ulrich (2001) minimal 3D tropical cyclone model -- dynamical
core + moisture/surface-flux/radiation physics + a convective closure.

Sigma-coordinate primitive equations on an f/beta-plane, `nz` free-
atmosphere layers (top-to-bottom) plus one boundary layer fixed at the
bottom (nlayers = nz+1 total), interfaces carrying sigma-dot. Finite
differences (periodic in x, rigid wall in y), 3rd-order Adams-Bashforth
time stepping.

Vertical structure generalized from the paper's own fixed 3-layer design
(nz=2: upper troposphere + lower/mid troposphere, plus the boundary layer)
to an arbitrary nz, following the pattern in NEDAS's qg model
(models/qg/python/strat.py): the layer/interface layout is built from
arrays (sigma_mid, sigma_int, d_sigma), and the hydrostatic cascade and
sigma-dot vertical advection are written as loops over those arrays rather
than named per-layer constants. For nz=2, the generated sigma values are
forced to match the paper's own Fig. 1/Table A1 numbers exactly, so this
is a strict generalization, not an approximation, of the original 3-layer
implementation (re-derived and checked to reduce exactly to the original
closed-form 3-layer expressions -- see dev log).

Two convective closures are available (`convection_scheme`):
  'ooyama'       -- the paper's own modified Ooyama (1969) closure (eqs.
                    16-37). Conceptually defined around exactly "boundary
                    layer + 2 free-atmosphere layers" (cloud-base flux
                    from the boundary layer, entrainment from "the middle
                    layer", detrainment into "the upper layer") -- not a
                    generic multi-level scheme. Only valid for nz=2.
  'betts_miller'  -- a simplified Betts (1986)/Betts-Miller-style column
                    relaxation, formulated purely in terms of a per-layer
                    reference profile (no named layers), so it works for
                    any nz. Precedented for exactly this purpose: Baik,
                    DeMaria & Raman (1990a) ran a 15-level axisymmetric TC
                    model with the Betts scheme (their Table 2), and
                    ZSU2001 repeatedly compares its own results against
                    theirs (though ZSU2001's own abstract says "11 levels"
                    for that model -- a citation discrepancy against the
                    primary source, not something to match here). See
                    betts_miller_adjustment()'s docstring for the specific
                    (simplified) formulation used here, and the dev log for
                    why this is not literally the same model as ZSU2001
                    once nz != 2, nor the same model as Baik et al. (their
                    model is axisymmetric radius-height, not this module's
                    Cartesian x-y channel).

A quick (non-faithful) comparison at nz=14 (15 total layers) against Baik
et al.'s reported control-simulation numbers landed in a broadly similar
wind-speed range (fluctuating ~50-72 m/s vs. their reported 58 m/s at
maturity) but shallower minimum pressure (~955-980 hPa vs. their 923 hPa)
and a qualitatively different life cycle (rapid intensification by ~25-50h
here vs. their slow 0-48h/rapid 96-144h/steady 144-192h three-stage
progression) -- expected given the very different vortex/sounding/domain/
geometry, not a discrepancy to chase. See dev log for the full comparison
and radius-height cross sections.

See techNotes/models/vort3d.md dev log for the paper's equations/parameters
and running notes on simplifications made here.
"""
import numpy as np

# ---------------------------------------------------------------------------
# constants (paper's values / standard atmospheric constants)
# ---------------------------------------------------------------------------
R = 287.04       # gas constant for dry air, J/kg/K
cp = 1004.0      # specific heat at constant pressure, J/kg/K
kappa = R / cp
g = 9.81
p0 = 1000.0e2     # Pa, reference pressure for Exner function
p_top = 100.0e2   # Pa, constant model-top pressure
Lv = 2.501e6      # latent heat of vaporization, J/kg
SST = 301.15      # 28 C, paper's value
tau_R = 12*3600.  # Newtonian radiative cooling timescale, s
tau_BM = 2*3600.  # Betts-Miller relaxation timescale, s (standard-literature O(1-3h) range)

# Appendix A, Table A1: p (mb), T (K), q (g/kg) at ALL FIVE of the paper's
# sigma positions (both layer midpoints and interfaces) -- used below to
# interpolate a sounding onto an arbitrary nz's layer positions, since the
# paper gives enough points to do this without inventing new numbers.
SOUNDING_SIGMA = np.array([1/6, 1/3, 11/18, 8/9, 17/18])
SOUNDING_T_TABLE = np.array([230.1, 256.1, 278.9, 293.8, 297.0])       # K
SOUNDING_Q_TABLE = np.array([0.01, 0.15, 4.82, 13.6, 13.9]) * 1e-3      # kg/kg
SOUNDING_P_FAR = 1015.0e2  # nominal far-field surface pressure, Pa (ps_far)
PSTAR_FAR = SOUNDING_P_FAR - p_top  # far-field p* = ps_far - p_top; sigma->pressure
                                    # conversions must use THIS, not ps_far itself
                                    # (p = sigma*p_star + p_top, eq. 1) -- an earlier
                                    # version used sigma*ps_far+p_top here, a real bug
                                    # (biased the sounding by using the wrong reference
                                    # pressure at every sigma level except sigma=1).

# kept for backward-compat / quick reference to the paper's nz=2 layer
# midpoint sounding values (Table A1 rows for layer1, layer3, layerb)
SOUNDING_P = np.array([252.5, 659.2, 964.2]) * 100.0
SOUNDING_Q = np.array([0.01, 4.82, 13.9]) * 1e-3
SOUNDING_T = np.array([230.1, 278.9, 297.0])


def theta_from_T_p(T, p):
    """Potential temperature from temperature T (K) and pressure p (Pa)."""
    return T * (p0 / p) ** kappa


def sounding_T(sigma):
    """Far-field environmental temperature (K) at a given sigma, linearly
    interpolated from the paper's Appendix A Table A1 (all 5 tabulated
    sigma positions, not just the nz=2 layer midpoints)."""
    return np.interp(sigma, SOUNDING_SIGMA, SOUNDING_T_TABLE)


def sounding_q(sigma):
    """Far-field environmental specific humidity (kg/kg) at a given sigma,
    linearly interpolated from the paper's Appendix A Table A1."""
    return np.interp(sigma, SOUNDING_SIGMA, SOUNDING_Q_TABLE)


def make_sigma_levels(nz, sigma_boundary_top=8/9):
    """
    Vertical layout: nz free-atmosphere layers (top-to-bottom) + 1 boundary
    layer occupying sigma in [sigma_boundary_top, 1] (paper's own boundary-
    layer depth, 1/9 in sigma, kept fixed regardless of nz -- the paper
    gives no rule for varying it, and it's a physically distinct layer, not
    something that should get thinner/thicker just because nz changes).

    For nz=2, returns the paper's own exact Fig. 1 values (NOT re-derived
    from a general rule -- the paper's actual interfaces aren't evenly
    spaced in sigma, and there's no way to recover that specific asymmetric
    choice from a generic formula). For other nz, the free troposphere
    [0, sigma_boundary_top] is divided into nz EQUAL-sigma-thickness
    layers -- a reasonable default the paper doesn't specify, since it only
    ever used nz=2.

    Returns:
        sigma_mid: (nz+1,) layer midpoint sigmas, index 0..nz-1 = free
            atmosphere top-to-bottom, index nz = boundary layer.
        sigma_int: (nz,) interface sigmas: [0..nz-2] between adjacent
            free-atmosphere layers, [nz-1] = sigma_boundary_top (the
            boundary-layer-top interface).
    """
    if nz == 2 and abs(sigma_boundary_top - 8/9) < 1e-12:
        sigma_mid = np.array([1/6, 11/18, 17/18])
        sigma_int = np.array([1/3, 8/9])
        return sigma_mid, sigma_int

    edges = np.linspace(0.0, sigma_boundary_top, nz + 1)
    sigma_mid_free = 0.5 * (edges[:-1] + edges[1:])
    sigma_int_free = edges[1:-1]
    sigma_mid_b = 0.5 * (sigma_boundary_top + 1.0)
    sigma_mid = np.concatenate([sigma_mid_free, [sigma_mid_b]])
    sigma_int = np.concatenate([sigma_int_free, [sigma_boundary_top]])
    return sigma_mid, sigma_int


def smith_vortex(r, vm=15.0, rm=120.0e3):
    """Smith et al. (1990) tangential wind profile, eq. (13)."""
    a = 1.78803
    b = 4.74736e-3
    c = 0.339806
    d = 5.37727e-4
    rp = r / rm
    return a * vm * rp * (1 + b * rp**4) / (1 + c * rp**2 + d * rp**6) ** 2


def make_grid(nx, ny, dx):
    """Cartesian (xx, yy) coordinate arrays, shape (ny, nx), centered on
    the domain (origin at the middle grid point) -- where the initial
    vortex is always placed."""
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    xx, yy = np.meshgrid(x, y, indexing='xy')
    return xx, yy


def gradient_wind_balance_pstar(r, v_tan, f0, rho0):
    """
    Simplified initialization: integrate gradient-wind balance
    dp/dr = rho*(f*v + v^2/r) radially outward from the vortex center to get
    the pressure perturbation associated with the initial axisymmetric
    vortex, instead of solving the paper's full nonlinear balance equation
    (their eqs. 14-15, an elliptic PDE via Kurihara & Bender 1980).
    Documented deviation -- see vort3d.md dev log.
    """
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    integrand = rho0 * (f0 * v_tan + v_tan**2 / np.maximum(r, 1e-6))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dr)])
    pstar_pert = cum - cum[-1]
    return pstar_pert


def qsat(T, p):
    """Saturation specific humidity via Bolton (1980). T in K, p in Pa; returns kg/kg."""
    T_c = T - 273.15
    es = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))  # Pa
    return 0.622 * es / np.maximum(p - 0.378*es, 1.0)


def surface_drag_coef(Vb):
    """Shapiro (1992) neutral drag coefficient, with a velocity cap (documented
    deviation, see dev log) since the linear formula has no physical ceiling."""
    R_F = 0.8
    V_cd_cap = 33.0
    return (1.024 + 0.05366*R_F*np.minimum(Vb, V_cd_cap)) * 1e-3


def random_pressure_field(nx, ny, dx, power_law, seed=None):
    """Unit-std random pressure perturbation field with a prescribed spectral
    slope -- paired with a geostrophic-wind derivation (Core.__init__) so the
    background flow starts in mass/geostrophic balance, unlike generating
    wind directly (the previous random_flow(), which left pstar with no
    matching perturbation at all -- see vort3d.md dev log, 2026-07-21)."""
    rng = np.random.default_rng(seed)
    noise_hat = np.fft.fft2(rng.standard_normal((ny, nx)))
    ki = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    kj = np.fft.fftfreq(ny, d=dx) * 2*np.pi
    KI, KJ = np.meshgrid(ki, kj)
    K = np.sqrt(KI**2 + KJ**2)
    K[0, 0] = 1.0
    p_hat = noise_hat * K**((power_law - 2) / 2.0)
    p_hat[0, 0] = 0.0
    p = np.real(np.fft.ifft2(p_hat))
    return (p - p.mean()) / p.std()


class Core:
    """
    Standalone (NEDAS-independent) integrator for the vort3d dynamical
    core: sigma-coordinate primitive equations on an f/beta-plane, `nz`
    free-atmosphere layers + 1 boundary layer, 3rd-order Adams-Bashforth
    time stepping. See the module docstring for the vertical-structure
    generalization and the two convective closures.

    Args:
        nx, ny (int): horizontal grid dimensions (paper: 200x200).
        dx (float): horizontal grid spacing, m (paper: 20000).
        nz (int): number of free-atmosphere layers (paper: 2). Total
            prognostic layers = nz+1 (always +1 boundary layer at the
            bottom). nz=2 uses the paper's own exact sigma levels; other
            nz use equal-sigma-thickness free-tropospheric layers (see
            make_sigma_levels).
        Vbg (float): random background-flow wind speed amplitude, m/s
            (0 = calm, matching the paper's own experiments).
        Vslope (float): background-flow kinetic-energy spectrum power law.
        bg_seed (int or None): RNG seed for the background flow.
        beta (float): df/dy, Coriolis beta parameter, /m/s (0 = pure
            f-plane, matching the paper's own experiments; >0 enables
            beta-drift).
        moist (bool): if False, run the dry dynamical core only (no
            surface fluxes, radiative cooling, condensation, or
            convection) -- valid for any nz.
        convection_scheme (str): 'ooyama' (the paper's own closure, only
            valid for nz=2) or 'betts_miller' (valid for any nz).
        sigma_boundary_top (float): sigma at the top of the boundary layer
            (paper's own value: 8/9). Only affects nz != 2 (the nz=2
            sigma levels are always the paper's exact Fig. 1 values,
            regardless of this argument).
        Vmax (float): initial vortex peak tangential wind speed, m/s
            (smith_vortex's own default: 15.0).
        Rmw (float): initial vortex radius of maximum wind, m
            (smith_vortex's own default: 120e3).
        vortex_x0, vortex_y0 (float): initial vortex center, m, relative to
            the domain center (0,0) -- default places it away from the
            domain center and toward the southern (negative-y) boundary
            (vortex_y0=-700e3), giving the vortex room to drift poleward/
            zonally over a long integration (observed beta-drift + Vbg
            advection can otherwise run it into a wall if it starts
            centered -- see vort3d.md dev log, 2026-07-22). Configurable
            (rather than hardcoded to the domain center) so
            generate_init_ensemble can perturb it per member, mirroring
            vort2d's loc_sprd.
        f0 (float): reference Coriolis parameter, /s, at beta=0 / y=0 (the
            domain-center latitude) -- default 2*7.292e-5*sin(20 deg),
            matching the paper's own fixed 20N assumption; beta then adds
            the y-dependence on top of this reference value.
        theta_offset, q_offset (float): domain-uniform (no horizontal
            gradient) perturbation added ONLY to the boundary layer's
            theta/q (K, kg/kg) -- all free-atmosphere layers, including the
            top level, are left exactly at the reference sounding. Meant as
            a simple ensemble IC-spread mechanism (domain-averaged
            boundary-layer thermodynamic uncertainty), analogous to
            vortex_x0/vortex_y0's position spread -- see
            Vort3DModel.generate_init_ensemble for where these get drawn
            per member. No accompanying dynamical adjustment needed since a
            spatially-uniform perturbation has zero horizontal gradient
            (same reasoning as u_bkg/v_bkg needing no matching pressure
            term).
        u_bkg, v_bkg (float): uniform (spatially-constant) steering flow,
            m/s, added on top of Vbg's turbulent background flow -- the
            standard simple "steering flow" scheme from the beta-drift/
            beta-and-advection TC-motion literature (a constant vector
            wind, distinct from Vbg's random field). Unlike Vbg, a uniform
            flow has zero gradients, so it's invariant under the model's
            (gradient-based) diffusion and doesn't get sheared apart by the
            vortex -- it actually persists and steers, which a domain-scale
            random field on a vortex-dominated small domain was observed
            not to (see vort3d.md dev log, 2026-07-22). No matching
            pressure perturbation is added: exactly balanced on an f-plane
            (both sides of geostrophic balance are zero for a spatially
            uniform field); for beta!=0 there's a small residual imbalance
            from f varying with y, neglected here (documented
            simplification, negligible next to the model's other
            approximations).

    Attributes set after construction: `u`, `v`, `theta`, `q` (each shape
    `(nz+1, ny, nx)`), `pstar` (shape `(ny, nx)`, column mass p*=ps-p_top).
    Advance the state in time with `step(dt)`.
    """

    def __init__(self, nx=100, ny=100, dx=20e3, nz=2, Vbg=0.0, Vslope=-3, bg_seed=None,
                 beta=0.0, moist=True, convection_scheme='ooyama', sigma_boundary_top=8/9,
                 Vmax=15.0, Rmw=120.0e3, vortex_x0=0.0, vortex_y0=-700.0e3,
                 u_bkg=0.0, v_bkg=0.0, f0=2*7.292e-5*np.sin(np.deg2rad(20.)),
                 theta_offset=0.0, q_offset=0.0):
        if moist and convection_scheme == 'ooyama' and nz != 2:
            raise ValueError(
                "convection_scheme='ooyama' is only supported for nz=2 -- eqs. "
                "16-37 are conceptually defined for exactly 'boundary layer + 2 "
                "free-atmosphere layers' (cloud-base flux from the boundary layer, "
                "entrainment from the middle layer, detrainment into the upper "
                "layer), not a generic multi-level scheme. Use convection_scheme="
                "'betts_miller' for other nz, or moist=False."
            )
        if moist and convection_scheme not in ('ooyama', 'betts_miller'):
            raise ValueError(f"unknown convection_scheme '{convection_scheme}'")
        self.nx, self.ny, self.dx = nx, ny, dx
        self.nz = nz               # number of free-atmosphere layers
        self.n = nz + 1             # total layers (free atmosphere + boundary)
        self.moist = moist
        self.convection_scheme = convection_scheme
        self.xx, self.yy = make_grid(nx, ny, dx)
        self.vortex_x0, self.vortex_y0 = vortex_x0, vortex_y0
        self.rr = np.hypot(self.xx - vortex_x0, self.yy - vortex_y0)

        self.f = f0 + beta * self.yy

        self.sigma_mid, self.sigma_int = make_sigma_levels(nz, sigma_boundary_top)
        # half-level (interface) sigmas, including top (0) and surface (1):
        # length n+1, half_sigma[k] and half_sigma[k+1] bound layer k
        self.half_sigma = np.concatenate([[0.0], self.sigma_int, [1.0]])
        self.d_sigma = np.diff(self.half_sigma)  # (n,) layer thicknesses

        n = self.n
        self.u = np.zeros((n, ny, nx))
        self.v = np.zeros((n, ny, nx))
        self.theta = np.zeros((n, ny, nx))
        self.q = np.zeros((n, ny, nx))
        for k in range(n):
            p_k = self.sigma_mid[k]*PSTAR_FAR + p_top
            self.theta[k] = theta_from_T_p(sounding_T(self.sigma_mid[k]), p_k)
            self.q[k] = sounding_q(self.sigma_mid[k])

        # domain-uniform boundary-layer perturbation (ensemble IC spread) -- only the
        # boundary layer (last index), free-atmosphere layers (incl. the top) untouched
        if theta_offset != 0.0:
            self.theta[-1] = self.theta[-1] + theta_offset
        if q_offset != 0.0:
            self.q[-1] = np.maximum(self.q[-1] + q_offset, 0.0)

        self.pstar = np.full((ny, nx), SOUNDING_P_FAR - p_top)

        # shared reference Coriolis parameter / boundary-layer density, used below both for
        # the vortex's own gradient-wind pstar and the background flow's geostrophic pstar
        f0_center = f0
        p_b = self.sigma_mid[-1]*PSTAR_FAR + p_top
        T_b = sounding_T(self.sigma_mid[-1])
        rho0 = p_b / (R * T_b)

        # initial vortex: the paper's own vortex is barotropic (same tangential
        # wind at every layer), but a strictly height-uniform wind is inconsistent
        # with thermal wind balance for any vortex with a warm core aloft --
        # deviation from the paper: taper vtan with height, full Vmax at the
        # boundary layer decaying upward (sigma-proportional), a simple
        # documented simplification not paired with a matching temperature
        # perturbation (see vort3d.md dev log, 2026-07-21).
        vtan = smith_vortex(self.rr, Vmax, Rmw)
        theta_ang = np.arctan2(self.yy - vortex_y0, self.xx - vortex_x0)
        taper = self.sigma_mid / self.sigma_mid[-1]
        for k in range(n):
            self.u[k] = -taper[k] * vtan * np.sin(theta_ang)
            self.v[k] = taper[k] * vtan * np.cos(theta_ang)

        # radial pressure perturbation via simplified gradient-wind balance,
        # using the boundary layer's (last layer's) sounding density
        r1d = np.linspace(0, self.rr.max(), 2000)
        v1d = smith_vortex(r1d, Vmax, Rmw)
        pstar_pert_1d = gradient_wind_balance_pstar(r1d, v1d, f0_center, rho0)
        pstar_pert = np.interp(self.rr.ravel(), r1d, pstar_pert_1d).reshape(self.rr.shape)
        self.pstar += pstar_pert

        # warm-core temperature perturbation, thermal-wind-consistent with the
        # tapered tangential wind above -- without this, the prescribed wind
        # has no matching temperature anomaly to dynamically sustain it, and
        # the vortex spends its first 1-2 days spinning down while the model's
        # own WISHE feedback organically builds up the missing warm core (an
        # initial-adjustment transient documented in vort3d.md dev log,
        # 2026-07-22/23 -- confirmed via a Vmax/nz sensitivity sweep that this
        # is a general dynamical/thermodynamical-imbalance-at-init effect, not
        # something more vertical levels alone can fix).
        #
        # Same simplified radial-integral construction as pstar_pert above
        # (gradient_wind_balance_pstar with rho0=1 gives geopotential-per-
        # unit-mass instead of pressure), evaluated at each level's own
        # taper*vtan, then converted to a temperature anomaly level-by-level
        # via the hypsometric relation (thickness <-> mean layer temperature)
        # -- a standard, order-of-magnitude-consistent simplification, not an
        # exact inversion of the model's own discrete Arakawa-Suarez
        # hydrostatic scheme, in the same documented-simplification spirit as
        # gradient_wind_balance_pstar itself vs. the paper's full nonlinear
        # balance equation. The boundary layer (k=n-1, taper=1) is left
        # unperturbed -- its support already comes entirely from pstar_pert.
        phi_pert_1d = np.array([gradient_wind_balance_pstar(r1d, taper[k]*v1d, f0_center, 1.0)
                                 for k in range(n)])
        theta_pert_1d = np.zeros((n, len(r1d)))
        for k in range(n-2, -1, -1):
            p_ref_k = self.sigma_mid[k]*PSTAR_FAR + p_top
            p_ref_below = self.sigma_mid[k+1]*PSTAR_FAR + p_top
            dT_1d = (phi_pert_1d[k] - phi_pert_1d[k+1]) / (R * np.log(p_ref_below/p_ref_k))
            theta_pert_1d[k] = dT_1d * (p0/p_ref_k)**kappa
        for k in range(n-1):
            theta_pert = np.interp(self.rr.ravel(), r1d, theta_pert_1d[k]).reshape(self.rr.shape)
            self.theta[k] += theta_pert

        if Vbg > 0:
            # generate as a geostrophically-balanced (pressure-derived) field, not
            # independent wind, so the background flow starts in mass/geostrophic balance --
            # deriving u,v directly (the previous approach) left pstar with no matching
            # perturbation at all, exciting a persistent gravity-wave adjustment (a "pstar
            # sweep") once integration started (see vort3d.md dev log, 2026-07-21).
            pstar_pert_bg = random_pressure_field(nx, ny, dx, Vslope, seed=bg_seed)
            u_bg = -self.ddy_interior(pstar_pert_bg) / (f0_center * rho0)
            v_bg = self.ddx(pstar_pert_bg) / (f0_center * rho0)
            # the pressure amplitude needed to hit the target Vbg (rms wind amplitude) isn't
            # known a priori for a general power-law spectrum -- generate at unit pressure-std
            # then rescale wind and pressure together (a linear relation, so this preserves
            # geostrophic balance) to hit it.
            wind_rms = np.sqrt(np.mean(u_bg**2 + v_bg**2))
            scale = Vbg / wind_rms if wind_rms > 0 else 0.0
            u_bg, v_bg, pstar_pert_bg = u_bg*scale, v_bg*scale, pstar_pert_bg*scale
            for k in range(n):
                self.u[k] += u_bg
                self.v[k] += v_bg
            self.pstar += pstar_pert_bg

        if u_bkg != 0.0 or v_bkg != 0.0:
            # uniform steering flow, see class docstring -- no matching pressure
            # perturbation (trivially balanced for beta=0, see docstring)
            for k in range(n):
                self.u[k] += u_bkg
                self.v[k] += v_bkg

        # rigid-wall BC: v=0 at the y-edges, enforced unconditionally (previously only
        # applied inside the Vbg>0 branch, leaving it unenforced for Vbg=0 cases at t=0 even
        # though step() re-enforces it every subsequent step -- incidental fix, harmless
        # since it only zeroes what should already be zero there for a well-behaved case).
        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0

        self._prev_tendencies = []  # for Adams-Bashforth

    def ddx(self, f):
        return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * self.dx)

    def ddy_interior(self, f):
        d = np.zeros_like(f)
        d[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * self.dx)
        return d

    def _pad_y(self, f, n=2):
        top = np.repeat(f[0:1, :], n, axis=0)
        bot = np.repeat(f[-1:, :], n, axis=0)
        return np.vstack([top, f, bot])

    def upwind3_dx(self, f, u):
        fm2 = np.roll(f, 2, axis=-1)
        fm1 = np.roll(f, 1, axis=-1)
        fp1 = np.roll(f, -1, axis=-1)
        fp2 = np.roll(f, -2, axis=-1)
        d_pos = (fm2 - 6*fm1 + 3*f + 2*fp1) / (6*self.dx)
        d_neg = (-2*fm1 - 3*f + 6*fp1 - fp2) / (6*self.dx)
        return np.where(u >= 0, d_pos, d_neg)

    def upwind3_dy(self, f, v):
        fp = self._pad_y(f, 2)
        fm2 = fp[0:-4, :]
        fm1 = fp[1:-3, :]
        f0 = fp[2:-2, :]
        fp1 = fp[3:-1, :]
        fp2 = fp[4:, :]
        d_pos = (fm2 - 6*fm1 + 3*f0 + 2*fp1) / (6*self.dx)
        d_neg = (-2*fm1 - 3*f0 + 6*fp1 - fp2) / (6*self.dx)
        return np.where(v >= 0, d_pos, d_neg)

    def diffuse4(self, f, k1):
        def lap(a):
            lx = (np.roll(a, -1, axis=-1) - 2*a + np.roll(a, 1, axis=-1)) / self.dx**2
            ly = np.zeros_like(a)
            ly[..., 1:-1, :] = (a[..., 2:, :] - 2*a[..., 1:-1, :] + a[..., :-2, :]) / self.dx**2
            return lx + ly
        return -k1 * lap(lap(f))

    def hydrostatic(self, theta, pstar):
        """
        Geopotential at each layer midpoint, via the Arakawa & Suarez
        (1983) layer-mean-Exner-function scheme the paper adopts (Appendix
        B), generalized to n=nz+1 layers as a loop instead of literal
        3-layer algebra (verified to reduce exactly to the original
        3-layer formulas for nz=2 -- see dev log for the derivation this
        was checked against).

        For each layer k (0=top free-atm layer .. n-1=boundary layer),
        bounded by half-levels k (above) and k+1 (below):
          Phat[m]  = naive Exner at half-level m's own pressure
          P[k]     = layer-mean Exner (mass-weighted over the layer's own
                     bounding half-levels)
          theta_hat[m] = interface theta at internal half-level m (weighted
                     average of the two adjacent layers' theta, using P)
          Phi[n-1] = 0 + cp*theta[n-1]*(Phat[n]-P[n-1])   (lowest layer,
                     special form: no layer below the boundary layer)
          Phi[k]   = Phi[k+1] + cp*theta_hat[k+1]*(P[k+1]-P[k])  (k<n-1)
        """
        n = self.n
        half_sigma = self.half_sigma

        Phat = []
        phat_p = []
        for m in range(n + 1):
            p = half_sigma[m]*pstar + p_top
            Phat.append((p/p0)**kappa)
            phat_p.append(p)

        P = []
        for k in range(n):
            Ph_lo, ph_lo = Phat[k], phat_p[k]
            Ph_hi, ph_hi = Phat[k+1], phat_p[k+1]
            P.append((Ph_hi*ph_hi - Ph_lo*ph_lo) / ((1+kappa)*(ph_hi-ph_lo)))

        theta_hat = [None] * (n + 1)
        for m in range(1, n):
            Ph_mid = Phat[m]
            th_lo, P_lo = theta[m-1], P[m-1]
            th_hi, P_hi = theta[m], P[m]
            theta_hat[m] = ((Ph_mid-P_lo)*th_lo + (P_hi-Ph_mid)*th_hi) / (P_hi-P_lo)

        Phi = [None] * n
        Phi_s = np.zeros_like(pstar)
        Phi[n-1] = Phi_s + cp*theta[n-1]*(Phat[n] - P[n-1])
        for k in range(n-2, -1, -1):
            Phi[k] = Phi[k+1] + cp*theta_hat[k+1]*(P[k+1] - P[k])

        def Phi_at_halflevel(m):
            """Phi at an internal half-level m (1<=m<=n-1), extended
            self-consistently from the layer below using the same
            theta_hat (see original 3-layer derivation for why this
            partial-span evaluation is exact, not an extra assumption)."""
            return Phi[m] + cp*theta_hat[m]*(P[m] - Phat[m])

        return dict(Phi=np.array(Phi), theta_hat=theta_hat, Phat=Phat, P=P,
                    Phi_at_halflevel=Phi_at_halflevel)

    def rhs(self):
        """Right-hand-side tendencies (du, dv, dtheta, dq, dpstar) for the
        dynamical core (eqs. 2-8: momentum, hydrostatic PGF, sigma-dot
        vertical advection, surface fluxes/radiative cooling if `moist`) --
        does NOT include diffusion or the convective closure, both applied
        separately in `step()`. Returns the tendency tuple used by AB3."""
        u, v, theta, q, pstar = self.u, self.v, self.theta, self.q, self.pstar
        n = self.n
        sigma_mid = self.sigma_mid
        d_sigma = self.d_sigma
        half_sigma = self.half_sigma

        p = [sigma_mid[k]*pstar + p_top for k in range(n)]
        T = [theta[k] * (p[k]/p0)**kappa for k in range(n)]
        hyd = self.hydrostatic(theta, pstar)
        Phi = hyd['Phi']

        dpstar_dx = self.ddx(pstar)
        dpstar_dy = self.ddy_interior(pstar)

        D = []
        for k in range(n):
            div_uv = self.ddx(u[k]) + self.ddy_interior(v[k])
            D.append(pstar*div_uv + u[k]*dpstar_dx + v[k]*dpstar_dy)

        dpstar_dt = -sum(D[k]*d_sigma[k] for k in range(n))

        # sigma_dot at each half-level (0 and n are top/surface, always 0)
        pstar_sigmadot = [np.zeros_like(pstar)]
        cum = np.zeros_like(pstar)
        for m in range(1, n):
            cum = cum + D[m-1]*d_sigma[m-1]
            pstar_sigmadot.append(-cum - half_sigma[m]*dpstar_dt)
        pstar_sigmadot.append(np.zeros_like(pstar))
        sigmadot = [ps / pstar for ps in pstar_sigmadot]
        self._last_sigmadot = sigmadot  # used by convective_venting()/betts_miller_adjustment()

        def vert_adv(chi):
            """'-sigma_dot*dchi/dsigma' tendency (eqs. 2,3,7,8), Appendix B
            flux-form, generalized to n layers (verified to reduce to the
            original 3-layer closed forms for nz=2). ADDED to the
            horizontal-advection tendency below (not subtracted -- see the
            sign-bug fix documented in the dev log)."""
            chihat = [0.0] * (n + 1)
            for m in range(1, n):
                chihat[m] = 0.5*(chi[m-1] + chi[m])
            adv = []
            for k in range(n):
                term_below = sigmadot[k+1]*(chihat[k+1] - chi[k])
                term_above = sigmadot[k]*(chi[k] - chihat[k])
                adv.append(-(term_below + term_above) / d_sigma[k])
            return np.array(adv)

        vert_adv_u = vert_adv(u)
        vert_adv_v = vert_adv(v)
        vert_adv_theta = vert_adv(theta)
        vert_adv_q = vert_adv(q)

        du = np.zeros_like(u)
        dv = np.zeros_like(v)
        dtheta = np.zeros_like(theta)
        dq = np.zeros_like(q)

        if self.moist:
            Vb = np.hypot(u[n-1], v[n-1])
            Cd = surface_drag_coef(Vb)
            Ch = Cd
            rho_b = p[n-1] / (R*T[n-1])
            qs_sst = qsat(np.full_like(pstar, SST), p[n-1])
            Fu = -rho_b*Cd*Vb*u[n-1]
            Fv = -rho_b*Cd*Vb*v[n-1]
            Fq = rho_b*Ch*Vb*(qs_sst - q[n-1])
            F_SH = rho_b*cp*Ch*Vb*(SST - T[n-1])
            mass_b = pstar*d_sigma[n-1]/g

            # cloud-shielding check for radiative cooling: last free-atm
            # layer's RH (generalizes the paper's "layer 3" check to n-2)
            qs_rad = qsat(T[n-2], p[n-2])
            rh_rad = q[n-2] / np.maximum(qs_rad, 1e-8)
            rad_active = (rh_rad <= 0.9).astype(float)

        for k in range(n):
            dudx, dudy = self.upwind3_dx(u[k], u[k]), self.upwind3_dy(u[k], v[k])
            dvdx, dvdy = self.upwind3_dx(v[k], u[k]), self.upwind3_dy(v[k], v[k])
            dthdx, dthdy = self.upwind3_dx(theta[k], u[k]), self.upwind3_dy(theta[k], v[k])
            dqdx, dqdy = self.upwind3_dx(q[k], u[k]), self.upwind3_dy(q[k], v[k])
            dPhidx, dPhidy = self.ddx(Phi[k]), self.ddy_interior(Phi[k])

            adv_u = -(u[k]*dudx + v[k]*dudy) + vert_adv_u[k]
            adv_v = -(u[k]*dvdx + v[k]*dvdy) + vert_adv_v[k]
            adv_theta = -(u[k]*dthdx + v[k]*dthdy) + vert_adv_theta[k]
            adv_q = -(u[k]*dqdx + v[k]*dqdy) + vert_adv_q[k]

            pgf_x = -dPhidx - R*T[k]*(sigma_mid[k]/p[k])*dpstar_dx
            pgf_y = -dPhidy - R*T[k]*(sigma_mid[k]/p[k])*dpstar_dy

            if self.moist:
                theta_env = theta_from_T_p(sounding_T(sigma_mid[k]), sigma_mid[k]*PSTAR_FAR+p_top)
                radcool = -rad_active*(theta[k] - theta_env)/tau_R
            else:
                radcool = 0.0

            du[k] = adv_u + self.f*v[k] + pgf_x
            dv[k] = adv_v - self.f*u[k] + pgf_y
            dtheta[k] = adv_theta + radcool
            dq[k] = adv_q

            if self.moist and k == n-1:
                du[k] += Fu/mass_b
                dv[k] += Fv/mass_b
                dtheta[k] += F_SH/(cp*mass_b)
                dq[k] += Fq/mass_b

        return du, dv, dtheta, dq, dpstar_dt

    def convective_venting(self, dt):
        """
        Modified Ooyama (1969) convective closure (eqs. 16-27, 37). Only
        called when nz==2 (asserted in __init__), so this references layer
        indices 0,1,2 directly (=upper troposphere, lower/mid troposphere,
        boundary layer, exactly the paper's layer1/layer3/layerb) rather
        than a generalized loop -- the closure itself isn't a generic
        multi-level scheme (see module docstring).
        """
        theta, q, pstar = self.theta, self.q, self.pstar
        sigma_mid = self.sigma_mid
        p = [sigma_mid[k]*pstar + p_top for k in range(3)]
        Pi = [(p[k]/p0)**kappa for k in range(3)]
        T = [theta[k]*Pi[k] for k in range(3)]
        hyd = self.hydrostatic(theta, pstar)
        Phi = hyd['Phi']
        s = [cp*T[k] + Phi[k] for k in range(3)]
        h = [s[k] + Lv*q[k] for k in range(3)]
        qstar = [qsat(T[k], p[k]) for k in range(3)]
        hstar = [s[k] + Lv*qstar[k] for k in range(3)]

        trigger = h[2] > np.maximum(hstar[0], hstar[1])

        chi = np.clip(1.0 - q[1]/np.maximum(qstar[1], 1e-8), 0.0, 0.99)
        Mbar4 = -pstar*self._last_sigmadot[2]/g   # half-level 2 = boundary-layer-top interface
        Mc4 = np.where(trigger & (Mbar4 > 0), Mbar4/(1.0 - chi), 0.0)
        mass_b_cap = (pstar*self.d_sigma[2]/g)/1800.0
        Mc4 = np.minimum(Mc4, mass_b_cap)
        Md4 = chi*Mc4

        eta = 1.0 + (h[2] - hstar[0])/np.maximum(hstar[0] - h[1], 1e3)
        eta = np.clip(eta, 0.5, 3.0)
        Mc2 = eta*Mc4
        Mbar2 = -pstar*self._last_sigmadot[1]/g   # half-level 1 = interface between layer0,layer1
        clear2 = Mc2 - Mbar2

        q2 = 0.5*(q[0] + q[1])
        theta_hat_1 = hyd['theta_hat'][1]
        Phat_1 = hyd['Phat'][1]
        T2 = theta_hat_1 * Phat_1
        s2 = cp*T2 + hyd['Phi_at_halflevel'](1)

        qc1 = qstar[0]

        Phi4 = hyd['Phi_at_halflevel'](2)
        p4 = self.half_sigma[2]*pstar + p_top
        Tw4 = T[1].copy()
        for _ in range(20):
            Tw4_new = (h[1] - Phi4 - Lv*qsat(Tw4, p4)) / cp
            Tw4 = 0.5*Tw4 + 0.5*Tw4_new
        Tw4 = np.clip(Tw4, 150.0, 330.0)
        qd4 = qsat(Tw4, p4)
        sd4 = cp*Tw4 + Phi4

        mass_b = pstar*self.d_sigma[2]/g
        mass_3 = pstar*self.d_sigma[1]/g
        mass_1 = pstar*self.d_sigma[0]/g

        dq1 = (Mc2*(qc1 - q[0]) - clear2*(q2 - q[0])) / mass_1
        ds1 = (-clear2*(s2 - s[0])) / mass_1
        dq3 = (clear2*(q2 - q[1]) - Md4*(qd4 - q[1])) / mass_3
        ds3 = (clear2*(s2 - s[1]) - Md4*(sd4 - s[1])) / mass_3
        dqb = (Md4*(qd4 - q[2])) / mass_b
        dsb = (Md4*(sd4 - s[2])) / mass_b

        self.q[0] = np.maximum(self.q[0] + dt*dq1, 0.0)
        self.theta[0] = self.theta[0] + dt*ds1/(cp*Pi[0])
        self.q[1] = np.maximum(self.q[1] + dt*dq3, 0.0)
        self.theta[1] = self.theta[1] + dt*ds3/(cp*Pi[1])
        self.q[2] = np.maximum(self.q[2] + dt*dqb, 0.0)
        self.theta[2] = self.theta[2] + dt*dsb/(cp*Pi[2])

        for k in range(3):
            self._condense_layer(k)

        return Mc4

    def betts_miller_adjustment(self, dt):
        """
        Simplified Betts (1986)/Betts-Miller-style convective adjustment,
        generalized to any nz (see module docstring for why this replaces
        Ooyama's closure for nz != 2, and Baik/DeMaria/Raman 1990a,b,1991
        as the literature precedent for "Betts scheme + many levels + TC").

        This is a SIMPLIFIED realization, not the full literature scheme
        (which iteratively adjusts a reference profile shape to conserve
        column enthalpy and guarantee non-negative precipitation): the
        reference profile at each free-atmosphere layer is defined as the
        SATURATED state with the same moist static energy as the boundary
        layer,

            h_ref(k) = cp*T_ref(k) + Phi(k) + Lv*qsat(T_ref(k), p(k)) = h_b

        solved iteratively for T_ref(k) (same under-relaxation iteration
        already used for the Ooyama closure's downdraft wet-bulb solve --
        the naive fixed-point form is not contractive here either, for the
        same reason: |Lv/cp * dqsat/dT| typically exceeds 1). This
        represents "what the column would look like if fully neutralized
        by deep convection sourced from the boundary layer" -- a
        reasonable, defensible simplification of a true moist-adiabat
        reference profile, not an exact one. theta and q in each free-
        atmosphere layer then relax toward theta_ref=T_ref*(p0/p)^kappa
        and q_ref=qsat(T_ref,p) over tau_BM, wherever the same
        boundary-layer-instability trigger used by Ooyama's closure
        (h_b > max(hstar over free-atm layers)) is satisfied. The boundary
        layer itself is not relaxed (matches the classic scheme, where the
        boundary layer is the moisture/energy SOURCE for the reference
        profile, not itself adjusted).

        Does not explicitly conserve column moist static energy (theta and
        q relax independently, and Phi isn't updated mid-relaxation) --
        documented approximation, consistent with the many other
        documented simplifications already in this model.
        """
        theta, q, pstar = self.theta, self.q, self.pstar
        n = self.n
        sigma_mid = self.sigma_mid
        p = [sigma_mid[k]*pstar + p_top for k in range(n)]
        Pi = [(p[k]/p0)**kappa for k in range(n)]
        T = [theta[k]*Pi[k] for k in range(n)]
        hyd = self.hydrostatic(theta, pstar)
        Phi = hyd['Phi']
        s = [cp*T[k] + Phi[k] for k in range(n)]
        h = [s[k] + Lv*q[k] for k in range(n)]
        qstar = [qsat(T[k], p[k]) for k in range(n)]
        hstar = [s[k] + Lv*qstar[k] for k in range(n)]

        h_b = h[n-1]
        trigger = h_b > np.max(np.array(hstar[0:n-1]), axis=0)

        for k in range(n-1):
            T_ref = T[k].copy()
            for _ in range(20):
                T_ref_new = (h_b - Phi[k] - Lv*qsat(T_ref, p[k])) / cp
                T_ref = 0.5*T_ref + 0.5*T_ref_new
            T_ref = np.clip(T_ref, 150.0, 330.0)
            theta_ref = T_ref / Pi[k]
            q_ref = qsat(T_ref, p[k])

            dtheta = np.where(trigger, -(theta[k] - theta_ref)/tau_BM, 0.0)
            dq = np.where(trigger, -(q[k] - q_ref)/tau_BM, 0.0)
            self.theta[k] = self.theta[k] + dt*dtheta
            self.q[k] = np.maximum(self.q[k] + dt*dq, 0.0)
            self._condense_layer(k)

    def _condense_layer(self, k):
        p_k = self.sigma_mid[k]*self.pstar + p_top
        Pi_k = (p_k/p0)**kappa
        for _ in range(3):
            T_k = self.theta[k] * Pi_k
            qs_k = qsat(T_k, p_k)
            excess = np.maximum(self.q[k] - qs_k, 0.0)
            if not np.any(excess > 0):
                break
            self.q[k] -= excess
            self.theta[k] += Lv*excess/(cp*Pi_k)

    def step(self, dt):
        """Advance the model state by one time step `dt` (seconds), in
        place: 3rd-order Adams-Bashforth for the dynamics (`rhs()`,
        automatically dropping to lower-order AB for the first two calls
        while `_prev_tendencies` fills up), then 4th-order horizontal
        diffusion as a separate explicit-Euler correction (split from the
        AB3-integrated tendency for stability, see dev log), then -- if
        `moist` -- condensation and the selected convective closure."""
        tend = self.rhs()
        self._prev_tendencies.append(tend)
        if len(self._prev_tendencies) > 3:
            self._prev_tendencies.pop(0)
        n_hist = len(self._prev_tendencies)

        if n_hist == 1:
            coefs = [1.0]
        elif n_hist == 2:
            coefs = [-0.5, 1.5]
        else:
            coefs = [5/12, -16/12, 23/12]

        du = sum(c*t[0] for c, t in zip(coefs, self._prev_tendencies))
        dv = sum(c*t[1] for c, t in zip(coefs, self._prev_tendencies))
        dtheta = sum(c*t[2] for c, t in zip(coefs, self._prev_tendencies))
        dq = sum(c*t[3] for c, t in zip(coefs, self._prev_tendencies))
        dpstar = sum(c*t[4] for c, t in zip(coefs, self._prev_tendencies))

        self.u = self.u + dt*du
        self.v = self.v + dt*dv
        self.theta = self.theta + dt*dtheta
        self.q = np.maximum(self.q + dt*dq, 0.0)
        self.pstar = self.pstar + dt*dpstar

        k1_diff = 0.0008*self.dx**4
        for k in range(self.n):
            self.u[k] = self.u[k] + dt*self.diffuse4(self.u[k], k1_diff)
            self.v[k] = self.v[k] + dt*self.diffuse4(self.v[k], k1_diff)
            self.theta[k] = self.theta[k] + dt*self.diffuse4(self.theta[k], k1_diff)
            self.q[k] = np.maximum(self.q[k] + dt*self.diffuse4(self.q[k], k1_diff), 0.0)
        # pstar was missing this same 4th-order hyperdiffusion that every other prognostic
        # field gets -- without it, the meridional (rigid-wall) centered-difference scheme's
        # undamped 2-delta-y null mode grows unchecked in pstar specifically (confirmed via a
        # 14-day truth run: pstar's y-profile showed a ~91% grid-point sign-flip rate, a
        # near-perfect checkerboard, vs. 4-11% for every diffused field, with std growing from
        # ~70 Pa at t=0 to 700-840 Pa by day 4-13).
        self.pstar = self.pstar + dt*self.diffuse4(self.pstar, k1_diff)

        if self.moist:
            for k in range(self.n):
                self._condense_layer(k)
            if self.convection_scheme == 'ooyama':
                self.convective_venting(dt)
            else:
                self.betts_miller_adjustment(dt)
            for k in range(self.n):
                self._condense_layer(k)

        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0

    def diagnostics(self):
        """Per-layer summary statistics (max wind speed, theta range, q
        range, all as plain lists indexed by layer 0..nz-1=free atmosphere
        then nz=boundary) plus the domain pstar range -- a quick sanity/
        progress check, not a substitute for the full state."""
        wind = np.hypot(self.u, self.v)
        return dict(
            max_wind=[float(wind[k].max()) for k in range(self.n)],
            min_theta=[float(self.theta[k].min()) for k in range(self.n)],
            max_theta=[float(self.theta[k].max()) for k in range(self.n)],
            q_range=[(float(self.q[k].min()), float(self.q[k].max())) for k in range(self.n)],
            pstar_range=(float(self.pstar.min()), float(self.pstar.max())),
        )
