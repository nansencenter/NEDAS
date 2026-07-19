"""
Zhu, Smith & Ulrich (2001) minimal 3D tropical cyclone model -- dynamical
core + moisture/surface-flux/radiation physics (phase 2). Convective
parameterization (Ooyama closure) not yet added -- see phase-3 task.

Sigma-coordinate primitive equations on an f-plane, 3 layers (1=upper
troposphere, 3=lower/mid troposphere, b=boundary layer) + interfaces at
sigma2, sigma4 carrying sigma-dot. Finite differences (periodic in x, rigid
wall in y), 3rd-order Adams-Bashforth time stepping.

Not yet wired into NEDAS -- standalone prototype for fast iteration.

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
f0 = 2 * 7.292e-5 * np.sin(np.deg2rad(20.))  # f-plane at 20N
Lv = 2.501e6      # latent heat of vaporization, J/kg
SST = 301.15      # 28 C, paper's value
tau_R = 12*3600.  # Newtonian radiative cooling timescale, s

# sigma levels (paper's values, Fig. 1)
sigma1, sigma2, sigma3, sigma4, sigmab = 1/6, 1/3, 11/18, 8/9, 17/18
d_sigma1 = sigma2 - 0.0     # layer-1 thickness in sigma
d_sigma3 = sigma4 - sigma2  # layer-3 thickness
d_sigmab = 1.0 - sigma4     # boundary-layer thickness

# initial sounding at each layer midpoint (Appendix A, Table A1)
# p (mb), q (g/kg), T (K); index 0=layer1, 1=layer3, 2=layerb
SOUNDING_P = np.array([252.5, 659.2, 964.2]) * 100.0     # Pa
SOUNDING_Q = np.array([0.01, 4.82, 13.9]) * 1e-3          # kg/kg
SOUNDING_T = np.array([230.1, 278.9, 297.0])              # K


def theta_from_T_p(T, p):
    return T * (p0 / p) ** kappa


SOUNDING_THETA = theta_from_T_p(SOUNDING_T, SOUNDING_P)


def smith_vortex(r, vm=15.0, rm=120.0e3):
    """Smith et al. (1990) tangential wind profile, eq. (13)."""
    a = 1.78803
    b = 4.74736e-3
    c = 0.339806
    d = 5.37727e-4
    rp = r / rm
    return a * vm * rp * (1 + b * rp**4) / (1 + c * rp**2 + d * rp**6) ** 2


def make_grid(nx, ny, dx):
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

    Integrand is >=0 everywhere, so the cumulative integral increases
    monotonically outward from the center (r=0) -- pressure is lowest at
    the vortex core, consistent with a cyclone, and asymptotes to a
    constant once v_tan decays to ~0. Anchor that asymptotic value to the
    far-field pressure (pert=0) rather than the center, since it's the
    far-field (ambient) pressure that's actually known.
    """
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    integrand = rho0 * (f0 * v_tan + v_tan**2 / np.maximum(r, 1e-6))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dr)])
    pstar_pert = cum - cum[-1]  # 0 in the far field, negative (low) at the core
    return pstar_pert


def qsat(T, p):
    """
    Saturation specific humidity via Bolton (1980)'s saturation vapor
    pressure formula. T in K, p in Pa; returns kg/kg.
    """
    T_c = T - 273.15
    es = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))  # Pa
    return 0.622 * es / np.maximum(p - 0.378*es, 1.0)


def surface_drag_coef(Vb):
    """Shapiro (1992) neutral drag coefficient, eq. (10) note in the paper
    (R_F=0.8, wind already taken as the near-surface/boundary-layer wind --
    a simplification of the paper's "reduced to 10m" diagnostic wind).

    Shapiro's formula is linear in V with no ceiling, which is a known
    idealization -- observations (e.g. Powell et al. 2003) show Cd actually
    levels off and even decreases above ~30-35 m/s in the real eyewall.
    Without that saturation, the surface-flux/wind feedback here is
    unbounded (found via a 48h test: wind reached 82 m/s by t=11h then
    NaN'd shortly after, even with downdrafts and a convective mass-flux
    cap in place -- see dev log). Evaluate the linear formula at
    min(Vb, V_cd_cap) rather than Vb directly.
    """
    R_F = 0.8
    V_cd_cap = 33.0
    return (1.024 + 0.05366*R_F*np.minimum(Vb, V_cd_cap)) * 1e-3


def random_flow(nx, ny, dx, amp, power_law, seed=None):
    """
    Random background wind field with a prescribed kinetic-energy power-law
    spectrum, same construction as NEDAS's vort2d util.random_flow (random
    streamfunction -> centered-difference wind, amplitude-normalized).
    Streamfunction power law = wind power law - 2. Self-contained here
    (doesn't import NEDAS) since this is a standalone prototype; the FFT-
    based generation implicitly treats the domain as periodic, which is a
    minor inconsistency with the dry core's rigid-wall y boundary but is
    only used to seed the initial condition, not enforced afterward (v is
    zeroed at the walls on the first step's BC application regardless).
    """
    rng = np.random.default_rng(seed)
    noise_hat = np.fft.fft2(rng.standard_normal((ny, nx)))
    ki = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    kj = np.fft.fftfreq(ny, d=dx) * 2*np.pi
    KI, KJ = np.meshgrid(ki, kj)
    K = np.sqrt(KI**2 + KJ**2)
    K[0, 0] = 1.0  # placeholder, psi_hat[0,0] zeroed below regardless
    psi_hat = noise_hat * K**((power_law - 2) / 2.0)
    psi_hat[0, 0] = 0.0
    psi = np.real(np.fft.ifft2(psi_hat))
    u = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2*dx)
    v = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2*dx)
    u = amp * (u - u.mean()) / u.std()
    v = amp * (v - v.mean()) / v.std()
    return u, v


class Core:
    def __init__(self, nx=100, ny=100, dx=20e3, Vbg=0.0, Vslope=-3, bg_seed=None,
                 beta=0.0, moist=True):
        self.nx, self.ny, self.dx = nx, ny, dx
        self.xx, self.yy = make_grid(nx, ny, dx)
        self.rr = np.hypot(self.xx, self.yy)

        # Coriolis parameter f=f0+beta*y (paper's eqs. 2-3; f0 is the value
        # at y=0, beta=df/dy). beta=0 (pure f-plane) reproduces the paper's
        # own experiments exactly; beta>0 enables beta-drift (the vortex
        # self-advecting via the asymmetric flow the Coriolis gradient
        # induces), not otherwise present since the vortex starts exactly
        # centered and axisymmetric on an f-plane.
        self.beta = beta
        self.f = f0 + beta*self.yy

        # moist=False switches off surface fluxes, radiative cooling,
        # condensation, and convective venting -- q remains a state
        # variable (for a simpler code path) but is then purely advected,
        # with no sources/sinks, i.e. runs the phase-1 dry dynamical core.
        self.moist = moist

        # prognostic fields: u,v,theta,q at 3 layers (0=layer1,1=layer3,2=layerb)
        self.u = np.zeros((3, ny, nx))
        self.v = np.zeros((3, ny, nx))
        self.theta = np.zeros((3, ny, nx))
        self.q = np.zeros((3, ny, nx))
        for k in range(3):
            self.theta[k] = SOUNDING_THETA[k]
            self.q[k] = SOUNDING_Q[k]

        ps_far = 1015.0e2  # nominal far-field surface pressure
        self.pstar = np.full((ny, nx), ps_far - p_top)

        # initial vortex: same tangential wind at each layer (paper: "the
        # initial axisymmetric vortex is barotropic")
        vtan = smith_vortex(self.rr)
        theta_ang = np.arctan2(self.yy, self.xx)
        for k in range(3):
            self.u[k] = -vtan * np.sin(theta_ang)
            self.v[k] = vtan * np.cos(theta_ang)

        # random background flow (same field added to all 3 layers, as with
        # the vortex itself), same construction as vort2d's initial_condition
        if Vbg > 0:
            u_bg, v_bg = random_flow(nx, ny, dx, Vbg, Vslope, seed=bg_seed)
            for k in range(3):
                self.u[k] += u_bg
                self.v[k] += v_bg
            self.v[:, 0, :] = 0.0
            self.v[:, -1, :] = 0.0

        # radial pressure perturbation via simplified gradient-wind balance
        r1d = np.linspace(0, self.rr.max(), 2000)
        v1d = smith_vortex(r1d)
        rho0 = SOUNDING_P[2] / (R * SOUNDING_T[2])
        pstar_pert_1d = gradient_wind_balance_pstar(r1d, v1d, f0, rho0)
        pstar_pert = np.interp(self.rr.ravel(), r1d, pstar_pert_1d).reshape(self.rr.shape)
        self.pstar += pstar_pert

        self._prev_tendencies = []  # for Adams-Bashforth

    def ddx(self, f):
        return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * self.dx)

    def ddy_interior(self, f):
        # 2nd-order centered in y; zero-gradient at edges (rigid-wall
        # companion condition, v itself is forced to 0 at the wall separately)
        d = np.zeros_like(f)
        d[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * self.dx)
        return d

    def _pad_y(self, f, n=2):
        """edge-replicate pad in y (rigid-wall approximation for the
        upwind stencil's extra ghost points)"""
        top = np.repeat(f[0:1, :], n, axis=0)
        bot = np.repeat(f[-1:, :], n, axis=0)
        return np.vstack([top, f, bot])

    def upwind3_dx(self, f, u):
        """3rd-order upwind-biased x-derivative (paper's Appendix B
        numerics), periodic. Standard 4-point-stencil upwind-biased scheme:
        unlike centered differencing, its leading truncation-error term is
        even-order (dissipative), giving inherent numerical damping of
        grid-scale noise -- switched to this (from 2nd-order centered)
        specifically because centered advection couldn't handle the sharp
        local gradients the convective closure creates; see dev log for the
        diagnosis (persistent, unresolved boundary-layer blowups under three
        independent stabilization attempts with centered advection)."""
        fm2 = np.roll(f, 2, axis=-1)
        fm1 = np.roll(f, 1, axis=-1)
        fp1 = np.roll(f, -1, axis=-1)
        fp2 = np.roll(f, -2, axis=-1)
        d_pos = (fm2 - 6*fm1 + 3*f + 2*fp1) / (6*self.dx)
        d_neg = (-2*fm1 - 3*f + 6*fp1 - fp2) / (6*self.dx)
        return np.where(u >= 0, d_pos, d_neg)

    def upwind3_dy(self, f, v):
        """3rd-order upwind-biased y-derivative, rigid-wall (edge-replicate
        padded, an approximation near the walls -- documented deviation)."""
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
        Geopotential at each layer midpoint and at the sigma2/sigma4
        interfaces, via the Arakawa & Suarez (1983) vertical differencing
        the paper explicitly adopts (Appendix B) -- NOT the naive local
        Exner function Pi(sigma_k)=(p_k/p0)**kappa used in an earlier
        version of this routine. Appendix B defines, for interior levels
        k with bounding half-levels (interfaces) k-1/2, k+1/2:

            Phat_{k+1/2} = (phat_{k+1/2}/p0)**kappa   -- naive/actual Exner
                                                          at the interface's
                                                          own pressure
            P_k = [Phat_{k+1/2}*phat_{k+1/2} - Phat_{k-1/2}*phat_{k-1/2}]
                  / [(1+kappa)*(phat_{k+1/2}-phat_{k-1/2})]
                                                       -- layer-MEAN Exner
            theta_hat_{k+1/2} = [(Phat_{k+1/2}-P_k)*theta_k
                                 + (P_{k+1}-Phat_{k+1/2})*theta_{k+1}]
                                / (P_{k+1}-P_k)        -- interface theta,
                                                          weighted so that
                                                          theta stays exactly
                                                          conserved
            Phi_k - Phi_{k+1} = cp*theta_hat_{k+1/2}*(P_{k+1}-P_k)
            Phi_b - Phi_s = cp*theta_b*(P_s - P_b)     -- lowest level: no
                                                          layer below the
                                                          boundary layer, so
                                                          this span uses
                                                          theta_b directly
                                                          rather than a
                                                          theta_hat

        P_k differs from the naive Pi(sigma_k) -- it's a mass-weighted mean
        over the layer's own two bounding interfaces, not the Exner function
        evaluated at the layer's midpoint pressure. Using the naive Pi_k
        (as an earlier version of this code did) is a systematic deviation
        from the paper's actual scheme, most consequential for the
        convective closure's moist static energy differences (h1*-h3 etc.),
        which are already the paper's most delicate numerical quantity (see
        dev log's phase-3 eta-runaway discussion).

        Phi at the interfaces themselves (sigma2, sigma4) isn't directly
        given by the paper's formula above (only midpoint-to-midpoint), but
        is needed for eq. 28 (downdraft wet-bulb calc, needs Phi4) and for
        s2 in eqs. 22/25 (needs Phi2). Extended here self-consistently: the
        same theta_hat used for the full layer-to-layer span is evaluated
        over the partial sub-span from one layer's midpoint to the shared
        interface; the two sub-spans sum exactly to the paper's full-span
        formula, e.g. cp*theta_hat_2*(P2-Phat_2) + cp*theta_hat_2*(Phat_2-P1)
        = cp*theta_hat_2*(P2-P1).
        """
        theta1, theta3, thetab = theta[0], theta[1], theta[2]

        def Phat(sigma):
            p = sigma*pstar + p_top
            return (p/p0)**kappa, p

        Phat_top, phat_top = Phat(0.0)
        Phat_2, phat_2 = Phat(sigma2)
        Phat_4, phat_4 = Phat(sigma4)
        Phat_s, phat_s = Phat(1.0)

        def Pmean(Ph_lo, ph_lo, Ph_hi, ph_hi):
            return (Ph_hi*ph_hi - Ph_lo*ph_lo) / ((1+kappa)*(ph_hi-ph_lo))

        P1 = Pmean(Phat_top, phat_top, Phat_2, phat_2)
        P2 = Pmean(Phat_2, phat_2, Phat_4, phat_4)
        P3 = Pmean(Phat_4, phat_4, Phat_s, phat_s)

        def theta_hat(Ph_mid, th_lo, P_lo, th_hi, P_hi):
            return ((Ph_mid-P_lo)*th_lo + (P_hi-Ph_mid)*th_hi) / (P_hi-P_lo)

        theta_hat_2 = theta_hat(Phat_2, theta1, P1, theta3, P2)
        theta_hat_4 = theta_hat(Phat_4, theta3, P2, thetab, P3)

        Phi_s = np.zeros_like(pstar)
        Phi_b_mid = Phi_s + cp*thetab*(Phat_s - P3)
        Phi_4 = Phi_b_mid + cp*theta_hat_4*(P3 - Phat_4)
        Phi_3_mid = Phi_b_mid + cp*theta_hat_4*(P3 - P2)
        Phi_2 = Phi_3_mid + cp*theta_hat_2*(P2 - Phat_2)
        Phi_1_mid = Phi_3_mid + cp*theta_hat_2*(P2 - P1)

        return dict(
            Phi=np.array([Phi_1_mid, Phi_3_mid, Phi_b_mid]),
            Phi2=Phi_2, Phi4=Phi_4,
            theta_hat_2=theta_hat_2, Phat_2=Phat_2,
        )

    def rhs(self):
        u, v, theta, q, pstar = self.u, self.v, self.theta, self.q, self.pstar
        sigmas = [sigma1, sigma3, sigmab]
        d_sigmas = [d_sigma1, d_sigma3, d_sigmab]

        p = [s*pstar + p_top for s in sigmas]
        T = [theta[k] * (p[k]/p0)**kappa for k in range(3)]
        Phi = self.hydrostatic(theta, pstar)['Phi']

        dpstar_dx = self.ddx(pstar)
        dpstar_dy = self.ddy_interior(pstar)

        # mass divergence per layer: D_k = div(p* V_k) = p*(du/dx+dv/dy) + V.grad(p*)
        D = []
        for k in range(3):
            div_uv = self.ddx(u[k]) + self.ddy_interior(v[k])
            D.append(pstar*div_uv + u[k]*dpstar_dx + v[k]*dpstar_dy)

        dpstar_dt = -(D[0]*d_sigma1 + D[1]*d_sigma3 + D[2]*d_sigmab)

        # p*.sigma_dot at interfaces sigma2 (between layer1/layer3) and
        # sigma4 (between layer3/layerb), from the vertically-integrated
        # continuity constraint (sigma_dot=0 at sigma=0,1)
        pstar_sigmadot_2 = -(D[0]*d_sigma1) - sigma2*dpstar_dt
        pstar_sigmadot_4 = -(D[0]*d_sigma1 + D[1]*d_sigma3) - sigma4*dpstar_dt
        sigmadot_2 = pstar_sigmadot_2 / pstar
        sigmadot_4 = pstar_sigmadot_4 / pstar
        self._last_sigmadot_4 = sigmadot_4  # used by convective_venting()
        self._last_sigmadot_2 = sigmadot_2

        def vert_adv(chi):
            """The '-sigma_dot*dchi/dsigma' tendency contribution (eqs. 2,3,
            7,8), discretized via Appendix B's flux-form vertical scheme
            (sigma_dot=0 at sigma=0,1): for layer k with bounding interface
            sigma_dots and simple-average interface values chihat=(chi_k+
            chi_k+1)/2, the exact advection-form reduction of Appendix B's
            flux-form equation (derived by subtracting chi_k times the
            layer continuity equation from the flux-form equation, both
            given in Appendix B; re-derived and confirmed symbolically
            with sympy here since it's easy to sign-flip by hand) is

                dchi_k/dt|vert = -(1/dsigma_k)*[sdot_hi*(chihat_hi-chi_k)
                                                 + sdot_lo*(chi_k-chihat_lo)]

            This IS the full vertical-advection tendency term already
            (sign included) -- it must be ADDED to the horizontal-advection
            tendency below, not subtracted. An earlier version of this code
            subtracted it (equivalent to flipping the sign of sigma_dot
            throughout the vertical advection only), confirmed wrong both
            by this derivation and by a physical check: subsidence
            (sigma_dot>0) between the warm upper layer and cooler layer
            below must WARM the layer below (replacing it with warmer air
            from aloft) -- the subtracted form gives the opposite sign."""
            chi1, chi3, chib = chi[0], chi[1], chi[2]
            adv1 = sigmadot_2*(chi1 - chi3) / (2*d_sigma1)
            adv3 = (sigmadot_4*(chi3 - chib) + sigmadot_2*(chi1 - chi3)) / (2*d_sigma3)
            advb = sigmadot_4*(chi3 - chib) / (2*d_sigmab)
            return np.array([adv1, adv3, advb])

        vert_adv_u = vert_adv(u)
        vert_adv_v = vert_adv(v)
        vert_adv_theta = vert_adv(theta)
        vert_adv_q = vert_adv(q)

        du = np.zeros_like(u)
        dv = np.zeros_like(v)
        dtheta = np.zeros_like(theta)
        dq = np.zeros_like(q)

        # surface fluxes (bulk aerodynamic, eq. 10) and Newtonian radiative
        # cooling -- both switched off when moist=False (dry-core-only
        # config), leaving pure advection-diffusion for theta/q. Applied to
        # the boundary layer only. Rates are slow (drag/exchange timescales
        # ~O(10h), see dev log) so unlike diffusion these are safely within
        # AB3's stability region and can go directly into the AB3 tendency.
        if self.moist:
            Vb = np.hypot(u[2], v[2])
            Cd = surface_drag_coef(Vb)
            Ch = Cd  # paper: Ch=Cd
            rho_b = p[2] / (R*T[2])
            qs_sst = qsat(np.full_like(pstar, SST), p[2])
            Fu = -rho_b*Cd*Vb*u[2]
            Fv = -rho_b*Cd*Vb*v[2]
            Fq = rho_b*Ch*Vb*(qs_sst - q[2])
            F_SH = rho_b*cp*Ch*Vb*(SST - T[2])
            mass_b = pstar*d_sigmab/g  # boundary-layer mass per unit area, kg/m^2

            # suppressed where layer-3 RH>90% (cloud shielding)
            qs3 = qsat(T[1], p[1])
            rh3 = q[1] / np.maximum(qs3, 1e-8)
            rad_active = (rh3 <= 0.9).astype(float)

        for k in range(3):
            # advective derivatives: 3rd-order upwind (see upwind3_dx/dy docstring)
            dudx, dudy = self.upwind3_dx(u[k], u[k]), self.upwind3_dy(u[k], v[k])
            dvdx, dvdy = self.upwind3_dx(v[k], u[k]), self.upwind3_dy(v[k], v[k])
            dthdx, dthdy = self.upwind3_dx(theta[k], u[k]), self.upwind3_dy(theta[k], v[k])
            dqdx, dqdy = self.upwind3_dx(q[k], u[k]), self.upwind3_dy(q[k], v[k])
            dPhidx, dPhidy = self.ddx(Phi[k]), self.ddy_interior(Phi[k])

            adv_u = -(u[k]*dudx + v[k]*dudy) + vert_adv_u[k]
            adv_v = -(u[k]*dvdx + v[k]*dvdy) + vert_adv_v[k]
            adv_theta = -(u[k]*dthdx + v[k]*dthdy) + vert_adv_theta[k]
            adv_q = -(u[k]*dqdx + v[k]*dqdy) + vert_adv_q[k]

            pgf_x = -dPhidx - R*T[k]*(sigmas[k]/p[k])*dpstar_dx
            pgf_y = -dPhidy - R*T[k]*(sigmas[k]/p[k])*dpstar_dy

            radcool = -rad_active*(theta[k] - SOUNDING_THETA[k])/tau_R if self.moist else 0.0

            # NOTE: diffusion is deliberately NOT included here -- see step().
            # It's a stiff, rapidly-decaying (real-eigenvalue) term, and 3rd-
            # order Adams-Bashforth has a very narrow stability region on the
            # negative real axis (|dt*rate| <~ 0.545) compared to explicit
            # Euler's (|dt*rate| < 2). Folding hyperdiffusion into the AB3
            # tendency exceeded that bound at the paper's own dt=15s/k1
            # values and blew up within ~70 steps (diagnosed empirically:
            # increasing k1 made it fail *faster*, the signature of exceeding
            # a stability threshold rather than underdamped grid noise).
            # Standard fix (matches how these models typically split stiff
            # diffusion from an AB/leapfrog dynamical core): apply diffusion
            # as a separate explicit-Euler correction after the AB3 update.
            du[k] = adv_u + self.f*v[k] + pgf_x
            dv[k] = adv_v - self.f*u[k] + pgf_y
            dtheta[k] = adv_theta + radcool
            dq[k] = adv_q

            if self.moist and k == 2:  # boundary layer: add surface fluxes
                du[k] += Fu/mass_b
                dv[k] += Fv/mass_b
                dtheta[k] += F_SH/(cp*mass_b)
                dq[k] += Fq/mass_b

        return du, dv, dtheta, dq, dpstar_dt

    def convective_venting(self, dt):
        """
        Modified Ooyama (1969) convective closure, implementing the paper's
        actual shared mass/moisture-conservation machinery (eqs. 16-27) with
        the Ooyama-specific mass-flux closure (eq. 37), replacing an earlier
        hand-derived approximation that repeatedly ran away under several
        independent stabilization attempts (see dev log for that history).

        Trigger (paper, Sec. 3): deep convection occurs where boundary-layer
        moist static energy h_b exceeds the saturated MSE of both the upper
        and lower troposphere, h1* and h3*.

        Cloud model (eqs. 16-20): cloud-base mass flux Mc4 rises from the
        boundary layer, entrains layer-3 environmental air at rate
        Me=(eta-1)*Mc4, and detrains into layer1 at rate Mc2=eta*Mc4 -- the
        detrained air is saturated at layer1's OWN environmental temperature
        (zero-buoyancy detrainment), not raw undiluted boundary-layer air.
        (An earlier version injected raw boundary-layer moisture directly
        into layer1's low pressure, where Lv*excess/(cp*Pi) blows up as the
        Exner function Pi->0 aloft -- a real, now-fixed bug.)

        Downdrafts (eq. 21, 28): mass flux Md4=chi*Mc4 sinks from layer3 to
        the boundary layer, carrying air cooled to its wet-bulb temperature
        by evaporating rain (solved iteratively). chi=1-eps, eps=q3/q3*
        (paper's own state-dependent definition -- replaces an earlier fixed
        chi=0.3 placeholder, which lacked this self-limiting feedback: as
        layer3 moistens, chi shrinks and downdraft cooling weakens, exactly
        opposite of what a fixed constant would give).

        Ooyama-specific (eq. 37): Mc4 tied directly to the resolved-scale
        mass flux at the boundary-layer top, Mc4=Mbar4/(1-chi). This
        identically forces the "clear-air" exchange term at level 4 to zero
        (Me4=Mc4-Md4-Mbar4=0 by construction), which drops several terms
        from the general eqs. 22-27 -- implemented below in that simplified
        (Ooyama-specific) form, not the fully general 3-scheme version.
        """
        theta, q, pstar = self.theta, self.q, self.pstar
        sigmas = [sigma1, sigma3, sigmab]
        p = [s*pstar + p_top for s in sigmas]
        Pi = [(p[k]/p0)**kappa for k in range(3)]
        T = [theta[k]*Pi[k] for k in range(3)]
        hyd = self.hydrostatic(theta, pstar)
        Phi = hyd['Phi']
        s = [cp*T[k] + Phi[k] for k in range(3)]
        h = [s[k] + Lv*q[k] for k in range(3)]
        qstar = [qsat(T[k], p[k]) for k in range(3)]
        hstar = [s[k] + Lv*qstar[k] for k in range(3)]

        trigger = h[2] > np.maximum(hstar[0], hstar[1])

        # eq. 37: Mc4 = Mbar4/(1-chi), Mbar4 = resolved-scale mass flux at
        # the boundary-layer top (positive upward)
        chi = np.clip(1.0 - q[1]/np.maximum(qstar[1], 1e-8), 0.0, 0.99)
        Mbar4 = -pstar*self._last_sigmadot_4/g
        Mc4 = np.where(trigger & (Mbar4 > 0), Mbar4/(1.0 - chi), 0.0)
        # safety limiter: cap Mc4 so a single step can't vent more than a
        # ~30min e-folding fraction of the boundary layer's mass, regardless
        # of how large the (now-corrected) formula computes it. Standard,
        # defensible practice in real convective parameterizations
        # independent of how exact the underlying thermodynamics are --
        # kept even after finding and fixing three real bugs upstream
        # (units error, unphysical layer1 injection, unstable wet-bulb
        # iteration, and a Phi sign error) as a numerical safety net.
        mass_b_cap = (pstar*d_sigmab/g)/1800.0
        Mc4 = np.minimum(Mc4, mass_b_cap)
        Md4 = chi*Mc4

        # eq. 19: eta=Mc2/Mc4, using layer3's ACTUAL (not saturated) h.
        # The paper notes h1*>h3 "is always satisfied in the present
        # calculations" -- true for their specific setup, not a general
        # guarantee. Traced a full-domain runaway to exactly this: eta grew
        # smoothly (1.09 at t=0.5h -> 5.30 at t=4h, physically reasonable)
        # then exploded to 255 by t=6h as h1*-h3 shrank through zero and
        # went negative (layer3 warmed enough, via the closure's own
        # entrainment/downdraft terms, to violate the paper's assumption).
        # eta represents a detrainment/updraft ratio that should stay
        # modest for a physically sensible cloud model -- clip it directly
        # rather than only flooring the denominator (which still let eta
        # reach values >250 before the floor engaged).
        eta = 1.0 + (h[2] - hstar[0])/np.maximum(hstar[0] - h[1], 1e3)
        eta = np.clip(eta, 0.5, 3.0)
        Mc2 = eta*Mc4
        Mbar2 = -pstar*self._last_sigmadot_2/g
        clear2 = Mc2 - Mbar2  # the "(Mc2-Mbar2)" clear-air exchange term

        # interface-2 values, per the paper's Appendix B: q (like wind
        # speed) uses the simple arithmetic mean of the two adjacent
        # layers, Ahat_{k+1/2}=(A_k+A_{k+1})/2; s uses the Arakawa-Suarez
        # theta-hat (weighted by layer-mean Exner functions, see
        # hydrostatic()'s docstring) converted to T via the interface's own
        # (naive) Exner function, plus the interface geopotential Phi2 --
        # replaces an earlier sigma-linear interpolation that matched
        # neither of the paper's two different actual interface formulas.
        q2 = 0.5*(q[0] + q[1])
        T2 = hyd['theta_hat_2'] * hyd['Phat_2']
        s2 = cp*T2 + hyd['Phi2']

        qc1 = qstar[0]  # detrained cloud air is saturated at layer1's own T

        # downdraft wet-bulb temperature at level 4 (eq. 28): solve
        # cp*Tw4 + Phi4 + Lv*qsat(Tw4,p4) = h3 iteratively
        Phi4 = hyd['Phi4']
        p4 = sigma4*pstar + p_top
        # naive fixed-point iteration (Tw4 = g(Tw4)) is NOT contractive here
        # -- |dg/dT| = Lv/cp * dqsat/dT typically exceeds 1, so it diverges
        # rather than converges (found via an immediate, ~5-minute NaN in
        # the first test of this scheme -- see dev log). Under-relaxation
        # (blend old/new estimate each iteration) is the standard fix for
        # this class of saturation-adjustment iteration.
        Tw4 = T[1].copy()
        for _ in range(20):
            Tw4_new = (h[1] - Phi4 - Lv*qsat(Tw4, p4)) / cp
            Tw4 = 0.5*Tw4 + 0.5*Tw4_new
        Tw4 = np.clip(Tw4, 150.0, 330.0)  # safety bound
        qd4 = qsat(Tw4, p4)
        sd4 = cp*Tw4 + Phi4

        mass_b = pstar*d_sigmab/g
        mass_3 = pstar*d_sigma3/g
        mass_1 = pstar*d_sigma1/g

        # eqs. 22,25 (layer1), 23,26 (layer3, Me4 term dropped: Ooyama sets
        # it identically to 0), 24,27 (boundary layer, same)
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

    def _condense_layer(self, k):
        """explicit condensation for a single layer (iterated a few times
        since qsat depends on T, which the latent-heat release just
        changed) -- shared by convective_venting()'s cascade and step()'s
        end-of-step condense() pass."""
        sigmas = [sigma1, sigma3, sigmab]
        p_k = sigmas[k]*self.pstar + p_top
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
        tend = self.rhs()
        self._prev_tendencies.append(tend)
        if len(self._prev_tendencies) > 3:
            self._prev_tendencies.pop(0)
        n = len(self._prev_tendencies)

        if n == 1:
            coefs = [1.0]
        elif n == 2:
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

        # diffusion: separate explicit-Euler correction, not part of the AB3
        # tendency -- see the note in rhs()
        k1_diff = 0.0008*self.dx**4
        for k in range(3):
            self.u[k] = self.u[k] + dt*self.diffuse4(self.u[k], k1_diff)
            self.v[k] = self.v[k] + dt*self.diffuse4(self.v[k], k1_diff)
            self.theta[k] = self.theta[k] + dt*self.diffuse4(self.theta[k], k1_diff)
            self.q[k] = np.maximum(self.q[k] + dt*self.diffuse4(self.q[k], k1_diff), 0.0)

        if self.moist:
            for k in range(3):
                self._condense_layer(k)

            # Ooyama convective venting: moves boundary-layer theta/q up into
            # layer3/layer1 when convectively unstable -- without this the
            # boundary layer's surface-flux-driven moist static energy has no
            # sink and runs away within ~1 day (see dev log). Condense again
            # afterward since the injected air is often immediately
            # supersaturated at its new (cooler) level.
            self.convective_venting(dt)
            for k in range(3):
                self._condense_layer(k)

        # rigid-wall BC: v=0 at y boundaries
        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0

    def diagnostics(self):
        wind = np.hypot(self.u, self.v)
        return dict(
            max_wind=[float(wind[k].max()) for k in range(3)],
            min_theta=[float(self.theta[k].min()) for k in range(3)],
            max_theta=[float(self.theta[k].max()) for k in range(3)],
            q_range=[(float(self.q[k].min()), float(self.q[k].max())) for k in range(3)],
            pstar_range=(float(self.pstar.min()), float(self.pstar.max())),
        )


if __name__ == '__main__':
    core = Core(nx=100, ny=100, dx=20e3)
    dt = 15.0
    print("t=0s:", core.diagnostics())
    nsteps = int(6*3600/dt)  # 6 hours first, cheap sanity check
    for step in range(nsteps):
        core.step(dt)
        if np.any(np.isnan(core.u)) or np.any(np.isnan(core.pstar)):
            print(f"NaN at step {step} (t={step*dt:.0f}s)")
            print("diagnostics before NaN would need earlier check")
            raise RuntimeError(f"NaN at step {step} (t={step*dt:.0f}s)")
        if (step+1) % int(3600/dt) == 0:
            print(f"t={(step+1)*dt/3600:.1f}h:", core.diagnostics())
