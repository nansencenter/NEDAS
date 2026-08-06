"""
Adapts the standalone core.Core prototype (sigma-coordinate primitive-
equation dynamical core + optional moist physics/convective closure, Zhu,
Smith & Ulrich 2001) to NEDAS's plain-dict state representation.

Core itself stays a self-contained, NEDAS-independent module (core.py,
unmodified) -- it's easier to keep validating/extending against the paper in
isolation. This module is the only place that knows about NEDAS's calling
convention (one array per named state variable rather than Core's own
stacked (n,ny,nx) per-field arrays).

Layer naming: free-atmosphere layers use their numeric index '0'..'nz-1'
(top to bottom), the boundary layer always uses 'b'. For nz=2 this is a
rename from the earlier (pre-generalization) '1'/'3'/'b' convention, which
was tied to the paper's own fixed 3-layer sigma numbering (sigma1, sigma3)
that doesn't generalize to arbitrary nz -- e.g. there's no natural "layer 5"
under that scheme. See vort3d.md dev log.
"""
import numpy as np
from .core import Core


def layer_names(nz):
    """['0','1',...,'nz-1','b'] -- free-atmosphere layers top-to-bottom,
    then the boundary layer."""
    return [str(k) for k in range(nz)] + ['b']


def pack_state(core: Core) -> dict:
    """Core's stacked (n,ny,nx) arrays -> a flat dict of per-layer 2D
    arrays, keyed by native variable name (e.g. 'u0','theta1','qb','pstar')."""
    names = layer_names(core.nz)
    state = {'pstar': core.pstar.copy()}
    for k, name in enumerate(names):
        state[f'u{name}'] = core.u[k].copy()
        state[f'v{name}'] = core.v[k].copy()
        state[f'theta{name}'] = core.theta[k].copy()
        state[f'q{name}'] = core.q[k].copy()
    return state


def unpack_state(state: dict, core: Core) -> None:
    """Load a flat state dict (as produced by pack_state) into a Core
    instance's prognostic arrays, in place."""
    names = layer_names(core.nz)
    core.pstar = state['pstar'].copy()
    for k, name in enumerate(names):
        core.u[k] = state[f'u{name}'].copy()
        core.v[k] = state[f'v{name}'].copy()
        core.theta[k] = state[f'theta{name}'].copy()
        core.q[k] = state[f'q{name}'].copy()


def initial_condition(nx, ny, dx, nz=2, beta=0.0, moist=True, convection_scheme='ooyama',
                       Vbg=0.0, Vslope=-3, bg_seed=None, Vmax=15.0, Rmw=120.0e3,
                       vortex_x0=0.0, vortex_y0=-700.0e3, u_bkg=0.0, v_bkg=0.0,
                       f0=2*7.292e-5*np.sin(np.deg2rad(20.)),
                       theta_offset=0.0, q_offset=0.0) -> dict:
    """Generate a vort3d initial condition (Smith et al. 1990 vortex +
    Jordan 1957 sounding, per ZSU2001) -- reuses Core's own __init__,
    which already does this, then flattens the result. Vmax/Rmw set the
    initial vortex's peak tangential wind / radius of maximum wind
    (smith_vortex's own defaults: 15.0 m/s, 120e3 m). vortex_x0/vortex_y0
    set the vortex center relative to the domain center (default: displaced
    toward the southern boundary, see Core's own docstring) -- pass
    different values per ensemble member to perturb the initial position.
    u_bkg/v_bkg add a uniform (spatially-constant) steering flow on top of
    Vbg's turbulent background flow (see Core's own docstring for why this
    is a distinct mechanism). f0 is the reference Coriolis parameter (default:
    20N, matching the paper). theta_offset/q_offset add a domain-uniform
    perturbation to the boundary layer only (see Core's own docstring)."""
    core = Core(nx=nx, ny=ny, dx=dx, nz=nz, beta=beta, moist=moist,
                convection_scheme=convection_scheme, Vbg=Vbg, Vslope=Vslope, bg_seed=bg_seed,
                Vmax=Vmax, Rmw=Rmw, vortex_x0=vortex_x0, vortex_y0=vortex_y0,
                u_bkg=u_bkg, v_bkg=v_bkg, f0=f0,
                theta_offset=theta_offset, q_offset=q_offset)
    return pack_state(core)


def advance_time(state: dict, nx, ny, dx, nz, beta, moist, convection_scheme, dt, duration_h,
                  dt_reduction_factor=0.5, max_dt_retries=3, min_dt=None) -> dict:
    """
    Advance the given state forward by duration_h hours using Core's AB3
    sigma-coordinate dynamical core (+ surface fluxes/radiative
    cooling/condensation/convective closure if moist=True).

    A fresh Core is constructed to get its grid/constants machinery; its
    own auto-generated initial condition is immediately overwritten by the
    given state (so this does NOT reset to a fresh vortex -- state comes in
    fully determined by the caller). The Adams-Bashforth tendency history
    (`_prev_tendencies`) restarts cold every call, same as vort2d's RK4
    approach carries no memory across forecast segments -- a documented
    simplification (loses one AB-order of accuracy for the first couple of
    steps after each restart, negligible for the paper's dt=15s).

    Adaptive dt: if the model blows up (NaN) at `dt`, retry this SAME duration_h
    segment from the SAME starting `state` with dt reduced by `dt_reduction_factor`, up to
    `max_dt_retries` times, floored at `min_dt` (default: dt * dt_reduction_factor**max_dt_retries).
    """
    if min_dt is None:
        min_dt = dt * (dt_reduction_factor ** max_dt_retries)

    attempt_dt = dt
    while True:
        core = Core(nx=nx, ny=ny, dx=dx, nz=nz, beta=beta, moist=moist,
                    convection_scheme=convection_scheme)
        unpack_state(state, core)
        nsteps = int(duration_h * 3600 / attempt_dt)
        blew_up = False
        for _ in range(nsteps):
            core.step(attempt_dt)
            if np.any(np.isnan(core.u)) or np.any(np.isnan(core.pstar)):
                blew_up = True
                break
        if not blew_up:
            return pack_state(core)
        next_dt = attempt_dt * dt_reduction_factor
        if next_dt < min_dt - 1e-9:
            raise RuntimeError('vort3d.util.advance_time: NaN detected in model run'
                                + ('' if attempt_dt == dt else f' (also retried down to dt={attempt_dt:g}s)'))
        attempt_dt = next_dt
