"""
Adapts the standalone zsu2001_dry_core.Core prototype (sigma-coordinate
primitive-equation dynamical core + optional moist physics/Ooyama
convective closure, Zhu, Smith & Ulrich 2001) to NEDAS's plain-dict state
representation.

Core itself stays a self-contained, NEDAS-independent module (core.py,
unmodified) -- it's easier to keep validating/extending against the paper in
isolation. This module is the only place that knows about NEDAS's calling
convention (one array per named state variable rather than Core's own
stacked (3,ny,nx) per-field arrays).
"""
import numpy as np
from .core import Core

# maps Core's internal layer index (0,1,2) to the suffix used in NEDAS's
# per-layer variable names (layer1=upper troposphere, layer3=lower/mid
# troposphere, layerb=boundary layer -- see the paper's Fig. 1)
LAYER_NAMES = ['1', '3', 'b']


def pack_state(core: Core) -> dict:
    """Core's stacked (3,ny,nx) arrays -> a flat dict of per-layer 2D
    arrays, keyed by native variable name (e.g. 'u1','theta3','qb','pstar')."""
    state = {'pstar': core.pstar.copy()}
    for k, name in enumerate(LAYER_NAMES):
        state[f'u{name}'] = core.u[k].copy()
        state[f'v{name}'] = core.v[k].copy()
        state[f'theta{name}'] = core.theta[k].copy()
        state[f'q{name}'] = core.q[k].copy()
    return state


def unpack_state(state: dict, core: Core) -> None:
    """Load a flat state dict (as produced by pack_state) into a Core
    instance's prognostic arrays, in place."""
    core.pstar = state['pstar'].copy()
    for k, name in enumerate(LAYER_NAMES):
        core.u[k] = state[f'u{name}'].copy()
        core.v[k] = state[f'v{name}'].copy()
        core.theta[k] = state[f'theta{name}'].copy()
        core.q[k] = state[f'q{name}'].copy()


def initial_condition(nx, ny, dx, beta=0.0, moist=True, Vbg=0.0, Vslope=-3, bg_seed=None) -> dict:
    """Generate a vort3d initial condition (Smith et al. 1990 vortex +
    Jordan 1957 sounding, per ZSU2001) -- reuses Core's own __init__,
    which already does this, then flattens the result."""
    core = Core(nx=nx, ny=ny, dx=dx, beta=beta, moist=moist,
                Vbg=Vbg, Vslope=Vslope, bg_seed=bg_seed)
    return pack_state(core)


def advance_time(state: dict, nx, ny, dx, beta, moist, dt, duration_h) -> dict:
    """
    Advance the given state forward by duration_h hours using Core's AB3
    sigma-coordinate dynamical core (+ surface fluxes/radiative
    cooling/condensation/Ooyama convective closure if moist=True).

    A fresh Core is constructed to get its grid/constants machinery; its
    own auto-generated initial condition is immediately overwritten by the
    given state (so this does NOT reset to a fresh vortex -- state comes in
    fully determined by the caller). The Adams-Bashforth tendency history
    (`_prev_tendencies`) restarts cold every call, same as vort2d's RK4
    approach carries no memory across forecast segments -- a documented
    simplification (loses one AB-order of accuracy for the first couple of
    steps after each restart, negligible for the paper's dt=15s).
    """
    core = Core(nx=nx, ny=ny, dx=dx, beta=beta, moist=moist)
    unpack_state(state, core)
    nsteps = int(duration_h * 3600 / dt)
    for _ in range(nsteps):
        core.step(dt)
        if np.any(np.isnan(core.u)) or np.any(np.isnan(core.pstar)):
            raise RuntimeError('vort3d.util.advance_time: NaN detected in model run')
    return pack_state(core)
