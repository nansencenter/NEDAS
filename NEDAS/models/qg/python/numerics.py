"""
Numerical routines: tridiagonal solvers, Robert-filtered leapfrog timestepper,
random forcing, and ring-integral diagnostics.

Translates Fortran numerics_lib.f90.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Vectorised tridiagonal solver (Thomas algorithm)
# ---------------------------------------------------------------------------

def tridiag_vec(f_in, sub, diag, sup):
    """Solve a tridiagonal system for many RHS vectors simultaneously.

    Solves  A * x = f  where A has:
      sub  : sub-diagonal, shape (nz-1,)  or  (nmask, nz-1)
      diag : diagonal,     shape (nz,)    or  (nmask, nz)
      sup  : super-diag,   shape (nz-1,)  or  (nmask, nz-1)
      f_in : RHS,          shape (nmask, nz)

    Returns x of shape (nmask, nz).

    When sub/diag/sup are 1-D (shared across wavenumbers), they are broadcast.
    When diag is 2-D the diagonal differs per wavenumber (PV inversion case).

    Follows the Thomas algorithm from Fortran tridiag in numerics_lib.f90.
    """
    nmask, nv = f_in.shape

    sub_  = np.broadcast_to(sub,  (nmask, nv - 1)).copy()
    diag_ = np.broadcast_to(diag, (nmask, nv)).copy()
    sup_  = np.broadcast_to(sup,  (nmask, nv - 1)).copy()
    f     = f_in.copy()

    gamma = np.zeros_like(f)

    bet        = diag_[:, 0].copy()
    x          = np.zeros_like(f)
    x[:, 0]    = f[:, 0] / bet

    for n in range(1, nv):
        gamma[:, n] = sup_[:, n - 1] / bet
        bet          = diag_[:, n] - sub_[:, n - 1] * gamma[:, n]
        x[:, n]      = (f[:, n] - sub_[:, n - 1] * x[:, n - 1]) / bet

    for n in range(nv - 2, -1, -1):
        x[:, n] -= gamma[:, n + 1] * x[:, n + 1]

    return x


def tridiag_cyc_vec(f_in, sub, diag, sup, corner_lo, corner_hi):
    """Cyclic tridiagonal solver (periodic vertical BC).

    corner_lo : sub-diagonal wrap value  op(1, -1)  -> mat(1, nz)
    corner_hi : super-diagonal wrap value op(nz, 1) -> mat(nz, 1)
    Shape conventions same as tridiag_vec.

    Follows Fortran tridiag_cyc in numerics_lib.f90 (Sherman-Morrison).
    """
    nmask, nv = f_in.shape

    alpha  = corner_hi
    beta   = corner_lo
    gamma_val = -diag[0] if np.ndim(diag) == 1 else -diag[:, 0]

    diag_mod = diag.copy() if np.ndim(diag) > 1 else np.tile(diag, (nmask, 1))
    diag_mod[:, 0]    -= gamma_val
    diag_mod[:, nv-1] -= alpha * beta / gamma_val

    x = tridiag_vec(f_in, sub, diag_mod, sup)

    u      = np.zeros_like(f_in)
    u[:, 0]    = gamma_val
    u[:, nv-1] = alpha
    z = tridiag_vec(u, sub, diag_mod, sup)

    factr = (x[:, 0] + beta * x[:, nv-1] / gamma_val) / \
            (1.0 + z[:, 0] + beta * z[:, nv-1] / gamma_val)

    return x - factr[:, np.newaxis] * z


# ---------------------------------------------------------------------------
# Leapfrog timestepper with Robert filter
# ---------------------------------------------------------------------------

def march(f_now, f_o, rhs, dt, rob, calls):
    """Advance prognostic field one step using leapfrog / Robert filter.

    calls=0 : Euler half-step (initialisation)
    calls=1 : first leapfrog
    calls>=2 : leapfrog + Robert filter on f_o

    Returns (f_new, f_o_new, calls_new).

    Arrays can have any shape; operates element-wise.
    Translates Fortran March2/3 from numerics_lib.f90.
    """
    if calls == 0:
        f_new = f_now + 0.5 * dt * rhs
        f_o_new = f_now.copy()
        calls_new = 1
    elif calls == 1:
        f_new = f_o + dt * rhs
        f_o_new = f_o
        calls_new = 2
    else:
        f_new = f_o + 2.0 * dt * rhs
        f_o_new = (1.0 - 2.0 * rob) * f_now + rob * (f_o + f_new)
        calls_new = 3

    return f_new, f_o_new, calls_new


# ---------------------------------------------------------------------------
# Portable pseudo-random number generator
# ---------------------------------------------------------------------------

_RAN_IFF = 0
_RAN_IY  = 0
_RAN_IR  = np.zeros(97, dtype=np.int64)
_RAN_IDUM = -7

_RAN_M  = 714025
_RAN_IA = 1366
_RAN_IC = 150889
_RAN_RM = 1.4005112e-6


def ran_reset(idum):
    """Seed the portable RNG (negative idum seeds a fresh sequence)."""
    global _RAN_IFF, _RAN_IY, _RAN_IR, _RAN_IDUM
    _RAN_IDUM = int(idum)
    _RAN_IFF = 0   # force re-initialisation on next call to ran()


def ran(nx, ny):
    """Return (nx, ny) pseudo-random floats in [0, 1).

    Replicates the Fortran Ran function in numerics_lib.f90 exactly so that
    forcing sequences can be reproduced across languages.
    """
    global _RAN_IFF, _RAN_IY, _RAN_IR, _RAN_IDUM

    M  = _RAN_M
    IA = _RAN_IA
    IC = _RAN_IC
    RM = _RAN_RM
    irsize = 97

    if _RAN_IDUM < 0 or _RAN_IFF == 0:
        _RAN_IFF = 1
        _RAN_IDUM = int(np.mod(IC - _RAN_IDUM, M))
        for j in range(irsize):
            _RAN_IDUM = int(np.mod(IA * _RAN_IDUM + IC, M))
            _RAN_IR[j] = _RAN_IDUM
        _RAN_IDUM = int(np.mod(IA * _RAN_IDUM + IC, M))
        _RAN_IY   = _RAN_IDUM

    result = np.empty((nx, ny), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            j = 1 + (irsize * _RAN_IY) // M
            j = max(0, min(irsize - 1, j - 1))   # 0-based, clamp
            _RAN_IY    = int(_RAN_IR[j])
            result[x, y] = _RAN_IY * RM
            _RAN_IDUM  = int(np.mod(IA * _RAN_IDUM + IC, M))
            _RAN_IR[j] = _RAN_IDUM

    return result


# ---------------------------------------------------------------------------
# Ring integral (for spectral diagnostics)
# ---------------------------------------------------------------------------

def ring_integral(f, kxv, kyv, nf):
    """Integrate field f over rings of constant |k|.

    f    : (..., nky, nkx) or (nky, nkx)
    Returns (..., nf) or (nf,) ring-averaged values.
    """
    kxv_i = np.round(kxv).astype(int)
    kyv_i = np.round(kyv).astype(int)
    radius = np.floor(
        np.sqrt(kxv_i[np.newaxis, :]**2 + kyv_i[:, np.newaxis]**2) + 0.5
    ).astype(int)

    batch = f.shape[:-2]
    result = np.zeros(batch + (nf,), dtype=f.dtype)
    for rad in range(1, nf + 1):
        mask = radius == rad
        if batch:
            result[..., rad - 1] = np.sum(f[..., mask], axis=-1)
        else:
            result[..., rad - 1] = np.sum(f[mask])
    return result
