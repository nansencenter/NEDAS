"""
Stratification tools: tridiagonal operator, vertical modes, triple interactions.

Translates Fortran strat_tools.f90 and the relevant parts of qg_run_tools.f90.
"""

import numpy as np
from scipy import linalg


# ---------------------------------------------------------------------------
# Stratification operator (psiq)
# ---------------------------------------------------------------------------

def strat_params(dz, drho, F, Fe, surface_bc='rigid_lid'):
    """Build tridiagonal coefficients for the stretching operator.

    Returns op of shape (nv, 3) where op[:, 0] = sub-diagonal,
    op[:, 1] = diagonal, op[:, 2] = super-diagonal.
    For surface_bc='surf_buoy', nv = nz+1; otherwise nv = nz.

    Translates Fortran strat_params in strat_tools.f90.
    The Fortran stores op(1:nv, -1:1); we map -1->0, 0->1, 1->2.
    """
    nz = len(dz)
    nv = nz + 1 if surface_bc == 'surf_buoy' else nz
    # op indexed [z, offset] where offset 0=sub, 1=diag, 2=super
    op = np.zeros((nv, 3), dtype=np.float64)

    # Sub-diagonal: op(2:nz, -1)
    op[1:nz, 0] = F / (dz[1:nz] * drho[0:nz-1])
    # Super-diagonal: op(1:nz-1, 1)
    op[0:nz-1, 2] = F / (dz[0:nz-1] * drho[0:nz-1])

    if surface_bc == 'periodic':
        op[0, 0]    = op[0, 2]   # wrap-around sub
        op[nz-1, 2] = op[nz-1, 0]

    elif surface_bc == 'surf_buoy':
        # psi at surface is in a delta sheet at z=0
        op[0, 0] = F / (dz[0] * drho[0])   # psi(0) coupling
        op[0, 2] = -1.0 / dz[0]            # b equation super

    # Diagonal for interior layers: op(:, 1) = -sub - super
    op[0:nz, 1] = -op[0:nz, 0] - op[0:nz, 2]

    if surface_bc == 'surf_buoy':
        # Restore buoyancy-row diagonal (overwritten by the loop above)
        op[0, 1] = 1.0 / dz[0]
    # Free lower surface correction
    op[nz-1, 1] -= Fe / dz[nz-1]

    return op


def get_layer_depths(dz):
    """Layer centre depths (negative downward), like Fortran Get_z."""
    nz = len(dz)
    z = np.zeros(nz, dtype=np.float64)
    z[0] = -dz[0] / 2.0
    for n in range(1, nz):
        z[n] = z[n-1] - (dz[n-1] + dz[n]) / 2.0
    return z


def get_vmodes(dz, drho, F, Fe=0.0, surface_bc='rigid_lid'):
    """Compute vertical modes via eigendecomposition of the stretching operator.

    Returns (kz, vmode) where:
      kz    : (nz,) deformation wavenumbers (ascending)
      vmode : (nz, nz) orthonormal mode matrix (columns = modes)

    Translates Fortran Get_vmodes in strat_tools.f90.
    """
    nz = len(dz)

    if nz == 1:
        return np.array([0.0]), np.array([[1.0]])

    if nz == 2:
        # Analytic result for two-layer case
        if Fe != 0.0:
            alpha = Fe / F
            a = np.sqrt(1.0 + (alpha / 2.0) ** 2)
            b = alpha / 2.0
            kz = np.array([np.sqrt(F / dz[0] * (1 + b - a)),
                           np.sqrt(F / dz[0] * (1 + b + a))])
            vmode = np.array([[1.0, a + b], [-1.0, a - b]]).T
        else:
            kz = np.array([0.0, np.sqrt(F) / np.sqrt(dz[0] * (1 - dz[0]))])
            vmode = np.array([[1.0, 1.0],
                              [np.sqrt((1 - dz[0]) / dz[0]),
                               -np.sqrt(dz[0] / (1 - dz[0]))]]).T
        # Normalise with layer-thickness weighting
        for n in range(nz):
            norm = np.sqrt(np.dot(vmode[:, n] * dz, vmode[:, n]))
            vmode[:, n] /= norm
            if vmode[0, n] < 0:
                vmode[:, n] = -vmode[:, n]
        return kz, vmode

    # General case: eigensolve
    op = strat_params(dz, drho, F, Fe, surface_bc)

    # Build full matrix from tridiagonal coefficients
    mat = np.diag(op[0:nz, 1]) + np.diag(op[1:nz, 0], -1) + np.diag(op[0:nz-1, 2], 1)
    if surface_bc == 'periodic':
        mat[0, nz-1] = op[0, 0]
        mat[nz-1, 0] = op[nz-1, 2]

    evals, evecs = linalg.eigh(mat)

    kz = np.where(evals < 0, np.sqrt(-evals), 0.0)

    # Sort ascending by kz
    order = np.argsort(kz)
    kz    = kz[order]
    vmode = evecs[:, order]

    # Normalise
    for n in range(nz):
        norm = np.sqrt(np.dot(vmode[:, n] * dz, vmode[:, n]))
        vmode[:, n] /= norm
        if vmode[0, n] < 0:
            vmode[:, n] = -vmode[:, n]

    return kz, vmode


def get_psiq(dz, drho, F, Fe=0.0, surface_bc='rigid_lid'):
    """Return the psiq tridiagonal operator used in Get_pv / Invert_pv.

    psiq[z, offset] with offset in {-1, 0, 1} (Fortran convention).
    We return shape (nv, 3) mapping to offsets [-1, 0, 1].
    """
    return strat_params(dz, drho, F, Fe, surface_bc)


# ---------------------------------------------------------------------------
# Triple-interaction coefficients
# ---------------------------------------------------------------------------

def get_tripint(dz, rho, surface_bc='rigid_lid', hf=10):
    """Triple-interaction coefficients for vertical modes.

    Returns tripint of shape (nz, nz, nz).

    Translates Fortran Get_tripint in strat_tools.f90, using spline
    interpolation to a higher-resolution grid for de-aliasing.
    """
    from scipy.interpolate import CubicSpline

    nz = len(dz)

    if nz == 2:
        ti = np.zeros((2, 2, 2))
        ti[0, 0, 0] = 1.0
        ti[0, 1, 1] = 1.0
        ti[1, 0, 1] = 1.0
        ti[1, 1, 0] = 1.0
        ti[1, 1, 1] = (1.0 - 2.0 * dz[0]) / np.sqrt(dz[0] * (1.0 - dz[0]))
        return ti

    # High-resolution uniform grid
    nzh = hf * nz
    dzh = np.full(nzh, 1.0 / nzh)
    zh = get_layer_depths(dzh)          # high-res layer centres (negative)
    z  = get_layer_depths(dz)           # coarse layer centres

    # Spline-interpolate density to high-res grid
    cs   = CubicSpline(-z[::-1], rho[::-1])   # z is negative; spline over positive depths
    rhoh = cs(-zh)                             # evaluate at high-res centres

    drhoh = np.diff(rhoh)
    assert np.all(drhoh > 0), 'density not monotonically increasing in tripint'

    _, vmodeh = get_vmodes(dzh, drhoh, F=1.0, Fe=0.0, surface_bc=surface_bc)

    ti = np.zeros((nz, nz, nz), dtype=np.float64)
    for k in range(nz):
        for j in range(nz):
            for i_idx in range(nz):
                ti[i_idx, j, k] = np.sum(dzh * vmodeh[:, i_idx]
                                          * vmodeh[:, j] * vmodeh[:, k])
    return ti


# ---------------------------------------------------------------------------
# Layer <-> mode projections
# ---------------------------------------------------------------------------

def layer2mode(f, vmode, dz):
    """Project layered spectral field onto vertical modes.

    f     : (..., nz, nky, nkx)  layered field (spectral or physical)
    vmode : (nz, nz)
    dz    : (nz,)
    Returns (..., nz, nky, nkx)
    """
    # modal projection: fm[:, m] = sum_z dz[z] * vmode[z, m] * f[:, z]
    # Using einsum: fm[..., m, k, l] = sum_z dz[z] * vmode[z, m] * f[..., z, k, l]
    return np.einsum('zm,...zxy->...mxy', (vmode * dz[:, np.newaxis]).T, f)


def mode2layer(fm, vmode):
    """Project modal field onto layers.

    fm    : (..., nz, nky, nkx)
    vmode : (nz, nz)
    Returns (..., nz, nky, nkx)
    """
    return np.einsum('zm,...mxy->...zxy', vmode, fm)
