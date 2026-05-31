"""
Spectral grid setup and FFT transforms for the QG model.

Faithfully translates the Fortran transform_tools.f90 staggered-grid scheme.

Conventions
-----------
Spectral arrays:  shape (..., nky, nkx) where nkx=2*kmax+1, nky=kmax+1
Physical arrays:  shape (..., ny, nx)   where nx=ny=2*(kmax+1)

kx index:  kxv[j] = j - kmax  in [-kmax, kmax]
ky index:  kyv[j] = j          in [0, kmax]

Staggered-grid dealiasing
--------------------------
Following Vallis (c1991) / Smith (c1998), each spec->phys transform packs
the straight physical field into Re(output) and the half-grid-shifted
(staggered) field into Im(output).  Products are computed separately on
the two grids (ir_prod) to avoid aliasing, then transformed back.

FFT conventions (from fft_fftw3_1pe.f90)
-----------------------------------------
Fortran fft(x, dirn=-1)  ->  FFTW_BACKWARD, scale=1   -> np.fft.ifft2(x)*N²
Fortran fft(x, dirn=+1)  ->  FFTW_FORWARD,  scale=1/N² -> np.fft.fft2(x)/N²
"""

import numpy as np
try:
    import jax.numpy as jnp
    _JAX = True
except ImportError:
    jnp = np
    _JAX = False


# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

def setup_spectral_grid(kmax):
    """Return wavenumber grids and index arrays for spec<->phys transforms.

    Returns a dict with keys:
      kxv, kyv          1-D wavenumber vectors
      kx_, ky_, ksqd_   2-D wavenumber grids (nky, nkx)
      nx, ny, nkx, nky
      kxup, kyup        indices into the nx×ny physical array (positive side)
      kxdn, kydn        indices for the conjugate (negative) side
      exx               phase-shift for staggered grid, shape (nky, nkx)
      sgn               (-1)^(m+n) sign matrix, shape (nx, ny)
      filter_mask       base de-aliasing mask (1 inside, 0 outside), (nky, nkx)
      lin2kx, lin2ky    packed index lists for the non-zero mask region
      nmask             number of active (kx, ky) points
    """
    nx = 2 * (kmax + 1)
    ny = nx
    nkx = 2 * kmax + 1
    nky = kmax + 1

    kxv = np.arange(-kmax, kmax + 1, dtype=np.float64)   # (nkx,)
    kyv = np.arange(0,     kmax + 1, dtype=np.float64)   # (nky,)

    # 2-D wavenumber grids: shape (nky, nkx)
    kx_ = np.broadcast_to(kxv[np.newaxis, :], (nky, nkx)).copy()
    ky_ = np.broadcast_to(kyv[:, np.newaxis], (nky, nkx)).copy()
    ksqd_ = kx_**2 + ky_**2
    ksqd_[0, kmax] = 0.1   # avoid division by zero at (kx=0, ky=0)

    # Index mapping from spectral half-plane into the full nx×ny FFT array.
    # Translates Fortran 1-based kxup = kxv + kmax + 2 to 0-based:
    kxv_i = np.arange(-kmax, kmax + 1, dtype=int)
    kyv_i = np.arange(0,     kmax + 1, dtype=int)
    kxup = kxv_i + kmax + 1          # in [1, nkx]
    kyup = kyv_i + kmax + 1          # in [kmax+1, nx-1]
    kxdn = -kxv_i + kmax + 1         # in [1, nkx]
    kydn = -kyv_i + kmax + 1         # in [1, kmax+1]

    # Phase shift for staggered grid: exx[ky_idx, kx_idx] = exp(i*pi*(kx+ky)/nx)
    exx = np.exp(1j * np.pi * (kxv[np.newaxis, :] + kyv[:, np.newaxis]) / nx)

    # Checkerboard sign matrix: sgn[m, n] = (-1)^(m+n)
    m = np.arange(nx)
    n = np.arange(ny)
    sgn = ((-1) ** (m[:, np.newaxis] + n[np.newaxis, :])).astype(np.float64)

    # De-aliasing mask (isotropic, Orszag criterion from qg_arrays.f90):
    #   filter = 0 where ksqd_ >= (8/9)*(kmax+1)^2
    #   Also zero out kx <= 0, ky = 0  (those come from conjugate symmetry)
    filter_mask = np.ones((nky, nkx), dtype=np.float64)
    filter_mask[ksqd_ >= (8.0 / 9.0) * (kmax + 1) ** 2] = 0.0
    filter_mask[0, :kmax + 1] = 0.0   # kx <= 0, ky = 0

    # Packed lists of active wavenumber indices
    flat_mask = filter_mask.ravel()
    active = np.where(flat_mask > 0)[0]
    lin2ky = (active // nkx).astype(np.int32)
    lin2kx = (active %  nkx).astype(np.int32)
    nmask = len(active)

    return dict(
        kmax=kmax, nx=nx, ny=ny, nkx=nkx, nky=nky,
        kxv=kxv, kyv=kyv, kx_=kx_, ky_=ky_, ksqd_=ksqd_,
        kxup=kxup, kyup=kyup, kxdn=kxdn, kydn=kydn,
        exx=exx, sgn=sgn,
        filter_mask=filter_mask,
        lin2kx=lin2kx, lin2ky=lin2ky, nmask=nmask,
    )


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------

def make_filter(g, filter_type='hyperviscous', filter_exp=8.0, k_cut=None,
                dealiasing='isotropic', filt_tune=1.0):
    """Build the full spectral filter (de-aliasing + small-scale damping).

    Parameters match the Fortran Init_filter in qg_init_tools.f90.
    Returns real array of shape (nky, nkx).
    """
    kmax = g['kmax']
    nx   = g['nx']
    ksqd_ = g['ksqd_']
    kx_   = g['kx_']
    ky_   = g['ky_']

    filt = np.ones((g['nky'], g['nkx']), dtype=np.float64)
    filt[0, :kmax + 1] = 0.0   # conjugate-symmetry side

    if dealiasing == 'isotropic':
        filt[ksqd_ >= (8.0 / 9.0) * (kmax + 1) ** 2] = 0.0
        kmax_da = np.sqrt(8.0 / 9.0) * (kmax + 1)
    elif dealiasing == 'orszag':
        filt[(np.abs(kx_) + np.abs(ky_)) >= (4.0 / 3.0) * (kmax + 1)] = 0.0
        kmax_da = np.sqrt(8.0 / 9.0) * (kmax + 1)
    else:   # 'none'
        kmax_da = kmax

    if filter_type == 'hyperviscous':
        mask = filt > 0.0
        filt[mask] = 1.0 / (
            1.0 + filt_tune * (4 * np.pi / nx) * (ksqd_[mask] / kmax_da**2) ** filter_exp
        )
    elif filter_type == 'exp_cutoff':
        if k_cut is None:
            raise ValueError('exp_cutoff filter requires k_cut')
        mask = filt > 0.0
        filt[mask] = np.exp(-((np.sqrt(ksqd_[mask]) - k_cut) / (kmax_da - k_cut)) ** filter_exp)
        filt[ksqd_ < k_cut**2] = 1.0
    # 'none': keep filt = 1 inside de-aliasing region

    return filt


# ---------------------------------------------------------------------------
# Spectral <-> physical transforms
# ---------------------------------------------------------------------------

def spec2grid_cc(wf, g):
    """Spectral to physical transform with staggered-grid packing.

    wf : complex array, shape (..., nky, nkx)
    Returns complex array shape (..., nx, ny):
      real part  = physical field on straight grid
      imag part  = physical field on staggered (half-shifted) grid

    Translates Fortran Spec2grid_cc2/3 from transform_tools.f90.
    """
    kmax = g['kmax']
    nx   = g['nx']
    ny   = g['ny']
    nkx  = g['nkx']
    nky  = g['nky']
    kxup = g['kxup']
    kyup = g['kyup']
    kxdn = g['kxdn']
    kydn = g['kydn']
    exx  = g['exx']
    sgn  = g['sgn']
    xp = np if not _JAX else jnp

    batch = wf.shape[:-2]
    wavefield = wf.copy() if not _JAX else wf

    # Enforce Hermitian symmetry along ky=0 for kx < 0.
    # Fortran: wavefield(1:kmax+1, 1) = conjg(wavefield(nkx:kmax+1:-1, 1))
    # Python (0-based): indices 0..kmax set from indices 2*kmax..kmax (reversed)
    if _JAX:
        ky0 = wavefield[..., 0, :]
        ky0 = ky0.at[..., :kmax + 1].set(xp.conj(ky0[..., 2*kmax:kmax-1:-1]))
        wavefield = wavefield.at[..., 0, :].set(ky0)
    else:
        wavefield[..., 0, :kmax + 1] = np.conj(wavefield[..., 0, 2*kmax:kmax-1:-1])

    # Build sparse nx×ny physical-space array
    phys_shape = batch + (nx, ny)
    physfield = xp.zeros(phys_shape, dtype=np.complex128)

    # Meshgrid of insertion indices
    ix_up, iy_up = np.ix_(kxup, kyup) if not _JAX else (
        xp.ix_(kxup, kyup))
    ix_dn, iy_dn = np.ix_(kxdn, kydn) if not _JAX else (
        xp.ix_(kxdn, kydn))

    # exx has shape (nky, nkx); wf has shape (..., nky, nkx)
    # We need to swap to (nkx, nky) for the insertion
    wf_t   = wavefield.swapaxes(-2, -1)    # (..., nkx, nky)
    exx_t  = exx.T                          # (nkx, nky)

    plus_  = wf_t + 1j * (exx_t * wf_t)
    minus_ = xp.conj(wf_t - 1j * (exx_t * wf_t))

    if _JAX:
        physfield = physfield.at[..., ix_up, iy_up].set(plus_)
        physfield = physfield.at[..., ix_dn, iy_dn].set(minus_)
    else:
        physfield[..., ix_up, iy_up] = plus_
        physfield[..., ix_dn, iy_dn] = minus_

    # FFT: Fortran fft(x, -1) = FFTW_BACKWARD (no normalization)
    #      = np.fft.ifft2(x) * N²
    N2 = nx * ny
    physfield = xp.fft.ifft2(physfield, axes=(-2, -1)) * N2
    physfield = sgn * physfield

    return physfield


def grid2spec(pf, g):
    """Physical (staggered-packed complex) to spectral transform.

    pf : complex array, shape (..., nx, ny)
    Returns complex array shape (..., nky, nkx).

    Translates Fortran Grid2spec_cc2/3 from transform_tools.f90.
    """
    kxup = g['kxup']
    kyup = g['kyup']
    kxdn = g['kxdn']
    kydn = g['kydn']
    exx  = g['exx']
    sgn  = g['sgn']
    nx   = g['nx']
    ny   = g['ny']
    xp = np if not _JAX else jnp

    physfield = sgn * pf
    # Fortran fft(x, +1) = FFTW_FORWARD / N²  = np.fft.fft2(x) / N²
    N2 = nx * ny
    physfield = xp.fft.fft2(physfield, axes=(-2, -1)) / N2

    # Extract at (kxup, kyup) and (kxdn, kydn)
    ix_up, iy_up = np.ix_(kxup, kyup)
    ix_dn, iy_dn = np.ix_(kxdn, kydn)

    Pp = physfield[..., ix_up, iy_up]   # (..., nkx, nky)
    Pm = physfield[..., ix_dn, iy_dn]   # (..., nkx, nky)

    exx_t = exx.T   # (nkx, nky)

    wavefield = (
        np.real(Pp + Pm)
        + 1j * np.imag(Pp - Pm)
        + (np.imag(Pp + Pm) + 1j * np.real(-Pp + Pm)) * np.conj(exx_t)
    )
    wavefield = 0.25 * wavefield

    # Swap back to (..., nky, nkx)
    return wavefield.swapaxes(-2, -1)


def spec2grid(wf, g):
    """Spectral to physical (real output), discards staggered grid.

    wf : complex (..., nky, nkx)
    Returns real (..., nx, ny).
    """
    return np.real(spec2grid_cc(wf, g))


# ---------------------------------------------------------------------------
# Physical-space products (de-aliased via staggered grid)
# ---------------------------------------------------------------------------

def ir_prod(f, g_field):
    """Product of two staggered-packed physical fields.

    f, g_field : complex (..., nx, ny)  with Re=straight, Im=staggered
    Returns complex (..., nx, ny) with the same packing.
    """
    return (np.real(f) * np.real(g_field)
            + 1j * np.imag(f) * np.imag(g_field))


def ir_pwr(f, pwr):
    """Raise staggered-packed field to power pwr (supports 2 and 0.5)."""
    return (np.real(f) ** pwr + 1j * np.imag(f) ** pwr)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

def jacob(fk, gk, g):
    """Arakawa Jacobian J(f, g) in spectral space.

    fk, gk : complex (..., nky, nkx)
    Returns complex (..., nky, nkx).

    J(f,g) = df/dx * dg/dy - df/dy * dg/dx
    """
    kx_ = g['kx_']
    ky_ = g['ky_']

    dfx = spec2grid_cc(1j * kx_ * fk, g)
    dgy = spec2grid_cc(1j * ky_ * gk, g)
    dfy = spec2grid_cc(1j * ky_ * fk, g)
    dgx = spec2grid_cc(1j * kx_ * gk, g)

    return grid2spec(ir_prod(dfx, dgy) - ir_prod(dfy, dgx), g)
