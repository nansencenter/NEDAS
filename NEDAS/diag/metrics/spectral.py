# #util func for diagnostics
import numpy as np
from NEDAS.utils.fft_lib import fft2, get_wn

def rmse(fld_ens, fld_tr):
    """
    RMSE of the ensemble mean against the truth field.

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members
        fld_tr:  np.ndarray, shape (...)        — truth / reference field

    Returns:
        float: spatially averaged RMSE
    """
    ens_mean = np.mean(fld_ens, axis=0)
    return float(np.sqrt(np.nanmean((ens_mean - fld_tr) ** 2)))


def spread(fld_ens):
    """
    Ensemble spread: sqrt of the spatial mean ensemble variance.

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members

    Returns:
        float: ensemble spread (same units as fld_ens)
    """
    return float(np.sqrt(np.nanmean(np.var(fld_ens, axis=0, ddof=1))))


def pattern_corr(fld_ens, fld_tr):
    """
    Anomaly pattern correlation between ensemble mean and truth.

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members
        fld_tr:  np.ndarray, shape (...)        — truth / reference field

    Returns:
        float: Pearson correlation coefficient in [-1, 1]
    """
    a = np.nanmean(fld_ens, axis=0).ravel()
    b = fld_tr.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask] - np.mean(a[mask])
    b = b[mask] - np.mean(b[mask])
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def crps(fld_ens, fld_tr):
    """
    Continuous Ranked Probability Score (CRPS) at each grid point.

    Uses the O(N log N) sorted-ensemble formula (Gneiting & Raftery 2007):
        CRPS = MAE(ens_mean, obs) - (1/2) * mean_spread

    which in the sorted-ensemble form (Ferro & Fricker 2012) is:
        CRPS = (1/N) * sum_i |x_(i) - y|
               - (1/N^2) * sum_i (2i - N + 1) * x_(i)

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members
        fld_tr:  np.ndarray, shape (...)        — truth / reference field

    Returns:
        np.ndarray, shape (...): pointwise CRPS (≥ 0, lower is better)
    """
    nens = fld_ens.shape[0]
    sorted_ens = np.sort(fld_ens, axis=0)
    # broadcast weights over all spatial dims
    w = (2 * np.arange(nens) - nens + 1).reshape((nens,) + (1,) * (fld_ens.ndim - 1))
    mae = np.mean(np.abs(fld_ens - fld_tr), axis=0)
    dispersion = np.sum(w * sorted_ens, axis=0) / nens ** 2
    return mae - dispersion


def mean_crps(fld_ens, fld_tr):
    """
    Spatially averaged CRPS.

    Returns:
        float
    """
    return float(np.nanmean(crps(fld_ens, fld_tr)))


def brier_score(fld_ens, fld_tr, threshold):
    """
    Brier score for the binary event (field > threshold).

    BS = mean( (P_ens(X > threshold) - 1{y > threshold})^2 )

    Args:
        fld_ens:   np.ndarray, shape (nens, ...) — ensemble members
        fld_tr:    np.ndarray, shape (...)        — truth / reference field
        threshold: float                          — event threshold (same units)

    Returns:
        float: Brier score in [0, 1] (lower is better)
    """
    p_ens = np.mean(fld_ens > threshold, axis=0).astype(float)
    event_obs = (fld_tr > threshold).astype(float)
    return float(np.nanmean((p_ens - event_obs) ** 2))


# some spectral diagnostics
def pwrspec2d(fld):
    """
    Horizontal 2D power spectrum p(k2d), on a regular Cartesian grid

    For very large grid, the map factors will cause slight errors in grid spacings
    but okay for relatively small grids. Of course for global analysis one shall use
    the spherical harmonics instead.

    Input:

    - fld: np.array, shape (..., ny, nx)
      n-dimensional input field, the last two dimensions are the horizontal directions (y,x)

    Returns:

    - wn: np.array, shape (nup,)
      Wavenumber in 2D, int(k2d), nup is the max wavenumber given ny,nx (whichever is larger)

    - pwr: np.array, shape (..., nup)
      The power spectrum, leading dimensions the same as fld but ny,nx replaced by nup.
    """
    ny, nx = fld.shape[-2:]
    kx, ky = get_wn(fld)
    nup = int(max(kx.max(), ky.max()))

    # 2d total wavenumber
    k2d = np.hypot(kx, ky)

    # 2d fft of fld, convert to power (variance)
    P = (np.abs(fft2(fld))/ny/nx)**2

    # sum all kx,ky points with same k2d
    wn = np.arange(0., nup)
    pwr = np.zeros(fld.shape[:-2] + (nup,))
    for k in range(nup):
        pwr[..., k] = np.mean(P[np.where(np.floor(k2d)==k)])
        # we show mean pwr spectrum, it will be more intuitive to
        # see the white noise spectrum as a flat line;
        #
        # however, in turbulence or atmospheric sciences, it is typical to
        # show pwr = sum(P[k2d==w]); where the white noise has +1 slope
        # and typical synoptic scale flows have a well-known -3 slope

    return wn, pwr
