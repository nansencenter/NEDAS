# ensemble verification metrics: scores that need the full ensemble
# distribution, not just its mean (for metrics on the ensemble mean vs a
# truth field -- e.g. RMSE, pattern correlation -- see spatial.py instead)
import numpy as np


def spread(fld_ens):
    """
    Ensemble spread: sqrt of the spatial mean ensemble variance.

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members

    Returns:
        float: ensemble spread (same units as fld_ens)
    """
    return float(np.sqrt(np.nanmean(np.var(fld_ens, axis=0, ddof=1))))


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
