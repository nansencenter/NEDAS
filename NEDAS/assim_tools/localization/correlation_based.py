import numpy as np
from NEDAS.utils.njit import njit


@njit
def gcv_localization_func(r, nens):
    """
    Correlation-based localization factor
    based on generalized cross-validation
    (cf. paper by I. Grooms)

    Parameters
    ----------
    r : np.ndarray
        Sample correlation coefficient.
    nens : int
        Ensemble size. Must be > 2.

    Returns
    -------
    np.ndarray
        Localization factor with the same shape as r.
    """
    factor = np.zeros(r.shape)

    r_flat = r.ravel()
    factor_flat = factor.ravel()

    for i in range(r_flat.size):
        r2 = r_flat[i] * r_flat[i]

        if (nens - 1) * r2 > 1.0:
            factor_flat[i] = (
                ((nens - 1) * r2 - 1.0)
                / ((nens - 2) * r2)
            )

    return factor

@njit
def prior_optimal_localization_func(r, nens):
    """
    Correlation-based localization factor
    based on Menetrier et al. (2015); see also
    Morzfeld & Hodyss (2023), equation (12).

    Parameters
    ----------
    r : np.ndarray
        Sample correlation coefficient.
    nens : int
        Ensemble size. Must be > 2.

    Returns
    -------
    np.ndarray
        Localization factor with the same shape as r.
    """
    r2 = r**2
    factor = np.zeros(r.shape)

    factor = ((nens - 1) * r2) / \
             (1. + nens * r2)

    return factor