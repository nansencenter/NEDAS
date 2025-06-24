import numpy as np
from NEDAS.utils.njit import njit

@njit
def gaspari_cohn_func(dist, roi):
    """
    Gaspari-Cohn localization function.

    Args:
        dist (np.ndarray): Distance between observation and state (being updated)
        roi (float or np.ndarray): The radius of influence, distance beyond which local_factor is tapered to 0

    Returns:
        np.ndarray: The localization factor, same shape as dist
    """
    shape = dist.shape
    dist = dist.flatten()
    lfactor = np.zeros(dist.shape)
    r = dist / (roi / 2)

    ind1 = np.where(r<1)
    loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
    lfactor[ind1] = loc1[ind1]

    ind2 = np.where(np.logical_and(r>=1, r<2))
    r[np.where(r==0)] = 1e-10  ##avoid divide by 0
    loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
    lfactor[ind2] = loc2[ind2]

    return lfactor.reshape(shape)

@njit
def step_func(dist, roi):
    """
    Step function for localization.
    """
    shape = dist.shape
    dist = dist.flatten()
    lfactor = np.zeros(dist.shape)
    lfactor[np.where(dist<=roi)] = 1.0
    return lfactor.reshape(shape)

@njit
def exponential_func(dist, roi):
    """
    Exponential decay localization function.
    """
    shape = dist.shape
    dist = dist.flatten()
    lfactor = np.exp(-dist/roi)
    return lfactor.reshape(shape)
