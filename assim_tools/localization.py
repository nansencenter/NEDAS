import numpy as np
from numba import njit

###localization:
@njit
def local_factor(dist, roi, localize_type='GC'):
    """
    Localization factor based on distance and radius of influence (roi)

    Inputs:
    - dist: np.array
      Distance between observation and state (being updated)

    - roi: float or np.array same shape as dist
      The radius of influence, distance beyond which local_factor is tapered to 0

    Return:
    - lfactor: np.array
      The localization factor, same shape as dist
    """

    shape = dist.shape
    dist = dist.flatten()
    lfactor = np.zeros(dist.shape)

    if localize_type == 'GC': ##Gaspari-Cohn function (default)
        r = dist / (roi / 2)

        ind1 = np.where(r<1)
        loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
        lfactor[ind1] = loc1[ind1]

        ind2 = np.where(np.logical_and(r>=1, r<2))
        r[np.where(r==0)] = 1e-10  ##avoid divide by 0
        loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
        lfactor[ind2] = loc2[ind2]

    elif localize_type == 'step':  #step function from 1 to 0 at roi
        lfactor[np.where(dist<=roi)] = 1.0

    elif localize_type == 'exp':  ##exponential decay
        lfactor = np.exp(-dist/roi)

    else:
        print('Error: unknown localization function type: '+localize_type)
        raise ValueError

    return lfactor.reshape(shape)

