###util func for diagnostics
import numpy as np
from fft_lib import fft2, get_wn

def rmse(fld, fld_tr):
    return


def pattern_corr(fld, fld_tr):
    return


##some spectral diagnostics
def pwrspec2d(fld):
    ##horizontal 2d power spectrum p(k2d), on a regular Cartesian grid
    ##for very large grid, the map factors will cause slight errors in grid spacings
    ##but okay for relatively small grids. Of course for global analysis one shall use
    ##the spherical harmonics instead
    ##  input fld[..., ny, nx]
    ##  return wn[0:nup], pwr[..., 0:nup], nup is the max wavenumber given ny,nx
    ny, nx = fld.shape[-2:]
    kx, ky = get_wn(fld)
    nup = int(max(kx.max(), ky.max()))

    ##2d total wavenumber
    k2d = np.hypot(kx, ky)

    ##2d fft of fld, convert to power (variance)
    P = (np.abs(fft2(fld))/ny/nx)**2

    ##sum all kx,ky points with same k2d
    wn = np.arange(0., nup)
    pwr = np.zeros(fld.shape[:-2] + (nup,))
    for k in range(nup):
        pwr[..., k] = np.mean(P[np.where(np.ceil(k2d)==k)])
        ##we show mean pwr spectrum, it will be more intuitive to
        ##see the white noise spectrum as a flat line;
        ##
        ##however, in turbulence or atmospheric sciences, it is typical to
        ##show pwr = sum(P[k2d==w]); where the white noise has +1 slope
        ##and typical synoptic scale flows have a well-known -3 slope

    return wn, pwr


###some ensemble metrics
def crps(fld_ens, fld_tr):
    return

