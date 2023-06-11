###util func for diagnostics
import numpy as np
from assim_tools.multiscale import fft2, get_wn, get_scale_comp

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
    nx = fld.shape[-1]
    ny = fld.shape[0]
    kx, ky = get_wn(fld)
    nupx = kx.max()
    nupy = ky.max()
    nup = int(max(nupx, nupy))

    ##2d total wavenumber
    k2d = np.sqrt((kx*(nup/nupx))**2 + (ky*(nup/nupy))**2)

    ##2d fft of fld, convert to power (variance)
    P = (np.abs(fft2(fld))/ny/nx)**2

    ##sum all kx,ky points with same k2d
    wn = np.arange(0., nup)
    pwr = np.zeros(fld.shape[:-2] + (nup,))
    for w in range(nup):
        pwr[..., w] = np.sum(P[np.where(np.ceil(k2d)==w)])
    return wn, pwr


###some ensemble metrics
def crps(fld_ens, fld_tr):
    return

