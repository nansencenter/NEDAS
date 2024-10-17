import numpy as np
from pyproj import Geod
from grid import Grid

from .confmap import ConformalMapping

def get_topaz_grid(grid_info_file):
    cm = ConformalMapping.init_from_file(grid_info_file)
    nx = cm._ires
    ny = cm._jres

    ##proj function that mimic what pyproj.Proj object does to convert x,y to lon,lat
    def proj(x, y, inverse=False):
        if not inverse:
            i, j = cm.ll2gind(y, x)
            xo = (i-1)*dx
            yo = (j-1)*dy
        else:
            i = np.atleast_1d(x/dx + 1)
            j = np.atleast_1d(y/dy + 1)
            yo, xo = cm.gind2ll(i, j)
        if xo.size == 1:
            return xo.item(), yo.item()
        return xo, yo

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    lat, lon = cm.gind2ll(ii+1., jj+1.)

    ##find grid resolution
    geod = Geod(ellps='sphere')
    _,_,dist_x = geod.inv(lon[:,1:], lat[:,1:], lon[:,:-1], lat[:,:-1])
    _,_,dist_y = geod.inv(lon[1:,:], lat[1:,:], lon[:-1,:], lat[:-1,:])
    dx = np.median(dist_x)
    dy = np.median(dist_y)

    ##the coordinates in topaz model native grid
    x = ii*dx
    y = jj*dy

    return Grid(proj, x, y)

##some variables will need (de)staggering in topaz grid:
##---                *--*--*
##---                |  |  |
##--- stencil:       u--p--*
##---                |  |  |
##---                q--v--*
##these two functions are uniq to topaz

def stagger(dat, vtype):
    ##stagger u,v for C-grid configuration
    dat_stag = dat.copy()
    if vtype == 'u':
        dat_stag[:, 1:] = 0.5*(dat[:, :-1] + dat[:, 1:])
        dat_stag[:, 0] = 3*dat[:, 1] - 3*dat[:, 2] + dat[:, 3]
    elif vtype == 'v':
        dat_stag[1:, :] = 0.5*(dat[:-1, :] + dat[1:, :])
        dat_stag[0, :] = 3*dat[1, :] - 3*dat[2, :] + dat[3, :]
    return dat_stag

def destagger(dat_stag, vtype):
    ##destagger u,v from C-grid
    dat = dat_stag.copy()
    if vstag == 'u':
        dat[:, :-1] = 0.5*(dat_stag[:, :-1] + dat_stag[:, 1:])
        dat[:, -1] = 3*dat_stag[:, -2] - 3*dat_stag[:, -3] + dat_stag[:, -4]
    elif vstag == 'v':
        dat[:-1, :] = 0.5*(dat_stag[:-1, :] + dat_stag[1:, :])
        dat[-1, :] = 3*dat_stag[-2, :] - 3*dat_stag[-3, :] + dat_stag[-4, :]
    return dat

