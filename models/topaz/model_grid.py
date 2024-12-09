import numpy as np
from grid import Grid

from .confmap import ConformalMapping

def get_topaz_grid(grid_info_file):
    proj = ConformalMapping.init_from_file(grid_info_file)

    ##the coordinates in topaz model native grid
    ii, jj = np.meshgrid(np.arange(proj._ires), np.arange(proj._jres))
    x = ii * proj._dx
    y = jj * proj._dy

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
    if vtype == 'u':
        dat[:, :-1] = 0.5*(dat_stag[:, :-1] + dat_stag[:, 1:])
        dat[:, -1] = 3*dat_stag[:, -2] - 3*dat_stag[:, -3] + dat_stag[:, -4]
    elif vtype == 'v':
        dat[:-1, :] = 0.5*(dat_stag[:-1, :] + dat_stag[1:, :])
        dat[-1, :] = 3*dat_stag[-2, :] - 3*dat_stag[-3, :] + dat_stag[-4, :]
    return dat

