import numpy as np
import pyproj
from .confmap import ConformalMapping
from grid import Grid

##topaz grid:
# ---                *--*--*
# ---                |  |  |
# --- stencil:       u--p--*
# ---                |  |  |
# ---                q--v--*
# --- NB: Python uses reversed indexing rel fortran

###parse grid.info and generate grid.Grid object
def get_grid(grid_info_file):
    cm = ConformalMapping.init_from_file(grid_info_file)
    nx = cm._ires
    ny = cm._jres

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    lat, lon = cm.gind2ll(ii+1., jj+1.)

    ##find grid resolution
    geod = pyproj.Geod(ellps='sphere')
    _,_,dist_x = geod.inv(lon[:,1:], lat[:,1:], lon[:,:-1], lat[:,:-1])
    _,_,dist_y = geod.inv(lon[1:,:], lat[1:,:], lon[:-1,:], lat[:-1,:])
    dx = np.median(dist_x)
    dy = np.median(dist_y)


    ##the coordinates in topaz model native grid
    x = ii*dx
    y = jj*dy

    ##proj function that mimic what pyproj.Proj object does
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

    return Grid(proj, x, y)
