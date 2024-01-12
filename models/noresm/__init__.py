import numpy as np
import glob
from pyproj import Proj
from grid import Grid
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator

resolution = 1 ##degree

##reference lon/lat data at 2-degree resolution
with Dataset(__path__[0]+'/lonlat.nc') as f:
    lon_ = f['lon'].data
    lat_ = f['lat'].data

##project to laea temporarily to deal with tripolar geometry
# help_proj = Proj("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0")

##output to lonlat grid with resolution
lon, lat = np.meshgrid(np.arange(-180, 180+resolution, resolution),
                       np.arange(-90, 90+resolution, resolution))

# def proj(x, y, inverse=False):
#     x = np.atleast_1d(x).astype(float)
#     y = np.atleast_1d(y).astype(float)

#     if inverse:
#         return xy2lon(x, y), xy2lat(x, y)

#     else:
#         x = np.mod(x + 110, 360) - 110
#         return lonlat2x(x, y), lonlat2y(x, y)


# def read_grid(path, **kwargs):
    # x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    # return Grid(proj, x, y, cyclic_dim='x')

