import numpy as np
from netCDF4 import Dataset
from grid_util import *

dx = 5 ##km, resolution of reference grid
th = 45 * np.pi/180 ##rotation angle
xcoord = np.arange(-2.5e6, 2.e6, dx*1000.)
ycoord = np.arange(-2.5e6, 2.e6, dx*1000.)
xi, yi = np.meshgrid(xcoord, ycoord)
x =  xi*np.cos(th) + yi*np.sin(th)
y = -xi*np.sin(th) + yi*np.cos(th)
RE = 6378273.
z = np.array(np.sqrt(RE**2 - x**2 - y**2))

plon, plat = xyz_to_lonlat(x, y, z)
nx, ny = x.shape
ptheta = get_theta(x, y)
x_corners = get_corners(x)
y_corners = get_corners(y)
z_corners = get_corners(z)

f2 = Dataset('reference_grid.nc', 'w', format="NETCDF4_CLASSIC")
f2.createDimension('n', size=4)
f2.createDimension('x', size=nx)
f2.createDimension('y', size=ny)
f2.createVariable('plat', float, ('x', 'y'))
f2.createVariable('plon', float, ('x', 'y'))
f2.createVariable('ptheta', float, ('x', 'y'))
f2.createVariable('x_corners', float, ('x', 'y', 'n'))
f2.createVariable('y_corners', float, ('x', 'y', 'n'))
f2.createVariable('z_corners', float, ('x', 'y', 'n'))
f2['plat'][:, :] = plat
f2['plon'][:, :] = plon
f2['ptheta'][:, :] = ptheta
f2['x_corners'][:, :, :] = x_corners
f2['y_corners'][:, :, :] = y_corners
f2['z_corners'][:, :, :] = z_corners
