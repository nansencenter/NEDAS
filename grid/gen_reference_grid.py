import numpy as np
from netCDF4 import Dataset

#Ying 2022
##given lat/lon defining a 2d irregular grid
##compute the corresponding theta (vector rotating angle)
##and stereographic projection coords (x, y, z)
##make 4 corners for conservative mapping interpolation
##--these variables are used in reference_grid.nc for nextsim
##  to output nc format statevector

def latlon2xyz(lat, lon):
    x = np.cos(lat*np.pi/180)*np.cos(lon*np.pi/180)
    y = np.cos(lat*np.pi/180)*np.sin(lon*np.pi/180)
    z = np.sin(lat*np.pi/180)
    return x, y, z

def xyz2latlon(x, y, z):
    lat = np.arcsin(z)*180/np.pi
    lon = np.arctan(y/x)*180/np.pi
    return lat, lon

def get_theta(x, y):
    nx, ny = x.shape
    theta = np.zeros((nx, ny))
    for j in range(ny):
        dx = x[1,j] - x[0,j]
        dy = x[1,j] - y[0,j]
        theta[0,j] = np.arctan(dy/dx)
        for i in range(1, nx-1):
            dx = x[i+1,j] - x[i-1,j]
            dy = y[i+1,j] - y[i-1,j]
            theta[i,j] = np.arctan(dy/dx)
        dx = x[nx-1,j] - x[nx-2,j]
        dy = y[nx-1,j] - y[nx-2,j]
        theta[nx-1,j] = np.arctan(dy/dx)
    return theta

def get_corners(x):
    nx, ny = x.shape
    xt = np.zeros((nx+1, ny+1))
    ##use linear interp in interior
    xt[1:nx, 1:ny] = 0.25*(x[1:nx, 1:ny] + x[1:nx, 0:ny-1] + x[0:nx-1, 1:ny] + x[0:nx-1, 0:ny-1])
    ##use 2nd-order polynomial extrapolat along borders
    xt[0, :] = 3*xt[1, :] - 3*xt[2, :] + xt[3, :]
    xt[nx, :] = 3*xt[nx-1, :] - 3*xt[nx-2, :] + xt[nx-3, :]
    xt[:, 0] = 3*xt[:, 1] - 3*xt[:, 2] + xt[:, 3]
    xt[:, ny] = 3*xt[:, ny-1] - 3*xt[:, ny-2] + xt[:, ny-3]
    ##make corners into new dimension
    x_corners = np.zeros((nx, ny, 4))
    x_corners[:, :, 0] = xt[0:nx, 0:ny]
    x_corners[:, :, 1] = xt[0:nx, 1:ny+1]
    x_corners[:, :, 2] = xt[1:nx+1, 1:ny+1]
    x_corners[:, :, 3] = xt[1:nx+1, 0:ny]
    return x_corners

f = Dataset('latlon.nc')
plat = f['latitude'][:, :]
plon = f['longitude'][:, :]
x, y, z = latlon2xyz(plat, plon)
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
