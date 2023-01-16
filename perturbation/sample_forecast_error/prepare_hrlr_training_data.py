import numpy as np
import cartopy
import grid.io.netcdf as nc
import pynextsim.lib as nsl
from pynextsim.irregular_grid_interpolator import IrregularGridInterpolator
from pyproj import Proj
from datetime import datetime, timedelta
import os

def arome_filename(t):
    return 'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{:04d}/{:02d}/{:02d}/arome_arctic_full_2_5km_{:04d}{:02d}{:02d}T{:02d}Z.nc'.format(t.year, t.month, t.day, t.year, t.month, t.day, t.hour)

def ec_filename(t):
    return '/cluster/work/users/yingyue/data/ECMWF_forecast_arctic/{:04d}/{:02d}/ec2_start{:04d}{:02d}{:02d}.nc'.format(t.year, t.month, t.year, t.month, t.day)

out_path = '/cluster/work/users/yingyue/data/training'

t0 = datetime.strptime('202101010000', '%Y%m%d%H%M')
dt = timedelta(days=1)
varname = {'x_wind_10m':'10U',
           'y_wind_10m':'10V'}

out_crs = cartopy.crs.LambertConformal(central_longitude=-25., central_latitude=77.5, standard_parallels=(77.5, 77.5))
out_proj = Proj(proj='lcc', lat_0=77.5, lon_0=-25, lat_1=77.5, lat_2=77.5, R=6.371e6)

f1 = ec_filename(t0)
lon, lat = np.meshgrid(nc.read(f1, 'lon'), nc.read(f1, 'lat'))
tmp = out_crs.transform_points(cartopy.crs.PlateCarree(), lon, lat)
x = tmp[:, :, 0]; y = tmp[:, :, 1]

f2 = arome_filename(t0)
y_ref, x_ref = np.meshgrid(nc.read(f2, 'y'), nc.read(f2, 'x'))
nx, ny = x_ref.shape

igi = IrregularGridInterpolator(x, y, x_ref, y_ref)


for n in range(1000):
    t = t0 + n*dt

    ##read arome data and output to hr
    f = nc.Dataset(arome_filename(t))
    dat_orig = dict()
    for v in varname:
        dat_orig[v] = f[v][0, 0, :, :]
        dat_grid = np.zeros((1, ny, nx))
        dat_grid[0, :, :] = dat_orig[v]
        nc.write(out_path+'/hr/{:03d}.nc'.format(n), {'t':1, 'y':ny, 'x':nx}, v, dat_grid)
    f.close()

    ##read ecmwf data and output to lr
    f = ec_filename(t)
    dt_in_file = timedelta(hours=6)
    t_index = int((t - datetime(t.year, t.month, t.day, 0, 0, 0))/dt_in_file)
    f = nc.Dataset(f)
    dat_orig = dict()
    for v in varname:
        dat_orig[v] = f[varname[v]][t_index, :, :]
    f.close()
    dat = dat_orig
    if 'x_wind_10m' in varname and 'y_wind_10m' in varname:
        dat['x_wind_10m'], dat['y_wind_10m'] = nsl.transform_vectors(out_proj, x, y, dat_orig['x_wind_10m'], dat_orig['y_wind_10m'], fill_polar_hole=True)
    for v in varname:
        dat_grid = np.zeros((1, ny, nx))
        dat_grid[0, :, :] = igi.interp_field(dat[v]).T
        nc.write(out_path+'/lr/{:03d}.nc'.format(n), {'t':1, 'y':ny, 'x':nx}, v, dat_grid)

