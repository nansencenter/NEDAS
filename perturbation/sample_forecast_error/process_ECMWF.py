import numpy as np
import grid.io.netcdf as nc
import grid
import config.constants as cc
from pynextsim.irregular_grid_interpolator import IrregularGridInterpolator
import datetime
import os
import sys

n_sample = int(sys.argv[1])

t0 = datetime.datetime.strptime('201911010000', '%Y%m%d%H%M')
dt_sample = datetime.timedelta(days=3)
dt = datetime.timedelta(days=1)
cycle_period = datetime.timedelta(minutes=cc.CYCLE_PERIOD)
Dt = datetime.timedelta(days=10)
varname = {'x_wind_10m':'10U',
           'y_wind_10m':'10V'}
           # 'atm_pressure':'MSL'}

dt_in_file = datetime.timedelta(hours=6)
assert(dt_in_file <= dt)

def filename(t):
    return cc.DATA_DIR+'/ECMWF_forecast/{:04d}/{:02d}/ec2_start{:04d}{:02d}{:02d}.nc'.format(t.year, t.month, t.year, t.month, t.day)


##get grid information
file0 = filename(t0)
lon, lat = np.meshgrid(nc.read(file0, 'lon'), nc.read(file0, 'lat'))
tmp = grid.crs.transform_points(grid.cartopy.crs.PlateCarree(), lon, lat)
x = tmp[:, :, 0]; y = tmp[:, :, 1]
igi = IrregularGridInterpolator(x, y, grid.x_ref, grid.y_ref)

for fcst_step in range(1, int(Dt/dt)+1):
    t1 = t0 + n_sample*dt_sample
    file1 = filename(t1)

    t2 = t1 - fcst_step*dt
    file2 = filename(t2)
    t_index = int(fcst_step*dt/dt_in_file)
    print('sample ', n_sample, 'fcst_step=', int(fcst_step*dt/cycle_period), t1, t2, t_index)

    dat_orig = dict()
    for v in varname:
        f1 = nc.Dataset(file1)
        f2 = nc.Dataset(file2)
        dat_orig[v] = f2[varname[v]][t_index, :, :] - f1[varname[v]][0, :, :]
        f1.close()
        f2.close()
        # dat_orig[v] = nc.read(file2, varname[v])[t_index, :, :] - nc.read(file1, varname[v])[0, :, :]

    dat = dat_orig
    if 'x_wind_10m' in varname and 'y_wind_10m' in varname:
        dat['x_wind_10m'], dat['y_wind_10m'] = grid.rotate_vector(x, y, dat_orig['x_wind_10m'], dat_orig['y_wind_10m'])

    out_path = cc.PERTURB_PARAM_DIR+'/sample_ECMWF/{:03d}/{:04d}'.format(n_sample, int(fcst_step*dt/cycle_period))
    if not os.path.exists(out_path):
            os.makedirs(out_path)

    for v in varname:
        dat_grid = np.zeros((1, cc.NY, cc.NX))
        dat_grid[0, :, :] = igi.interp_field(dat[v]).T
        nc.write(out_path+'/perturb.nc', {'t':1, 'y':cc.NY, 'x':cc.NX}, v, dat_grid)

