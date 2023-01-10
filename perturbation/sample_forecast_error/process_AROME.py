import numpy as np
import grid.io.netcdf as nc
import grid
import config.constants as cc
from pynextsim.irregular_grid_interpolator import IrregularGridInterpolator
from datetime import datetime, timedelta
import os
import sys

n_sample = int(sys.argv[1])

t0 = datetime.strptime('201911010000', '%Y%m%d%H%M')
dt_sample = timedelta(days=1.5)
dt = timedelta(minutes=cc.CYCLE_PERIOD)
cycle_period = timedelta(minutes=cc.CYCLE_PERIOD)
Dt = timedelta(hours=66)
varname = {'x_wind_10m':'x_wind_10m',
           'y_wind_10m':'y_wind_10m'}
           # 'atm_pressure':'air_pressure_at_sea_level'}

dt_in_file = timedelta(hours=1)
assert(dt_in_file <= dt)

def filename(t):
    return 'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{:04d}/{:02d}/{:02d}/arome_arctic_full_2_5km_{:04d}{:02d}{:02d}T{:02d}Z.nc'.format(t.year, t.month, t.day, t.year, t.month, t.day, t.hour)


##get grid information
file0 = filename(t0)
lon = nc.read(file0, 'longitude')
lat = nc.read(file0, 'latitude')
nx, ny = lat.shape

for fcst_step in range(1, int(Dt/dt)+1):
    t1 = t0 + n_sample*dt_sample
    file1 = filename(t1)

    t2 = t1 + fcst_step*dt
    file2 = filename(t2)
    t_index = int(fcst_step*dt/dt_in_file)
    print('sample ', n_sample, 'fcst_step=', int(fcst_step*dt/cycle_period), t1, t2, t_index)

    dat_orig = dict()
    for v in varname:
        f1 = nc.Dataset(file1)
        f2 = nc.Dataset(file2)
        dat_orig[v] = f1[varname[v]][t_index, 0, :, :] - f2[varname[v]][0, 0, :, :]
        f1.close()
        f2.close()

    out_path = cc.PERTURB_PARAM_DIR+'/sample_AROME/{:03d}/'.format(n_sample)
    if not os.path.exists(out_path):
            os.makedirs(out_path)

    for v in varname:
        dat_grid = np.zeros((1, ny, nx))
        dat_grid[0, :, :] = dat_orig[v]
        nc.write(out_path+'/perturb_{:04d}.nc'.format(int(fcst_step*dt/cycle_period)), {'t':1, 'y':ny, 'x':nx}, v, dat_grid)

