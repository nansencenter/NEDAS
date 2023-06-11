##ERA5 reanalysis data
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset

def filename(path, mem, t):
    ##t: datetime obj
    return path+'/ERA5/ERA5_y{:04d}.nc'.format(t.year)

##variable naming convention
variables = {'atmos_surf_wind':    {'name':('u10', 'v10'), 'is_vector':True,  'unit':'m/s'  },
             'atmos_surf_temp':    {'name':'t2m',          'is_vector':False, 'unit':'K'    },
             'atmos_surf_dew_temp':{'name':'d2m',          'is_vector':False, 'unit':'K'    },
             'atmos_surf_press':   {'name':'msl',          'is_vector':False, 'unit':'Pa'   },
             'atmos_precip':       {'name':'mtpr',         'is_vector':False, 'unit':'kg/m2/s'},
             }

###time interval in each file
dt_in_file = timedelta(hours=3)


def get_var(in_file, var_name, t):
    f = Dataset(in_file)
    t0 = datetime(t.year, 1, 1)
    t_index = int((t-t0)/dt_in_file)
    if variables[var_name]['is_vector']:
        var_u = f[variables[var_name]['name'][0]][t_index, :].data
        var_v = f[variables[var_name]['name'][1]][t_index, :].data
        var_u = np.hstack((var_u[:, 361:], var_u[:, :361]))
        var_v = np.hstack((var_v[:, 361:], var_v[:, :361]))
        var = np.zeros((2,)+var_u.shape)
        var[0, :] = var_u
        var[1, :] = var_v
    else:
        var = f[variables[var_name]['name']][t_index, :].data
        ##ERA5 has longitude from 0 to 360, shift index so that it goes -180 to 180
        var = np.hstack((var[:, 361:], var[:, :361]))
    return var

def get_xy(in_file):
    f = Dataset(in_file)
    lon = f['longitude'][:].data
    lon = np.hstack((lon[361:]-360, lon[:361]))  ##shift lon to -180:180
    lat = f['latitude'][:].data
    x, y = np.meshgrid(lon, lat)
    return x, y

import pyproj
proj = pyproj.Proj('+proj=longlat')

def get_grid(in_file):
    from grid import Grid
    return Grid(proj, *get_xy(in_file), cyclic_dim='x', pole_dim='y', pole_index=(0,))

