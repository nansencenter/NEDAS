##ECMWF forecast (hres)
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset

def filename(path, t):
    ##t: datetime obj
    return path+'/ECMWF_forecast/{:04d}/{:02d}/ec2_start{:04d}{:02d}{:02d}.nc'.format(t.year, t.month, t.year, t.month, t.day)

##variable naming convention
variables = {'atmos_surf_wind':      {'name':('10U', '10V'), 'is_vector':True,  'unit':'m/s'},
             'atmos_surf_temp':      {'name':'2T',           'is_vector':False, 'unit':'K'},
             'atmos_surf_dew_temp':  {'name':'2D',           'is_vector':False, 'unit':'K'},
             'atmos_surf_press':     {'name':'MSL',          'is_vector':False, 'unit':'Pa'},
             'atmos_precip':         {'name':'TP',           'is_vector':False,  'unit':'Mg/m2'},
             }

###time interval in each file
dt_in_file = timedelta(hours=6)

def get_var(in_file, var_name, t):
    f = Dataset(in_file)
    ##TODO: get variable according to t obj, ncfiles contain time info already
    t0 = datetime(t.year, t.month, t.day)
    t_index = int((t-t0)/dt_in_file)
    if variables[var_name]['is_vector']:
        var_u = f[variables[var_name]['name'][0]][t_index, :].data
        var_v = f[variables[var_name]['name'][1]][t_index, :].data
        var = np.zeros((2,)+var_u.shape)
        var[0, :] = var_u
        var[1, :] = var_v
    else:
        var = f[variables[var_name]['name']][t_index, :].data
    return var

def get_xy(in_file):
    f = Dataset(in_file)
    lon = f['lon'][:].data
    lat = f['lat'][:].data
    x, y = np.meshgrid(lon, lat)
    return x, y

import pyproj
proj = pyproj.Proj('+proj=longlat')

def get_grid(in_file):
    from grid import Grid
    return Grid(proj, *get_xy(in_file), cyclic_dim='x', pole_dim='y', pole_index=(0,))

