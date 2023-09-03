import numpy as np
from netCDF4 import Dataset
import glob
from datetime import datetime, timedelta
from grid import Grid
from pyproj import Proj

##variable dictionary for ERA5 naming convention
variables = {'atmos_surf_wind': {'name':('u10', 'v10'), 'dtype':'float', 'is_vector':True, 'z_units':None, 'units':'m/s'},
             'atmos_surf_temp': {'name':'t2m', 'dtype':'float', 'is_vector':False, 'z_units':None, 'units':'K'},
             'atmos_surf_dew_temp': {'name':'d2m', 'dtype':'float', 'is_vector':False, 'z_units':None, 'units':'K'},
             'atmos_surf_pres': {'name':'msl', 'dtype':'float', 'is_vector':False, 'z_units':None, 'units':'Pa'},
             'atmos_precip': {'name':'mtpr', 'dtype':'float', 'is_vector':False, 'z_units':None, 'units':'kg/m2/s'},
             'atmos_snowfall': {'name':'msr', 'dtype':'float', 'is_vector':False, 'z_units':None, 'units':'kg/m2/s'},
             'atmos_down_shortwave': {'name':'msdwswrf', 'dtype':'float', 'is_vector':False, 'z_units':None, 'units':'W/m2'},
             'atmos_down_longwave': {'name':'msdwlwrf', 'dtype':'float', 'is_vector':False, 'z_units':None, 'units':'W/m2'}, }

###format filename
def filename(path, **kwargs):
    if 'time' in kwargs:
        year = '{:04d}'.format(kwargs['time'].year)
    else:
        year = '????'

    if 'member' in kwargs:
        mem = '{:03d}'.format(kwargs['member']+1)
    else:
        mem = ''

    if 'native_name' in kwargs:
        var = kwargs['native_name']
    else:
        var = '*'

    search = path+'/'+mem+'/ERA5_'+var+'_y'+year+'.nc'
    flist = glob.glob(search)
    assert len(flist) > 0, 'no matching files found'+search
    return flist[0]


##find the nearest index in data for the given t
def time_index(time_series, time):
    t_ = (time - datetime(1900, 1, 1)) / timedelta(hours=1)
    ind = np.abs(time_series - t_).argmin()
    assert t_ - time_series[ind] == 0., "time index not found in file"
    return ind


def read_var(path, grid, **kwargs):
    assert 'name' in kwargs, 'please specify which variable (name=?) to get'
    name = kwargs['name']
    assert name in variables, 'variable name '+name+' not listed in variables'

    if 'time' in kwargs:
        ts = Dataset(filename(path, **kwargs))['time'][:].data
        t_index = time_index(ts, kwargs['time'])
    else:
        t_index = 0

    if 'level' in kwargs:
        z_index = kwargs['level']
    else:
        z_index = 0

    if variables[name]['is_vector']:
        x_name, y_name = variables[name]['name']
        f = Dataset(filename(path, **kwargs, native_name=x_name))
        var_x = f[x_name][t_index, ...].data
        f = Dataset(filename(path, **kwargs, native_name=y_name))
        var_y = f[y_name][t_index, ...].data
        var = np.array([var_x, var_y])  ##assemble into var[2,...]

    else:
        name = variables[name]['name']
        f = Dataset(filename(path, **kwargs, native_name=name))
        var = f[name][t_index, ...].data

    return var


##ERA5 map projection
##get corresponding Grid object
def read_grid(path, **kwargs):
    proj = Proj('+proj=longlat')

    f = Dataset(filename(path, **kwargs))
    lon = f['longitude'][:].data
    lat = f['latitude'][:].data
    x, y = np.meshgrid(lon, lat)

    return Grid(proj, x, y, cyclic_dim='x', pole_dim='y', pole_index=(0,))


