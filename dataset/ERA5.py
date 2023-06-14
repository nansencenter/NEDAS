##ERA5 reanalysis data
import numpy as np
import pyproj
from netCDF4 import Dataset
import glob
from datetime import datetime, timedelta
from grid import Grid
from assim_tools import variables, units_convert

##variable dictionary for ERA5 naming convention
var_dict = {'atmos_surf_wind': {'name':('u10', 'v10'), 'nz':0, 'units':'m/s'},
            'atmos_surf_temp': {'name':'t2m', 'nz':0, 'units':'K'},
            'atmos_surf_dew_temp': {'name':'d2m', 'nz':0, 'units':'K'},
            'atmos_surf_press': {'name':'msl', 'nz':0, 'units':'Pa'},
            'atmos_precip': {'name':'mtpr', 'nz':0, 'units':'kg/m2/s'},
            'atmos_snowfall': {'name':'msr', 'nz':0, 'units':'kg/m2/s'},
            'atmos_down_shortwave': {'name':'msdwswrf', 'nz':0, 'units':'W/m2'},
            'atmos_down_longwave': {'name':'msdwlwrf', 'nz':0, 'units':'W/m2'}, }

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

    flist = glob.glob(path+'/'+mem+'/ERA5_'+var+'_y'+year+'.nc')
    assert len(flist) > 0, 'no matching files found'
    return flist[0]

##find the nearest index in data for the given t
def time_index(time_series, time):
    t_ = (time - datetime(1900, 1, 1)) / timedelta(hours=1)
    ind = np.abs(time_series - t_).argmin()
    assert t_ - time_series[ind] == 0., "time index not found in file"
    return ind

def get_var(path, **kwargs):
    assert 'name' in kwargs, 'please specify which variable (name=?) to get'
    var_name = kwargs['name']
    assert var_name in var_dict, 'variable name '+var_name+' not listed in var_dict'

    if 'time' in kwargs:
        ts = Dataset(filename(path, **kwargs))['time'][:].data
        t_index = time_index(ts, kwargs['time'])
    else:
        t_index = 0

    if 'level' in kwargs:
        z_index = kwargs['level']
    else:
        z_index = 0

    if variables[var_name]['is_vector']:
        x_name, y_name = var_dict[var_name]['name']
        f = Dataset(filename(path, **kwargs, native_name=x_name))
        var_x = f[x_name][t_index, ...].data
        f = Dataset(filename(path, **kwargs, native_name=y_name))
        var_y = f[y_name][t_index, ...].data
        var = np.array([var_x, var_y])  ##assemble into var[2,...]
        if var_dict[var_name]['nz'] > 0: ##if nz>0, there is z dimension in data
            var = var[:, z_index, ...]
    else:
        name = var_dict[var_name]['name']
        f = Dataset(filename(path, **kwargs, native_name=name))
        var = f[name][t_index, ...].data
        if var_dict[var_name]['nz'] > 0:
            var = var[z_index, ...]

    ##ERA5 has longitude from 0 to 360, shift index so that it goes -180 to 180
    lon = f['longitude'][:].data
    i = np.searchsorted(lon, 180.)
    inds = np.hstack((np.arange(i+1, len(lon)), np.arange(0, i+1)))
    var = np.take(var, inds, axis=-1)

    ##convert units
    var = units_convert(variables[var_name]['units'], var_dict[var_name]['units'], var)
    return var

def get_xy(filename):
    f = Dataset(filename)
    lon = f['longitude'][:].data
    ##shift lon to -180:180
    i = np.searchsorted(lon, 180.)
    lon = np.hstack((lon[i+1:]-360, lon[:i+1]))
    lat = f['latitude'][:].data
    x, y = np.meshgrid(lon, lat)
    return x, y

##ERA5 map projection
proj = pyproj.Proj('+proj=longlat')

##get corresponding Grid object
def get_grid(filename):
    x, y = get_xy(filename)
    return Grid(proj, x, y, cyclic_dim='x', pole_dim='y', pole_index=(0,))

