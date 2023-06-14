##ECMWF forecast (hres)
import numpy as np
import pyproj
from netCDF4 import Dataset
import glob
from datetime import datetime, timedelta
from grid import Grid
from assim_tools import variables, units_convert

##variable dictionary for naming convention
var_dict = {'atmos_surf_wind': {'name':('10U', '10V'), 'nens':0, 'nz':0, 'units':'m/s'},
            'atmos_surf_temp': {'name':'2T', 'nens':0, 'nz':0, 'units':'K'},
            'atmos_surf_dew_temp': {'name':'2D', 'nens':0, 'nz':0, 'units':'K'},
            'atmos_surf_press': {'name':'MSL', 'nens':0, 'nz':0, 'units':'Pa'},
            'atmos_precip': {'name':'TP', 'nens':0, 'nz':0, 'units':'Mg/m2/3h'},
            'atmos_down_shortwave': {'name':'SSRD', 'nens':0, 'nz':0, 'units':'W s/m2'},
            'atmos_down_longwave': {'name':'STRD', 'nens':0, 'nz':0, 'units':'W s/m2'}, }

##format filename
def filename(path, **kwargs):
    if 'time' in kwargs:
        year = '{:04d}'.format(kwargs['time'].year)
        month = '{:02d}'.format(kwargs['time'].month)
        day = '{:02d}'.format(kwargs['time'].day)
    else:
        year = '????'
        month = '??'
        day = '??'
    flist = glob.glob(path+'/'+year+'/'+month+'/ec2_start'+year+month+day+'.nc')
    assert len(flist) > 0, 'no matching files found'
    return flist[0]

def time_index(time_series, time):
    t_ = (time - datetime(1950, 1, 1)) / timedelta(hours=1)
    ind = np.abs(time_series - t_).argmin()
    assert t_ - time_series[ind] == 0., "time index not found in file"
    return ind

def get_var(path, **kwargs):
    f = Dataset(filename(path, **kwargs))

    assert 'name' in kwargs, 'please specify which variable (name=?) to get'
    var_name = kwargs['name']
    assert var_name in var_dict, 'variable name '+var_name+' not listed in var_dict'

    if 'time' in kwargs:
        ts = f['time'][:].data
        t_index = time_index(ts, kwargs['time'])
    else:
        t_index = 0

    if 'member' in kwargs:
        m_index = kwargs['member']
    else:
        m_index = 0

    if 'level' in kwargs:
        z_index = kwargs['level']
    else:
        z_index = 0

    if variables[var_name]['is_vector']:
        x_name, y_name = var_dict[var_name]['name']
        var_x = f[x_name][t_index, ...].data
        var_y = f[y_name][t_index, ...].data
        var = np.array([var_x, var_y])
        if var_dict[var_name]['nens'] > 0:
            var = var[:, m_index, ...]
        if var_dict[var_name]['nz'] > 0:
            var = var[:, z_index, ...]
    else:
        name = var_dict[var_name]['name']
        var = f[name][t_index, ...].data
        if var_dict[var_name]['nens'] > 0:
            var = var[m_index, ...]
        if var_dict[var_name]['nz'] > 0:
            var = var[z_index, ...]

    ##convert units
    var = units_convert(variables[var_name]['units'], var_dict[var_name]['units'], var)
    return var

def get_xy(filename):
    f = Dataset(filename)
    lon = f['lon'][:].data
    lat = f['lat'][:].data
    x, y = np.meshgrid(lon, lat)
    return x, y

##ECMWF map projection
proj = pyproj.Proj('+proj=longlat')

def get_grid(filename):
    x, y = get_xy(filename)
    return Grid(proj, x, y, cyclic_dim='x', pole_dim='y', pole_index=(0,))

