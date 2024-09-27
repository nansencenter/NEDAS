import numpy as np
import os
from datetime import datetime, timedelta
from utils.netcdf_lib import nc_write_var, nc_read_var
from utils.conversion import t2s, s2t, dt1h

##specifics about generic_ps_atm (atmospheric forcing) files for nextsim
nt_per_day = 4  ##number of fields per day (per file)

from grid import Grid
from .gmshlib import proj
grid = Grid.regular_grid(proj, -2.5e6, 2.498e6, -2e6, 2.5e6, 3e3, centered=True)

def filename(**kwargs):

    if 'path' in kwargs:
        path = kwargs['path']
    else:
        path = '.'

    if 'member' in kwargs and kwargs['member'] is not None:
        mstr = '{:03d}'.format(kwargs['member']+1)
    else:
        mstr = ''

    time = kwargs['time']
    return os.path.join(path, mstr, "data", "GENERIC_PS_ATM", "generic_ps_atm_"+time.strftime('%Y%m%d')+".nc")

##translation of varname in generic_ps_atm to ERA5
variables = {'atmos_surf_wind': {'name':('x_wind_10m', 'y_wind_10m'), 'is_vector':True, 'long_name':('10 metre wind speed in x direction (U10M)', '10 metre wind speed in y direction (V10M)'), 'units':'m/s'},
             'atmos_surf_temp': {'name':'air_temperature_2m', 'is_vector':False, 'long_name':'Screen level temperature (T2M)', 'units':'K'},
             'atmos_surf_dew_temp': {'name':'dew_point_temperature_2m', 'is_vector':False, 'long_name':'Screen level dew point temperature (D2M)', 'units':'K'},
             'atmos_surf_press': {'name':'atm_pressure', 'is_vector':False, 'long_name':'Mean Sea Level Pressure (MSLP)', 'units':'Pa'},
             'atmos_precip': {'name':'total_precipitation_rate', 'is_vector':False, 'long_name':'Total precipitation rate', 'units':'kg/m2/s'},
             'atmos_snowfall': {'name':'snowfall_rate', 'is_vector':False, 'long_name':'Snowfall rate', 'units':'kg/m2/s'},
             'atmos_down_shortwave': {'name':'instantaneous_downwelling_shortwave_radiation', 'is_vector':False, 'long_name':'Surface SW downwelling radiation rate', 'units':'W/m2'},
             'atmos_down_longwave': {'name':'instantaneous_downwelling_longwave_radiation', 'is_vector':False, 'long_name':'Surface LW downwelling radiation rate', 'units':'W/m2'},
             }

def read_var(**kwargs):
    fname = filename(**kwargs)
    assert 'name' in kwargs, 'please specify variable name in read_forcing'
    name = kwargs['name']
    assert name in variables, 'variable '+name+' not defined in atmos_forcing.variables'
    time = kwargs['time']
    nt_in_file = int(np.round(time.hour / (24/nt_per_day)))

    if variables[name]['is_vector']:
        u = nc_read_var(fname, variables[name]['name'][0])[nt_in_file, ...]
        v = nc_read_var(fname, variables[name]['name'][1])[nt_in_file, ...]
        data = np.array([u, v])
    else:
        data = nc_read_var(fname, variables[name]['name'])[nt_in_file, ...]
    return data

##write forcing file for nextsim
def write_var(data, **kwargs):
    fname = filename(**kwargs)
    assert 'name' in kwargs, 'please specify variable name in write_forcing'
    name = kwargs['name']
    assert name in variables, 'variable '+name+' not defined in atmos_forcing.variables'
    time = kwargs['time']
    nt_in_file = int(np.round(time.hour / (24/nt_per_day)))
    ny, nx = data.shape[-2:]

    if variables[name]['is_vector']:
        for i in range(2):
            data_attr={'long_name':variables[name]['long_name'][i],
                       'standard_name':variables[name]['name'][i],
                       'units':variables[name]['units'],
                       'grid_mapping':'projection_stereo'}
            nc_write_var(fname, {'time':None, 'y':ny, 'x':nx}, variables[name]['name'][i], data[i, ...], recno={'time':nt_in_file}, attr=data_attr)

    else:
        data_attr={'long_name':variables[name]['long_name'],
                   'standard_name':variables[name]['name'],
                   'units':variables[name]['units'],
                   'grid_mapping':'projection_stereo'}
        nc_write_var(fname, {'time':None, 'y':ny, 'x':nx}, variables[name]['name'], data, recno={'time':nt_in_file}, attr=data_attr)

