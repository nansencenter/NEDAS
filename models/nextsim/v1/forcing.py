import numpy as np
import os
from datetime import datetime, timedelta
from utils.netcdf_lib import nc_write_var, nc_read_var
from utils.conversion import t2s, s2t, dt1h
from perturb import random_perturb

##specific about generic_ps_atm files
nt_per_day = 4  ##number of fields per day (per file)

from pyproj import Proj
proj = Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)
from grid import Grid
grid = Grid.regular_grid(proj, -2.5e6, 2.498e6, -2e6, 2.5e6, 3e3, centered=True)

###generic atmos forcing file for nextsim

def filename(**kwargs):
    path = kwargs.get('path', '.')
    time = kwargs.get('time', datetime(2007,1,1))
    return os.path.join(path, "generic_ps_atm_"+time.strftime('%Y%m%d')+".nc")

##translation of varname in generic_ps_atm to ERA5
variables = {'atmos_surf_wind': {'name':('x_wind_10m', 'y_wind_10m'), 'is_vector':True, 'long_name':('10 metre wind speed in x direction (U10M)', '10 metre wind speed in y direction (V10M)'), 'units':'m/s'},
             'atmos_surf_temp': {'name':'ait_temperature_2m', 'is_vector':False, 'long_name':'Screen level temperature (T2M)', 'units':'K'},
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
    assert name in variables, 'variable '+name+' not defined in forcing.variables'
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
    assert name in variables, 'variable '+name+' not defined in forcing.variables'
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


##perturb forcing
def perturb_var(**kwargs):
    """
    Add random perturbation to the given list of variables
    perturb_opt: dict
        - variables: list of variable names
        - type: list of str: 'gaussian', 'displace', ...
        - amp: list of float
        - hcorr: list of float
        - tcorr: list of float
    """
    perturb_opt = kwargs['perturb']
    time = kwargs['time']
    prev_time = time - kwargs['forecast_period'] * dt1h

    for i in range(len(perturb_opt['variables'])):
        name = perturb_opt['variables'][i]
        assert name in variables, 'variable '+name+' not defined in forcing.variables'
        perturb_type = perturb_opt['type'][i]
        amp = perturb_opt['amp'][i]
        hcorr = perturb_opt['hcorr'][i]
        tcorr = perturb_opt['tcorr'][i]
        if variables[name]['is_vector']:
            vlist = [0, 1]
        else:
            vlist = [None]

        ##read the forcing variable to be perturbed
        data = read_var(name=name, **kwargs)

        ##perturb the forcing variable
        perturb = np.zeros(data.shape)
        for v in vlist:
            ##previous perturbation (if not exist then this is the first time step)
            prev_perturb_file = os.path.join(kwargs['path'], 'perturb_'+name+'_'+t2s(prev_time)+'.npy')
            if os.path.exists(prev_perturb_file):
                prev_perturb = np.load(prev_perturb_file, allow_pickle=True).item()[v, ...]
            else:
                prev_perturb = None
            data[v, ...], perturb[v, ...] = random_perturb(data[v, ...], grid, perturb_type, amp, hcorr, tcorr, prev_perturb=prev_perturb)

        ##write perturbed forcing variable back to file
        write_var(data, name=name, **kwargs)

        ##save a copy of perturbation (will be used as prev_perturb for next_time)
        perturb_file = os.path.join(kwargs['path'], 'perturb_'+name+'_'+t2s(time)+'.npy')
        np.save(perturb_file, perturb)

