import numpy as np
from assim_tools.basic_io import nc_write_var
from datetime import datetime, timedelta

def filename(path, t):
    return path+"/generic_ps_atm_{:04d}{:02d}{:02d}.nc".format(t.year, t.month, t.day)

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

##write forcing file for nextsim
def write_forcing(fname, varname, data, t, nt_per_day, grid):
    # assert var[0].shape[0] == len(t) == nt_per_day, "nt per day mismatch with var and/or t"
    ny, nx = grid.x.shape
    time = float((t - datetime(1900,1,1,0,0,0))/timedelta(days=1)) + np.arange(nt_per_day)/nt_per_day
    nc_write_var(fname, {'time':0}, 'time', time, {'standard_name':'time', 'long_name':'time', 'units':'days since 1900-01-01 00:00:00', 'calendar':'standard'})
    nc_write_var(fname, {'y':ny}, 'y', grid.y[:, 0], {'standard_name':'projection_y_coordinate', 'units':'m', 'axis':'Y'})
    nc_write_var(fname, {'x':nx}, 'x', grid.x[0, :], {'standard_name':'projection_x_coordinate', 'units':'m', 'axis':'X'})
    lon, lat = grid.proj(grid.x, grid.y, inverse=True)
    nc_write_var(fname, {'y':ny, 'x':nx}, 'longitude', lon, {'standard_name':'longitude', 'long_name':'longitude', 'units':'degree_east'})
    nc_write_var(fname, {'y':ny, 'x':nx}, 'latitude', lat, {'standard_name':'latitude', 'long_name':'latitude', 'units':'degree_north'})

    for v in varname:
        nc_write_var(fname, {'time':0, 'y':ny, 'x':nx}, v, data[v], {'long_name':variables[v]['long_name'], 'standard_name':variables[v]['name'], 'units':variables[v]['units'], 'grid_mapping':'projection_stereo'})
