import numpy as np
import matplotlib.pyplot as plt
import config.constants as cc
import grid
import grid.io.netcdf as nc
import datetime
import cartopy
from pynextsim.irregular_grid_interpolator import IrregularGridInterpolator
import sys

def filename(varname, t):
    return cc.SCRATCH+"/data/ERA5/ERA5_{}_y{:04d}.nc".format(varname, t.year)

def out_file(t):
    return cc.SCRATCH+"/data/GENERIC_PS_ATM/reference/generic_ps_atm_{:04d}{:02d}{:02d}.nc".format(t.year, t.month, t.day)

##translation of varname in generic_ps_atm to ERA5
varname = {'x_wind_10m':'u10',
           'y_wind_10m':'v10',
           'air_temperature_2m':'t2m',
           'dew_point_temperature_2m':'d2m',
           'instantaneous_downwelling_shortwave_radiation':'msdwswrf',
           'instantaneous_downwelling_longwave_radiation':'msdwlwrf',
           'atm_pressure':'msl',
           'total_precipitation_rate':'mtpr',
           'snowfall_rate':'msr'}
long_name = {'x_wind_10m':'10 metre wind speed in x direction (U10M)',
            'y_wind_10m':'10 metre wind speed in y direction (V10M)',
            'air_temperature_2m':'Screen level temperature (T2M)',
            'dew_point_temperature_2m':'Screen level dew point temperature (D2M)',
            'instantaneous_downwelling_shortwave_radiation':'Surface SW downwelling radiation rate',
            'instantaneous_downwelling_longwave_radiation':'Surface LW downwelling radiation rate',
            'atm_pressure':'Mean Sea Level Pressure (MSLP)',
            'total_precipitation_rate':'Total precipitation rate',
            'snowfall_rate':'Snowfall rate'}
units = {'x_wind_10m':'m/s',
        'y_wind_10m':'m/s',
        'air_temperature_2m':'K',
        'dew_point_temperature_2m':'K',
        'instantaneous_downwelling_longwave_radiation':'W/m^2',
        'instantaneous_downwelling_shortwave_radiation':'W/m^2',
        'atm_pressure':'Pa',
        'total_precipitation_rate':'kg/m^2/s',
        'snowfall_rate':'kg/m^2/s'}
##time dimension interval in ERA5 files
dt_in_file = datetime.timedelta(hours=1)

t_start = datetime.datetime.strptime(cc.DATE_START, '%Y%m%d%H%M')
t_end = datetime.datetime.strptime(cc.DATE_END, '%Y%m%d%H%M')
cp = datetime.timedelta(hours=cc.CYCLE_PERIOD/60)
n_per_day = int(np.maximum(1, 24*60/cc.CYCLE_PERIOD))

f = nc.Dataset(filename('u10', t_start))
lon, lat = np.meshgrid(f['longitude'][:], f['latitude'][:])
tmp = grid.crs.transform_points(cartopy.crs.PlateCarree(), lon, lat)
x = tmp[:,:,0]
y = tmp[:,:,1]
igi = IrregularGridInterpolator(x, y, grid.x_ref, grid.y_ref)
nx, ny = grid.x_ref.shape
longitude = igi.interp_field(lon).T
latitude = igi.interp_field(lat).T

t = t_start
while t < t_end:
    dat_out = dict()
    for v in varname:
        dat_out[v] = np.zeros((n_per_day, ny, nx))

    for n in range(n_per_day):
        t1 = t + n*cp
        t0 = datetime.datetime(t1.year,1,1,0,0,0)
        t_index = int((t1-t0)/dt_in_file)  ##time index in ERA5 files

        dat_orig = dict()
        for v in varname:
            dat_orig[v] = nc.Dataset(filename(varname[v], t1))[varname[v]][t_index, :, :]

        dat = dat_orig
        if 'x_wind_10m' in varname and 'y_wind_10m' in varname:
            dat['x_wind_10m'], dat['y_wind_10m'] = grid.rotate_vector(x, y, dat_orig['x_wind_10m'], dat_orig['y_wind_10m'])

        for v in varname:
            dat_out[v][n, :, :] = igi.interp_field(dat[v]).T

    ##write output
    time = float((t - datetime.datetime(1900,1,1,0,0,0))/datetime.timedelta(days=1)) + np.arange(n_per_day)/n_per_day
    nc.write(out_file(t), {'time':0}, 'time', time, {'standard_name':'time', 'long_name':'time', 'units':'days since 1900-01-01 00:00:00', 'calendar':'standard'})
    nc.write(out_file(m_index, t), {'y':ny}, 'y', grid.y_ref[0, :], {'standard_name':'projection_y_coordinate', 'units':'m', 'axis':'Y'})
    nc.write(out_file(m_index, t), {'x':nx}, 'x', grid.x_ref[:, 0], {'standard_name':'projection_x_coordinate', 'units':'m', 'axis':'X'})

    for v in varname:
        nc.write(out_file(m_index, t), {'time':0, 'y':ny, 'x':nx}, v, dat_out[v], {'long_name':long_name[v], 'standard_name':v, 'units':units[v], 'grid_mapping':'projection_stereo'})

    nc.write(out_file(m_index, t), {'y':ny, 'x':nx}, 'longitude', longitude, {'standard_name':'longitude', 'long_name':'longitude', 'units':'degree_east'})
    nc.write(out_file(m_index, t), {'y':ny, 'x':nx}, 'latitude', latitude, {'standard_name':'latitude', 'long_name':'latitude', 'units':'degree_north'})

    t += datetime.timedelta(days=1)
