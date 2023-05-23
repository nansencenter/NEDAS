##ECMWF forecast (hres)
import numpy as np
import config as cc
from datetime import datetime, timedelta
from netCDF4 import Dataset

def filename(t):
    ##t: datetime obj
    return cc.DATA_DIR+'/ECMWF_forecast/{:04d}/{:02d}/ec2_start{:04d}{:02d}{:02d}.nc'.format(t.year, t.month, t.year, t.month, t.day)

##variable naming convention
var_dic = {
        'wind_u':'10U',
        'wind_v':'10V',
          }

###time interval in each file
dt_in_file = timedelta(hours=6)

def get_var(in_file, var_name, t):
    f = Dataset(in_file)
    t0 = datetime(t.year, t.month, t.day)
    t_index = int((t-t0)/dt_in_file)
    if var_name == 'wind':
        var_u = f[var_dic['wind_u']][t_index, :]
        var_v = f[var_dic['wind_v']][t_index, :]
        var = np.zeros((2,)+var_u.shape)
        var[0, :] = var_u
        var[1, :] = var_v
    else:
        var = f[var_dic[var_name]][t_index, :]
    return var

def get_xy(in_file):
    f = Dataset(in_file)
    lon = f['lon'][:]
    lat = f['lat'][:]
    x, y = np.meshgrid(lon, lat)
    return x, y

import pyproj
proj = pyproj.Proj('+proj=latlong')

