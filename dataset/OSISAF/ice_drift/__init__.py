import numpy as np
import glob
from datetime import datetime, timedelta
import pyproj

variables = {'seaice_drift': {'dtype':'float', 'is_vector':True, 'z_type':'z', 'units':'m/s'},
            }

##osisaf grid definition
proj = pyproj.Proj("+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45")
x, y = np.meshgrid(np.arange(-3750000, 3626000, 62500), np.arange(5750000, -5251000, -62500))


def filename(path, **kwargs):
    name = kwargs['name'] if 'name' in kwargs else 'seaice

    return path+'/'+
