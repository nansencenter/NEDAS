import numpy as np
import importlib
from datetime import datetime, timedelta

variables = {'velocity': {'dtype':'float', 'is_vector':True, 'z_units':'*', 'units':'*'},
             'streamfunc': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
             'temperature': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
             'vorticity': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
            }

def random_network(path, grid, mask, z, truth_path, **kwargs):

    nobs = 5000  ##number of obs
    y = np.random.uniform(grid.ymin, grid.ymax, nobs)
    x = np.random.uniform(grid.xmin, grid.xmax, nobs)

    obs_seq = {'obs': np.full(nobs, np.nan),
               't': np.full(nobs, kwargs['time']),
               'z': np.zeros(nobs),
               'y': y,
               'x': x,
               'err_std': np.ones(nobs) * kwargs['err']['std']
               }

    return obs_seq


obs_operator = {}


