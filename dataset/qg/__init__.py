import numpy as np
import importlib
from datetime import datetime, timedelta

variables = {'velocity': {'dtype':'float', 'is_vector':True, 'z_units':'*', 'units':'*'},
             'streamfunc': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
             'temperature': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
             'vorticity': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
            }

def random_network(path, grid, mask, z, truth_path, **kwargs):

    nobs = 1000  ##number of obs
    y = np.random.uniform(grid.ymin, grid.ymax, nobs)
    x = np.random.uniform(grid.xmin, grid.xmax, nobs)

    # obs_thin = 1
    # x, y = np.meshgrid(np.arange(grid.xmin, grid.xmax, obs_thin),
    #                    np.arange(grid.ymin, grid.ymax, obs_thin))
    # x = x.flatten()
    # y = y.flatten()
    # nobs = x.size

    # z = np.random.uniform(1, 7, nobs)
    z = np.zeros(nobs)

    obs_seq = {'obs': np.full(nobs, np.nan),
               't': np.full(nobs, kwargs['time']),
               'z': z,
               'y': y,
               'x': x,
               'err_std': np.ones(nobs) * kwargs['err']['std']
               }

    return obs_seq


obs_operator = {}


