import numpy as np
import importlib
from datetime import datetime, timedelta
import os

variables = {'velocity': {'dtype':'float', 'is_vector':True, 'z_units':'*', 'units':'*'},
             'streamfunc': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
             'temperature': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
             'vorticity': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
            }

def random_network(path, grid, mask, model_z, truth_path, **kwargs):

    nobs = kwargs.get('nobs', 1000)  ##number of obs

    obs_y = np.random.uniform(grid.ymin, grid.ymax, nobs)
    obs_x = np.random.uniform(grid.xmin, grid.xmax, nobs)

    # obs_thin = 3
    # obs_x, obs_y = np.meshgrid(np.arange(grid.xmin, grid.xmax, obs_thin),
    #                            np.arange(grid.ymin, grid.ymax, obs_thin))
    # obs_x = obs_x.flatten()
    # obs_y = obs_y.flatten()
    # nobs = obs_x.size

    # obs_z = np.random.uniform(0, 1, nobs)
    obs_z = np.zeros(nobs)

    obs_seq = {'obs': np.full(nobs, np.nan),
               't': np.full(nobs, kwargs['time']),
               'z': obs_z,
               'y': obs_y,
               'x': obs_x,
               'err_std': np.ones(nobs) * kwargs['err']['std']
               }

    return obs_seq


obs_operator = {}


