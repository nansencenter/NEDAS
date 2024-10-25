import numpy as np
import importlib

variables = {'state': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
            }

def random_network(path, grid, mask, z, truth_path, **kwargs):

    nobs = kwargs.get('nobs', 1)
    obs_x = np.random.uniform(grid.xmin, grid.xmax, nobs)

    obs_seq = {'obs': np.full(nobs, np.nan),
               't': np.full(nobs, kwargs['time']),
               'z': np.zeros(nobs),
               'y': np.zeros(nobs),
               'x': obs_x,
               'err_std': np.ones(nobs) * kwargs['err']['std'],
              }

    return obs_seq

