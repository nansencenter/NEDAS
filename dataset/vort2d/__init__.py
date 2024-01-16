import numpy as np
from datetime import datetime, timedelta

variables = {'velocity': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'m/s'},
             'vortex_position': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'m'},
             'vortex_intensity': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m/s'},
             'vortex_size':  {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m'},
            }


def random_network(path, grid, mask, z, **kwargs):
    if kwargs['name'] == 'velocity':
        nobs = 500  ##number of obs
        # obs_range = 180000  ##observed range from vortex center, m

        obs_seq = {'obs': np.full(nobs, np.nan),
                   't': np.full(nobs, kwargs['time']),
                   'z': np.zeros(nobs),
                   'y': np.random.uniform(grid.ymin, grid.ymax, nobs),
                   'x': np.random.uniform(grid.xmin, grid.xmax, nobs),
                   'err_std': np.ones(nobs) * kwargs['err']['std']
                  }

    ##TODO: diagnostic vortex obs
    ##get truth velocity field

    return obs_seq


##some observation operators obs_operator[model][obs_var_name]
obs_operator = {}

def get_vortex_position(path, grid, mask, z, **kwargs):
    
    u, v = (X[:, :, 0], X[:, :, 1])
    ni, nj = u.shape
    zeta = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0) - np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1))/2.0
    zmax = -999
    ic, jc = (-1, -1)
    ##coarse search
    buff = 6
    for i in range(buff, ni-buff):
        for j in range(buff, nj-buff):
            z = np.sum(zeta[i-buff:i+buff, j-buff:j+buff])
            if z > zmax:
                zmax = z
                ic, jc = (i, j)

    return (center_x, center_y)


def get_vortex_intensity():

    return Vmax


def get_vortex_size():

    return Rsize


obs_operator['vort2d'] = {'vortex_position': get_vortex_position,
                          'vortex_intensity': get_vortex_intensity,
                          'vortex_size': get_vortex_size, }


