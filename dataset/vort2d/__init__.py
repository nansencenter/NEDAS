import numpy as np
import importlib
from datetime import datetime, timedelta

variables = {'velocity': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'m/s'},
             'vortex_position': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'m'},
             'vortex_intensity': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m/s'},
             'vortex_size':  {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m'},
            }


def random_network(path, grid, mask, z, truth_path, **kwargs):

    ##get truth vortex position, some network is vortex-following
    velocity = get_velocity(truth_path, grid, **kwargs)
    i, j = vortex_position(velocity[0,...], velocity[1,...])
    true_center_x, true_center_y = grid.x[j,i], grid.y[j,i]

    if 'network_type' in kwargs:
        network_type = kwargs['network_type']
    else:
        network_type = 'global'

    if kwargs['name'] == 'velocity':

        if network_type == 'global':
            nobs = 1000  ##number of obs
            y = np.random.uniform(grid.ymin, grid.ymax, nobs)
            x = np.random.uniform(grid.xmin, grid.xmax, nobs)

        elif network_type == 'targeted':
            nobs = 800  ##note: number of obs in entire domain
                        ##later only obs within range will be kept
            obs_range = 180000  ##observed range from vortex center, m
            y = np.random.uniform(grid.ymin, grid.ymax, nobs)
            x = np.random.uniform(grid.xmin, grid.xmax, nobs)

            dist = np.hypot(x - true_center_x, y - true_center_y)
            ind = np.where(dist <= obs_range)
            x = x[ind]
            y = y[ind]
            nobs = x.size

        else:
            raise ValueError('unknown network type: '+network_type)

        obs_seq = {'obs': np.full(nobs, np.nan),
                   't': np.full(nobs, kwargs['time']),
                   'z': np.zeros(nobs),
                   'y': y,
                   'x': x,
                   'err_std': np.ones(nobs) * kwargs['err']['std']
                  }

    elif kwargs['name'] == 'vortex_position':
        obs_seq = {'obs': np.array([[np.nan, np.nan]]),
                   't': np.array([kwargs['time']]),
                   'z': np.array([0]),
                   'y': np.array([true_center_y]),
                   'x': np.array([true_center_x]),
                   'err_std': np.array([kwargs['err']['std']])
                   }

    elif kwargs['name'] in ['vortex_intensity', 'vortex_size']:
        obs_seq = {'obs': np.array([np.nan]),
                   't': np.array([kwargs['time']]),
                   'z': np.array([0]),
                   'y': np.array([true_center_y]),
                   'x': np.array([true_center_x]),
                   'err_std': np.array([kwargs['err']['std']])
                   }

    else:
        raise ValueError('unknown obs variable: '+kwargs['name'])

    return obs_seq


###utility functions for obs diagnostics

def vortex_position(u, v):
    ny, nx = u.shape

    ##compute vorticity
    zeta = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1) - np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0)) / 2.0

    ##search for max vorticity
    zmax = -999
    center_x, center_y = -1, -1
    buff = 6
    for j in range(buff, ny-buff):
        for i in range(buff, nx-buff):
            z = np.sum(zeta[j-buff:j+buff, i-buff:i+buff])
            if z > zmax:
                zmax = z
                center_i, center_j = i, j

    return center_i, center_j


def vortex_intensity(u, v):
    return np.max(np.hypot(u, v))


def vortex_size(u, v, center_i, center_j):
    wind = np.hypot(u, v)
    ny, nx = wind.shape

    nr = 30
    wind_min = 15
    wind_rad = np.zeros(nr)
    count_rad = np.zeros(nr)
    for j in range(-nr, nr+1):
        for i in range(-nr, nr+1):
            r = int(np.sqrt(i**2+j**2))
            if r < nr:
                wind_rad[r] += wind[int(center_j+j)%ny, int(center_i+i)%nx]
                count_rad[r] += 1
    wind_rad = wind_rad/count_rad

    if np.max(wind_rad)<wind_min or np.where(wind_rad>=wind_min)[0].size==0:
        Rsize = -1
    else:
        i1 = np.where(wind_rad>=wind_min)[0][-1] ###last point with wind > 35knot
        if i1==nr-1:
            Rsize = i1
        else:
            Rsize = i1 + (wind_rad[i1] - wind_min) / (wind_rad[i1] - wind_rad[i1+1])

    return Rsize


##package the functions into obs_operator, to be used in assim_tools.obs.state_to_obs
obs_operator = {}

def get_velocity(path, grid, **kwargs):
    model = importlib.import_module('models.vort2d')
    ##get the velocity field from model
    kwargs['name'] = 'velocity'
    return model.read_var(path, grid, **kwargs)


def get_vortex_position(path, grid, mask, z, **kwargs):
    velocity = get_velocity(path, grid, **kwargs)
    center_i, center_j = vortex_position(velocity[0,...], velocity[1,...])
    obs_seq = np.zeros((2, 1), dtype='float')
    obs_seq[0,0] = grid.x[center_j, center_i]
    obs_seq[1,0] = grid.y[center_j, center_i]
    return obs_seq


def get_vortex_intensity(path, grid, mask, z, **kwargs):
    velocity = get_velocity(path, grid, **kwargs)
    Vmax = vortex_intensity(velocity[0,...], velocity[1,...])
    return np.array([Vmax])


def get_vortex_size(path, grid, mask, z, **kwargs):
    velocity = get_velocity(path, grid, **kwargs)
    center_i, center_j = vortex_position(velocity[0,...], velocity[1,...])
    Rsize = vortex_size(velocity[0,...], velocity[1,...], center_i, center_j)
    Rsize = Rsize * grid.dx
    return np.array([Rsize])


obs_operator['vort2d'] = {'vortex_position': get_vortex_position,
                          'vortex_intensity': get_vortex_intensity,
                          'vortex_size': get_vortex_size, }


