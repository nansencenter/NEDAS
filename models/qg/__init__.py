import numpy as np
import os, glob
from datetime import datetime, timedelta
from pyproj import Proj
from grid import Grid
from .util import *

restart_dt = int(os.environ['cycle_period'])
kmax = int(os.environ['kmax'])
nz = int(os.environ['nz'])

variables = {'velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'restart_dt':restart_dt, 'levels':np.arange(nz), 'units':'*'},
             'streamfunc': {'name':'psi', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':np.arange(nz), 'units':'*'},
             'vorticity': {'name':'zeta', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':np.arange(nz), 'units':'*'},
             'temperature': {'name':'temp', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':np.arange(nz), 'units':'*'},
           }


def filename(path, **kwargs):
    if 'member' in kwargs and kwargs['member'] is not None:
        mstr = '{:03d}'.format(kwargs['member']+1)
    else:
        mstr = ''

    assert 'time' in kwargs, 'missing time in kwargs'
    tstr = kwargs['time'].strftime('%Y%m%d_%H')

    return path+'/'+mstr+'/output_'+tstr+'.bin'


uniq_grid_key = ()

def read_grid(path, **kwargs):
    ##get size from kmax
    n = 2*(kmax+1)
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    grid = Grid(Proj('+proj=stere'), x, y, cyclic_dim='xy')
    return grid


def write_grid(path, grid, **kwargs):
    pass


def read_mask(path, grid):
    mask = np.full(grid.x.shape, False)  ##no grid points are masked
    return mask


def read_var(path, grid, **kwargs):
    assert 'name' in kwargs, 'missing variable name in kwargs'
    name = kwargs['name']
    assert name in variables, 'variable name '+name+' not listed in variables'
    fname = filename(path, **kwargs)

    if 'k' in kwargs:
        k = kwargs['k']
    else:
        k = 0  ##read the first layer by default
    assert k>=0 and k<nz, f'level index {k} is not within range 0-{nz}'

    psik = read_data_bin(fname, kmax, nz, k)

    if name == 'streamfunc':
        return spec2grid(psik).T

    elif name == 'velocity':
        uk = psi2u(psik)
        vk = psi2v(psik)
        u = spec2grid(uk).T
        v = spec2grid(vk).T
        return np.array([u, v])

    elif name == 'vorticity':
        zetak = psi2zeta(psik)
        return spec2grid(zetak).T

    elif name == 'temperature':
        tempk = psi2temp(psik)
        return spec2grid(tempk).T


def write_var(path, grid, var, **kwargs):
    ##check kwargs
    assert 'name' in kwargs, 'missing variable name in kwargs'
    name = kwargs['name']
    assert name in variables, 'variable name '+name+' not listed in variables'
    fname = filename(path, **kwargs)

    if 'k' in kwargs:
        k = kwargs['k']
    else:
        k = 0  ##read the first layer by default
    assert k>=0 and k<nz, f'level index {k} is not within range 0-{nz}'

    if name == 'streamfunc':
        psik = grid2spec(var.T)

    elif name == 'velocity':
        uk = grid2spec(var[0,...].T)
        vk = grid2spec(var[1,...].T)
        psik = zeta2psi(uv2zeta(uk, vk))

    elif name == 'vorticity':
        zetak = grid2spec(var.T)
        psik = zeta2psi(zetak)

    elif name == 'temperature':
        tempk = grid2spec(var.T)
        psik = temp2psi(tempk)

    write_data_bin(fname, kmax, nz, k, psik)


uniq_z_key = ('k')
z_units = '*'

def z_coords(path, grid, **kwargs):
    assert 'k' in kwargs, 'qg.z_coords: missing k in kwargs'
    z = np.ones(grid.x.shape) * kwargs['k']
    return z


