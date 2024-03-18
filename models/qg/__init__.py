import numpy as np
import os, glob
from datetime import datetime, timedelta
from pyproj import Proj
from grid import Grid
from .util import *

restart_dt = int(os.environ['cycle_period'])
kmax = int(os.environ['kmax'])
nz = int(os.environ['nz'])
dz = float(os.environ['dz'])
levels = np.arange(0, nz, dz)

variables = {'velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
             'streamfunc': {'name':'psi', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
             'vorticity': {'name':'zeta', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
             'temperature': {'name':'temp', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
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

    k1 = int(k)
    if k1 < nz-1:
        k2 = k1+1
    else:
        k2 = k1

    psik1 = read_data_bin(fname, kmax, nz, k1)
    psik2 = read_data_bin(fname, kmax, nz, k2)

    if name == 'streamfunc':
        var1 = spec2grid(psik1).T
        var2 = spec2grid(psik2).T

    elif name == 'velocity':
        uk1 = psi2u(psik1)
        vk1 = psi2v(psik1)
        u1 = spec2grid(uk1).T
        v1 = spec2grid(vk1).T
        var1 = np.array([u1, v1])
        uk2 = psi2u(psik2)
        vk2 = psi2v(psik2)
        u2 = spec2grid(uk2).T
        v2 = spec2grid(vk2).T
        var2 = np.array([u2, v2])

    elif name == 'vorticity':
        zetak1 = psi2zeta(psik1)
        var1 = spec2grid(zetak1).T
        zetak2 = psi2zeta(psik2)
        var2 = spec2grid(zetak2).T

    elif name == 'temperature':
        tempk1 = psi2temp(psik1)
        var1 = spec2grid(tempk1).T
        tempk2 = psi2temp(psik2)
        var2 = spec2grid(tempk2).T

    ##vertical interp between var1 and var2
    if k1 < nz-1:
        return (var1*(k2-k) + var2*(k-k1)) / (k2-k1)
    else:
        return var1


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

    if k==int(k):
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

        write_data_bin(fname, kmax, nz, int(k), psik)


uniq_z_key = ('k')
z_units = '*'

def z_coords(path, grid, **kwargs):
    assert 'k' in kwargs, 'qg.z_coords: missing k in kwargs'
    z = np.ones(grid.x.shape) * kwargs['k']
    return z


