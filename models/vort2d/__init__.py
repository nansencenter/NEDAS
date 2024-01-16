import numpy as np
import glob
from datetime import datetime, timedelta
from netcdf_lib import nc_read_var, nc_write_var

from .model import initialize, advance_time
from .param import *

##dictionary from state_def names to native variable names and properties
##List of properties:
##   name: native variable name in restart files, tuple of (u-name,v-name)
##         if vector field components are stored in separate native variables
##   dtype: double/flout/int
##   is_vector: if true the variable contains (u, v) components
##   restart_dt: how freq model output is available, in hours
##   levels: vertical level index list
##   units: native physical units for the variable
variables = {'velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'restart_dt':restart_dt, 'levels':np.array([0]), 'units':'m/s'}, }


def filename(path, **kwargs):
    """parse kwargs and find matching filename"""
    if 'member' in kwargs and kwargs['member'] is not None:
        mstr = '_mem{:03d}'.format(kwargs['member']+1)
    else:
        mstr = ''

    assert 'time' in kwargs, 'missing time in kwargs'
    tstr = kwargs['time'].strftime('%Y%m%d_%H')

    return path+'/'+tstr+mstr+'.nc'


from pyproj import Proj
from grid import Grid

##topaz grid is fixed in time/space, so no keys needed
uniq_grid_key = ()

###path and kwargs here are dummy input since the grid is fixed
def read_grid(path, **kwargs):
    ##any map projection will work, just define one arbitrarily
    proj = Proj('+proj=stere')

    ##define the coordinates
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    x = ii*dx
    y = jj*dx

    ##the domain is doubly periodic
    grid = Grid(proj, x, y, cyclic_dim='xy')

    return grid


def write_grid(path, **kwargs):
    pass


def read_mask(path, grid):
    mask = np.full(grid.x.shape, False)  ##no grid points are masked
    return mask


##get the state variable with name in state_def
##and other kwargs: time, level, and member to pinpoint where to get the variable
##returns a 2D field defined on grid from read_grid
def read_var(path, grid, **kwargs):
    ##check name in kwargs and read the variables from file
    assert 'name' in kwargs, 'please specify which variable to get, name=?'
    name = kwargs['name']
    assert name in variables, 'variable name '+name+' not listed in variables'
    fname = filename(path, **kwargs)

    if 'is_vector' in kwargs:
        is_vector = kwargs['is_vector']
    else:
        is_vector = variables[name]['is_vector']

    if is_vector:
        var1 = nc_read_var(fname, variables[name]['name'][0])[0, ...]
        var2 = nc_read_var(fname, variables[name]['name'][1])[0, ...]
        var = np.array([var1, var2])
    else:
        var = nc_read_var(fname, variables[name]['name'])[0, ...]

    return var


##output updated variable with name='varname' defined in state_def
##to the corresponding model restart file
def write_var(path, grid, var, **kwargs):
    ##check kwargs
    assert 'name' in kwargs, 'missing name in kwargs'
    name = kwargs['name']
    assert name in variables, 'variable name '+name+' not listed in variables'
    fname = filename(path, **kwargs)

    assert 'is_vector' in kwargs, 'missing is_vector in kwargs'
    if kwargs['is_vector']:
        for i in range(2):
            nc_write_var(fname, {'t':None, 'y':grid.ny, 'x':grid.nx}, variables[name]['name'][i], var[i,...], recno={'t':0})
    else:
        nc_write_var(fname, {'t':None, 'y':grid.ny, 'x':grid.nx}, var, variables[name]['name'], var, recno={'t':0})


##for z coordinates, nothing to do here other than return all zeros
uniq_z_key = ()
def z_coords(path, grid, **kwargs):
    return np.zeros(grid.x.shape)




