import numpy as np
import glob
from datetime import datetime, timedelta

from conversion import units_convert
from .bin_io import read_data, write_data

##Note: we only work with restart files, normal nextsim binfile have some variables names that
##are different from restart files, e.g. Concentration instead of M_conc
variables = {'seaice_conc': {'name':'M_conc', 'dtype':'float', 'is_vector':False, 'restart_dt':3, 'levels':[0], 'units':'%' },
             'seaice_thick': {'name':'M_thick', 'dtype':'float', 'is_vector':False, 'restart_dt':3, 'levels':[0], 'units':'m' },
             'seaice_damage': {'name':'M_damage', 'dtype':'float', 'is_vector':False, 'restart_dt':3, 'levels':[0], 'units':'%' },
             'snow_thick': {'name':'M_snow_thick', 'dtype':'float', 'is_vector':False, 'restart_dt':3, 'levels':[0], 'units':'m' },
             'seaice_velocity': {'name':'M_VT', 'dtype':'float', 'is_vector':True, 'restart_dt':3, 'levels':[0], 'units':'m/s' },
             }


def filename(path, **kwargs):
    name = kwargs['name'] if 'name' in kwargs else list(variables.keys())[0]

    if 'time' in kwargs and kwargs['time'] is not None:
        assert isinstance(kwargs['time'], datetime), 'Error: time is not a datetime object'
        tstr = kwargs['time'].strftime('%Y%m%dT%H%M%SZ')
    else:
        tstr = '*'
    if 'dt' in kwargs:
        dt = kwargs['dt'] * timedelta(hours=1)
    else:
        dt = 0

    if 'member' in kwargs and kwargs['member'] is not None:
        mstr = '{:03d}'.format(kwargs['member']+1)
    else:
        mstr = ''

    search = path+'/'+mstr+'/restart/field_'+tstr+'.bin'
    flist = glob.glob(search)
    assert len(flist)>0, 'no matching files found: '+search
    return flist[0]


##the map projection used in nextsim is defined in gmshlib:
from .gmshlib import read_mshfile, proj
from grid import Grid

##nextsim use moving mesh, so it varies for each member and time
uniq_grid_key = ('member', 'time')


def read_grid(path, **kwargs):
    """read native grid from mesh file, returns a grid obj"""
    meshfile = filename(path, **kwargs).replace('field', 'mesh')
    x = read_data(meshfile, 'Nodes_x')
    y = read_data(meshfile, 'Nodes_y')
    elements = read_data(meshfile, 'Elements')
    ne = int(elements.size/3)
    triangles = elements.reshape((ne, 3)) - 1
    return Grid(proj, x, y, regular=False, triangles=triangles)


def write_grid(path, **kwargs):
    """
    write updated mesh back to mesh file

    Note: now we assume that number of mesh elements and their indices doesn't change!
    only updating the mesh node position x,y
    """
    meshfile = filename(path, **kwargs).replace('field', 'mesh')

    write_data(meshfile, 'Nodes_x', grid.x)
    write_data(meshfile, 'Nodes_y', grid.y)


def read_grid_from_msh(mshfile):
    """
    get the grid object directly from .msh definition file
    this function is uniq to nextsim, not required by assim_tools
    """
    info = read_mshfile(mshfile)
    x = info['nodes_x']
    y = info['nodes_y']
    triangles = np.array([np.array(el.node_indices) for el in info['triangles']])
    return Grid(proj, x, y, regular=False, triangles=triangles)


def read_var(path, grid, **kwargs):
    """read native variable defined on native grid from model restart files"""
    ##check name in kwargs and read the variables from file
    vname = kwargs['name']
    assert vname in variables, 'variable name '+vname+' not listed in variables'
    fname = filename(path, **kwargs)

    var = read_data(fname, variables[vname]['name'])

    ##nextsim restart file concatenates u,v component, so reshape if is_vector
    if variables[vname]['is_vector']:
        var = var.reshape((2, -1))

    ##convert units if native unit is not the same as required by kwargs
    if 'units' in kwargs:
        units = kwargs['units']
    else:
        units = variables[vname]['units']
    var = units_convert(units, variables[vname]['units'], var)

    return var


def write_var(path, grid, var, **kwargs):
    """write native variable back to a model restart file"""
    fname = filename(path, **kwargs)

    ##check name in kwargs and read the variables from file
    assert 'name' in kwargs, 'please specify which variable to write, name=?'
    vname = kwargs['name']
    assert vname in variables, "variable name "+vname+" not listed in variables"

    ##nextsim restart file concatenates u,v component, so flatten if is_vector
    if kwargs['is_vector']:
        var = var.flatten()

    ##convert units back if necessary
    var = units_convert(kwargs['units'], variables[vname]['units'], var, inverse=True)

    ##output the var to restart file
    write_data(fname, variables[vname]['name'], var)


##since all nextsim variables are at surface, there is no need for z_coords calculation
uniq_z_key = ()

def z_coords(path, grid, **kwargs):
    ##for nextsim, just discard inputs and simply return zero as z_coords
    return np.zeros(grid.x.shape)


def read_param(path, **kwargs):
    param = 0
    return param


def write_param(path, param, **kwargs):
    pass


