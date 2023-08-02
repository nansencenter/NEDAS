##Standard functions called by assim_tools with **kwargs as input parameters
##to obtain a 2D field from model states.
##List of functions:
##   filename: parse arguments and generate name of model state files
##   get_grid, write_grid: read/write of model mesh from/to the mesh file
##   get_var, write_var: read/write of model state variables from/to the restart file
##   z_coords: recipe for obtaining z coordinates from model states
##   get_obs_var: compute values corresponding to the observation variables (the obs prior)
##Common function inputs:
##   path: directory where the model restart files are stored
##   grid: Grid object from get_grid
##List of keys in **kwargs:
##   name: variable name defined in assim_tools.state_variables
##   time: datetime obj for the valid time for the state
##   level: vertical level index for the state
##   member: index for the ensemble member (0 to nens-1) from which the state is obtained,
##           if is None, then obtain the state from a deterministic run

import numpy as np
import glob
from datetime import datetime

from assim_tools import state_variables, units_convert
from .bin_io import read_data, write_data

##dictionary from assim_tools.state_variables to native variable names and properties
##Note: we only work with restart files, normal nextsim binfile have some variables names that
##are different from restart files, e.g. Concentration instead of M_conc
##List of properties:
##   name: native variable name in restart files, tuple of (u-name,v-name) if vector field
##         components are stored in separate native variables
##   levels: list of available vertical level indices from model output (0 indicates single layer)
##   units: physical units for the native variables
levels = [0]  ##nextsim only has surface variables
var_dict = {'seaice_conc': {'name':'M_conc', 'levels':levels, 'units':'%'},
            'seaice_thick': {'name':'M_thick', 'levels':levels, 'units':'m'},
            'seaice_damage': {'name':'M_damage', 'levels':levels, 'units':'%'},
            'snow_thick': {'name':'M_snow_thick', 'levels':levels, 'units':'m'},
            'seaice_drift': {'name':'M_VT', 'levels':levels, 'units':'m/s'},
           }

##parse kwargs and find matching filename
##for keys in kwargs that are not set, here we define the default values
##key values in kwargs will also be checked for erroneous values here
def filename(path, **kwargs):
    if 'time' in kwargs:
        assert isinstance(kwargs['time'], datetime), 'time shall be a datetime object'
        tstr = kwargs['time'].strftime('%Y%m%dT%H%M%SZ')
    else:
        tstr = '*'
    if 'member' in kwargs:
        assert kwargs['member'] >= 0, 'member index shall be >= 0'
        mstr = '{:03d}'.format(kwargs['member']+1)
    else:
        mstr = ''
    if 'name' in kwargs:
        assert kwargs['name'] in var_dict, 'variable name is not defined in var_dict'
        if 'level' in kwargs:
            assert kwargs['level'] in var_dict[kwargs['name']]['levels'], 'level index is not available'

    ##get a list of filenames with matching kwargs
    search = path+'/'+mstr+'/field_'+tstr+'.bin'
    flist = glob.glob(search)
    assert len(flist)>0, 'no matching files found: '+search
    ##typically there will be only one matching file given the kwargs,
    ##if there is a list of matching files, then we return the first one
    return flist[0]

##the map projection used in nextsim is defined in gmshlib:
from .gmshlib import read_mshfile, proj

from grid import Grid

##keys in kwargs for which the grid obj needs to be redefined
##nextsim use moving mesh, so it varies for each member and time
uniq_grid = ('member', 'time')

##read grid from model mesh.bin file, returns a grid obj
def get_grid(path, **kwargs):
    meshfile = filename(path, **kwargs).replace('field', 'mesh')
    x = read_data(meshfile, 'Nodes_x')
    y = read_data(meshfile, 'Nodes_y')
    elements = read_data(meshfile, 'Elements')
    ne = int(elements.size/3)
    triangles = elements.reshape((ne, 3)) - 1
    return Grid(proj, x, y, regular=False, triangles=triangles)

##get the grid object directly from .msh definition file
##this function is uniq to nextsim, not required by assim_tools
def get_grid_from_msh(mshfile):
    info = read_mshfile(mshfile)
    x = info['nodes_x']
    y = info['nodes_y']
    triangles = np.array([np.array(el.node_indices) for el in info['triangles']])
    return Grid(proj, x, y, regular=False, triangles=triangles)

##get state variable with name='varname' defined in assim_tools.state_variables
##and other kwargs: time, level, and member to pinpoint where to get the variable
##returns a 2D field defined on grid from get_grid
def get_var(path, grid, **kwargs):
    ##check name in kwargs and read the variables from file
    assert 'name' in kwargs, 'please specify which variable to get, name=?'
    var_name = kwargs['name']
    assert var_name in var_dict, 'variable name '+var_name+' not listed in var_dict'
    fname = filename(path, **kwargs)

    var = read_data(fname, var_dict[var_name]['name'])

    ##nextsim restart file concatenates u,v component, so reshape if is_vector
    if state_variables[var_name]['is_vector']:
        var = var.reshape((2, -1))

    ##convert units if model units (var_dict['units']) is not the same as
    ##defined in assim_tools.state_variables['units']
    var = units_convert(state_variables[var_name]['units'], var_dict[var_name]['units'], var)

    return var

def write_grid(path, grid, **kwargs):
    ##Note: now we assume that number of mesh elements and their indices doesn't change!
    ##only updating the mesh node position x,y
    meshfile = filename(path, **kwargs).replace('field', 'mesh')

    write_data(meshfile, 'Nodes_x', grid.x)
    write_data(meshfile, 'Nodes_y', grid.y)

def write_var(path, grid, var, **kwargs):
    fname = filename(path, **kwargs)

    ##check name in kwargs and read the variables from file
    assert 'name' in kwargs, 'please specify which variable to write, name=?'
    var_name = kwargs['name']
    assert var_name in var_dict, "variable name "+var_name+" not listed in var_dict"

    ##nextsim restart file concatenates u,v component, so flatten if is_vector
    if state_variables[var_name]['is_vector']:
        var = var.flatten()

    ##convert units back if not the same as defined in state_variables
    var = units_convert(state_variables[var_name]['units'], var_dict[var_name]['units'], var, inverse=True)

    ##output the var to restart file
    write_data(fname, var_dict[var_name]['name'], var)

##keys in kwargs for which the z_coords needs to be separately calculated
##since all nextsim variables are at surface, there is no need for z_coords calculation
uniq_z = ()

##calculate vertical coordinates given the 3D model state
##inputs: path, grid, **kwargs: same as filename() inputs
##        z_units: the output units for z_coords ('m' if height/depth, 'Pa' if pressure, etc)
def z_coords(path, grid, z_units, **kwargs):
    ##for nextsim, just discard inputs and simply return zero as z_coords
    return np.zeros(grid.x.shape)


##get obs prior
def get_obs_var(path, grid, **kwargs):
    var_name = kwargs['name']
    if var_name in var_dict:
        return get_var(path, grid, **kwargs)
    elif var_name in obs_op_dict:
        return obs_op_dict[var_name](path, grid, **kwargs)
    else:
        raise ValueError('observation '+var_name+' is not available from nextsim')

