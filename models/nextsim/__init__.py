import numpy as np
import glob

from assim_tools.state_def import variables, units_convert
from .bin_io import get_info, read_data, write_data

##the map projection used in nextsim
from .gmshlib import read_mshfile, proj
from grid import Grid

##nextsim use moving mesh, so it varies for each member and time
uniq_grid = ('member', 'time')

def get_grid(path, **kwargs):
    meshfile = filename(path, **kwargs).replace('field', 'mesh')
    x = read_data(meshfile, 'Nodes_x')
    y = read_data(meshfile, 'Nodes_y')
    elements = read_data(meshfile, 'Elements')
    ne = int(elements.size/3)
    triangles = elements.reshape((ne, 3)) - 1
    return Grid(proj, x, y, regular=False, triangles=triangles)

def get_grid_from_msh(mshfile):
    info = read_mshfile(mshfile)
    x = info['nodes_x']
    y = info['nodes_y']
    triangles = np.array([np.array(el.node_indices) for el in info['triangles']])
    return Grid(proj, x, y, regular=False, triangles=triangles)

##convert NEDAS variables to nextsim variable names and units
##Note: we only work with restart files
##     normal nextsim binfile have some variables names different
##     from restart files, e.g. Concentration instead of M_conc
var_dict = {'seaice_conc': {'name':'M_conc', 'units':'%'},
            'seaice_thick': {'name':'M_thick', 'units':'m'},
            'seaice_damage': {'name':'M_damage', 'units':'%'},
            'snow_thick': {'name':'M_snow_thick', 'units':'m'},
            'seaice_velocity': {'name':'M_VT', 'units':'m/s'},
            'seaice_drift': {'name':'M_UT', 'units':'m'},
           }

def filename(path, **kwargs):
    if 'time' in kwargs:
        tstr = kwargs['time'].strftime('%Y%m%dT%H%M%SZ')
    else:
        tstr = '*'
    if 'member' in kwargs:
        mstr = '{:03d}'.format(kwargs['member']+1)
    else:
        mstr = ''
    flist = glob.glob(path+'/'+mstr+'/field_'+tstr+'.bin')
    assert len(flist)>0, 'no matching files found'
    return flist[0]

def get_var(path, grid, **kwargs):
    fname = filename(path, **kwargs)

    assert 'name' in kwargs, 'please specify which variable to get, name=?'
    var_name = kwargs['name']
    assert var_name in var_dict, "variable name "+var_name+" not listed in var_dict"

    var = read_data(fname, var_dict[var_name]['name'])

    if variables[var_name]['is_vector']:
        var = var.reshape((2, -1))

    var = units_convert(variables[var_name]['units'], var_dict[var_name]['units'], var)

    return var

def write_var(path, grid, var, **kwargs):
    pass

def write_grid(path, grid, **kwargs):
    pass
