##Standard functions called by assim_tools with **kwargs as input parameters
##to obtain a 2D field from model states.
##List of functions:
##   filename: parse arguments and generate name of model state files
##   read_grid, write_grid: read/write of model mesh from/to the mesh file
##   read_var, write_var: read/write of model state variables from/to the restart file

##Common function inputs:
##   path: directory where the model restart files are stored
##   grid: Grid object from read_grid

##List of keys in **kwargs:
##   name: variable name defined in state_def
##   t, dt: datetime obj, and duration in hours, defining the time window for the state
##   z, dz: vertical level index, and interval, defining the vertical layer for the state
##   member: index for the ensemble member (0 to nens-1) from which the state is obtained,
##           if is None, then obtain the state from a deterministic run

import numpy as np
import glob
from functools import lru_cache
from datetime import datetime
from netCDF4 import Dataset

from conversion import units_convert
from models.topaz.confmap import ConformalMapping
from models.topaz.abfile import ABFileRestart, ABFileArchv, ABFileBathy, ABFileGrid, ABFileForcing

##dictionary from state_def names to native variable names and properties
##Note: topaz file has several conventions for units, we only deal with
##        restart files here. (check with expert first if you attempt
##       to use this for other file types, daily mean, archiv etc.)
##List of properties:
##   name: native variable name in restart files, tuple of (u-name,v-name)
##         if vector field components are stored in separate native variables
##   dtype: double/flout/int
##   is_vector: if true the variable contains (u, v) components
##   restart_dt: how freq model output is available, in hours
##   levels: vertical level index list
##   units: native physical units for the variable
levels = np.arange(1, 51, 1)  ##ocean levels, from top to bottom, k=1..nz
level_sfc = np.array([0])    ##some variables are only defined on surface level k=0
variables = {'ocean_velocity':    {'name':('u', 'v'), 'dtype':'float', 'is_vector':True,  'restart_dt':168, 'levels':levels, 'units':'m/s'},
             'ocean_layer_thick': {'name':'dp',       'dtype':'float', 'is_vector':False, 'restart_dt':168, 'levels':levels, 'units':'Pa'},
             'ocean_temp':        {'name':'temp',     'dtype':'float', 'is_vector':False, 'restart_dt':168, 'levels':levels, 'units':'K'},
             'ocean_saln':        {'name':'saln',     'dtype':'float', 'is_vector':False, 'restart_dt':168, 'levels':levels, 'units':'psu'},
             'ocean_b_velocity':  {'name':('ubavg', 'vbavg'), 'dtype':'float', 'is_vector':True, 'restart_dt':168, 'levels':level_sfc, 'units':'m/s'},
             'ocean_b_press':     {'name':'pbavg',    'dtype':'float', 'is_vector':False, 'restart_dt':168, 'levels':level_sfc, 'units':'Pa'},
             'ocean_mixl_depth':  {'name':'dpmixl',   'dtype':'float', 'is_vector':False, 'restart_dt':168, 'levels':level_sfc, 'units':'Pa'},
             'seaice_velocity':   {'name':('uvel', 'vvel'), 'dtype':'float', 'is_vector':True, 'restart_dt':168, 'levels':level_sfc, 'units':'m/s'},
             'seaice_conc_cat1':  {'name':'aicen', 'dtype':'float', 'is_vector':False, 'restart_dt':168, 'levels':level_sfc, 'units':'%'},
             'seaice_thick':      {'name':'vicen', 'dtype':'float', 'is_vector':False, 'restart_dt':168, 'levels':level_sfc, 'units':'m'},
             }

##some constants
ONEM = 9806.  ##pressure (Pa) in 1 meter water

def filename(path, **kwargs):
    """
    Parse kwargs and find matching filename
    for keys in kwargs that are not set, here we define the default values
    key values in kwargs will also be checked for erroneous values here
    """
    if 'time' in kwargs and kwargs['time'] is not None:
        assert isinstance(kwargs['time'], datetime), 'time shall be a datetime object'
        tstr = kwargs['time'].strftime('%Y_%j_%H_0000')
    else:
        tstr = '????_???_??_0000'
    if 'member' in kwargs and kwargs['member'] is not None:
        assert kwargs['member'] >= 0, 'member index shall be >= 0'
        mstr = '_mem{:03d}'.format(kwargs['member'])
    else:
        mstr = ''

    ##get a list of filenames with matching kwargs
    search = path+'/'+'TP5restart.'+tstr+mstr+'.a'
    flist = glob.glob(search)
    assert len(flist)>0, 'no matching files found: '+search
    ##typically there will be only one matching file given the kwargs,
    ##if there is a list of matching files, then we return the first one
    return flist[0]


from pyproj import Geod
from grid import Grid

##some variables will need (de)staggering in topaz grid:
##---                *--*--*
##---                |  |  |
##--- stencil:       u--p--*
##---                |  |  |
##---                q--v--*
##these two functions are uniq to topaz
def stagger(dat, v_name):
    ##stagger u,v for C-grid configuration
    dat_ = dat.copy()
    if v_name == 'u':
        dat_[:, 1:] = 0.5*(dat[:, :-1] + dat[:, 1:])
        dat_[:, 0] = 3*dat[:, 1] - 3*dat[:, 2] + dat[:, 3]
    elif v_name == 'v':
        dat_[1:, :] = 0.5*(dat[:-1, :] + dat[1:, :])
        dat_[0, :] = 3*dat[1, :] - 3*dat[2, :] + dat[3, :]
    return dat_


def destagger(dat, v_name):
    ##destagger u,v from C-grid
    dat_ = dat.copy()
    if v_name == 'u':
        dat_[:, :-1] = 0.5*(dat[:, :-1] + dat[:, 1:])
        dat_[:, -1] = 3*dat[:, -2] - 3*dat[:, -3] + dat[:, -4]
    elif v_name == 'v':
        dat_[:-1, :] = 0.5*(dat[:-1, :] + dat[1:, :])
        dat_[-1, :] = 3*dat[-2, :] - 3*dat[-3, :] + dat[-4, :]
    return dat_


##keys in kwargs for which the grid obj needs to be redefined: ('member', 'time', 'k')
##topaz grid is fixed in time/space, so no keys needed
uniq_grid_key = ()

def read_grid(path, **kwargs):
    """
    Parse grid.info and generate grid.Grid object
    kwargs here are dummy input since the grid is fixed
    """
    grid_info_file = path+'/topo/grid.info'
    cm = ConformalMapping.init_from_file(grid_info_file)
    nx = cm._ires
    ny = cm._jres

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    lat, lon = cm.gind2ll(ii+1., jj+1.)

    ##find grid resolution
    geod = Geod(ellps='sphere')
    _,_,dist_x = geod.inv(lon[:,1:], lat[:,1:], lon[:,:-1], lat[:,:-1])
    _,_,dist_y = geod.inv(lon[1:,:], lat[1:,:], lon[:-1,:], lat[:-1,:])
    dx = np.median(dist_x)
    dy = np.median(dist_y)

    ##the coordinates in topaz model native grid
    x = ii*dx
    y = jj*dy

    ##proj function that mimic what pyproj.Proj object does to convert x,y to lon,lat
    def proj(x, y, inverse=False):
        if not inverse:
            i, j = cm.ll2gind(y, x)
            xo = (i-1)*dx
            yo = (j-1)*dy
        else:
            i = np.atleast_1d(x/dx + 1)
            j = np.atleast_1d(y/dy + 1)
            yo, xo = cm.gind2ll(i, j)
        if xo.size == 1:
            return xo.item(), yo.item()
        return xo, yo

    return Grid(proj, x, y)


def write_grid(path, **kwargs):
    pass


##topaz stored a separate landmask in depth.a file
def read_mask(path, grid, **kwargs):
    depthfile = path+'/topo/depth.a'
    f = ABFileBathy(depthfile, 'r', idm=grid.nx, jdm=grid.ny)
    depth = f.read_field('depth')
    f.close()
    return (depth==0)


def read_depth(path, grid):
    depthfile = path+'/topo/depth.a'
    f = ABFileBathy(depthfile, 'r', idm=grid.nx, jdm=grid.ny)
    depth = f.read_field('depth').data
    f.close()
    return -depth


def read_var(path, grid, **kwargs):
    """
    Read the state variable from a model restart file

    Inputs:
    - path: str
      Path to the directory where the restart files are stored

    - grid: Grid obj
      Describes the model native grid, from read_grid()

    - **kwargs: time, level, and member to pinpoint where to get the variable

    Return:
    - var: np.array
      a 2D field defined on grid with the state variable
    """
    ##check name in kwargs and read the variables from file
    assert 'name' in kwargs, 'please specify which variable to get, name=?'
    name = kwargs['name']
    assert name in variables, 'variable name '+name+' not listed in variables'
    fname = filename(path, **kwargs)

    if 'k' in kwargs:
        k = kwargs['k']
    else:
        k = variables[name]['levels'][0]  ##get the first level if not specified
    if 'mask' in kwargs:
        mask = kwargs['mask']
    else:
        mask = None

    if 'is_vector' in kwargs:
        is_vector = kwargs['is_vector']
    else:
        is_vector = variables[name]['is_vector']

    if 'units' in kwargs:
        units = kwargs['units']
    else:
        units = variables[name]['units']

    f = ABFileRestart(fname, 'r', idm=grid.nx, jdm=grid.ny)
    if is_vector:
        var1 = f.read_field(variables[name]['name'][0], level=k, tlevel=1, mask=mask)
        var2 = f.read_field(variables[name]['name'][1], level=k, tlevel=1, mask=mask)
        var = np.array([var1, var2])
    else:
        var = f.read_field(variables[name]['name'], level=k, tlevel=1, mask=mask)
    f.close()

    var = units_convert(units, variables[name]['units'], var)

    return var


def write_var(path, grid, var, **kwargs):
    """
    Write a variable (overwrite) to the model restart file
    """
    ##check name in kwargs
    assert 'name' in kwargs, 'please specify which variable to write, name=?'
    name = kwargs['name']
    assert name in variables, 'variable name '+name+' not listed in variables'
    fname = filename(path, **kwargs)

    ##same logic for setting level indices as in read_var()
    if 'k' in kwargs:
        k = kwargs['k']
    else:
        k = variables[name]['levels'][0]
    if 'mask' in kwargs:
        mask = kwargs['mask']
    else:
        mask = None

    ##open the restart file for over-writing
    ##the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile
    f = ABFileRestart(fname, 'r+', idm=grid.nx, jdm=grid.ny, mask=True)

    ##convert units back if necessary
    var = units_convert(kwargs['units'], variables[name]['units'], var, inverse=True)

    if kwargs['is_vector']:
        for i in range(2):
            f.overwrite_field(var[i,...], mask, variables[name]['name'][i], level=k, tlevel=1)
    else:
        f.overwrite_field(var, mask, variables[name]['name'], level=k, tlevel=1)
    f.close()


##keys in kwargs for which the z_coords needs to be separately calculated.
##for topaz, the isopycnal coordinates vary for member, time, and level k
uniq_z_key = ('member', 'time', 'k')
z_units = 'm'

@lru_cache(maxsize=3)
def z_coords(path, grid, **kwargs):
    """
    Calculate vertical coordinates given the 3D model state

    Inputs:
    - path, grid, **kwargs same as filename() inputs

    Return:
    - z: np.array
      The corresponding z field
    """
    ##some defaults if not set in kwargs
    if 'units' not in kwargs:
        kwargs['units'] = z_units
    if 'k' not in kwargs:
        kwargs['k'] = 0

    z = np.zeros(grid.x.shape)

    if kwargs['k'] == 0:
        ##if level index is 0, this is the surface, so just return zeros
        return z

    else:
        ##get layer thickness and convert to units
        rec = kwargs.copy()
        rec['name'] = 'ocean_layer_thick'
        rec['units'] = variables['ocean_layer_thick']['units'] ##should be Pa
        if kwargs['units'] == 'm':
            dz = - read_var(path, grid, **rec) / ONEM ##in meters, negative relative to surface
        elif kwargs['units'] == 'Pa':
            dz = read_var(path, grid, **rec)
        else:
            raise ValueError('do not know how to calculate z_coords for z_units = '+kwargs['units'])

        ##use recursive func, get previous layer z and add dz
        kwargs['k'] -= 1
        z_prev = z_coords(path, grid, **kwargs)
        return z_prev + dz


