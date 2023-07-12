from assim_tools import variables, units_convert
from .abfile import ABFileRestart, ABFileArchv, ABFileBathy, ABFileGrid
from .grid import get_grid
import numpy as np
import glob

##convert NEDAS variables to topaz restart .ab files native names and units
##Note: topaz file has several conventions for units, we only deal with
##       snapshot restart files here. (check with expert first if you attempt
##       to use this for other file types, daily mean, archiv etc.)
var_dict = {'ocean_velocity': {'name':('u', 'v'), 'nz':50, 'units':'m/s'},
            'ocean_layer_depth': {'name':'dp', 'nz':50, 'units':'m'},
            'ocean_temp': {'name':'temp', 'nz':50, 'units':'K'},
            'ocean_salinity': {'name':'saln', 'nz':50, 'units':'psu'},
            'ocean_surf_height': {'name':'ssh', 'nz':0, 'units':'m'},
            }

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

def filename(path, **kwargs):
    if 'time' in kwargs:
        time = kwargs['time']
        tstr = time.strftime('%Y_%j_%H')
    else:
        tstr = '????_???_??'
    if 'member' in kwargs:
        member = kwargs['member']
    else:
        member = 0

    search = path+'/'+'TP4restart'+tstr+'_mem{:03d}'.format(member+1)+'.a'
    flist = glob.glob(search)
    assert len(flist)>0, 'no matching files found: '+search
    return flist[0]

def get_mask(path, grid):
    depthfile = path+'/topo/depth.a'
    f = ABFileBathy(depthfile, 'r', idm=grid.nx, jdm=grid.ny)
    mask = f.read_field('depth').mask
    f.close()
    return mask

def get_var(path, grid, **kwargs):
    fname = filename(path, **kwargs)

    assert 'name' in kwargs, 'please specify which variable to get, name=?'
    var_name = kwargs['name']
    assert var_name in var_dict, 'variable name '+var_name+' not listed in var_dict'

    if 'level' in kwargs:
        level = kwargs['level']
    else:
        if var_dict[var_name]['nz'] == 0:
            level = 0
        else:
            level = 1
    if 'mask' in kwargs:
        mask = kwargs['mask']
    else:
        mask = None

    f = ABFileRestart(fname, 'r', idm=grid.nx, jdm=grid.ny)
    if variables[var_name]['is_vector']:
        var1 = f.read_field(var_dict[var_name]['name'][0], level=level, tlevel=1, mask=mask)
        var2 = f.read_field(var_dict[var_name]['name'][1], level=level, tlevel=1, mask=mask)
        var = np.array([var1, var2])
    else:
        var = f.read_field(var_dict[var_name]['name'], level=level, tlevel=1, mask=mask)
    f.close()

    var = units_convert(variables[var_name]['units'], var_dict[var_name]['units'], var)
    return var


def write_var(path, grid, var, **kwargs):
    fname = filename(path, **kwargs)

    if 'mask' in kwargs:
        mask = kwargs['mask']
    else:
        mask = None

    f = ABFileRestart(fname, 'r+', idm=grid.nx, jdm=grid.ny, mask=True)

    var = units_convert(variables[var_name]['units'], var_dict[var_name]['units'], var, inverse=True)

    if variables[var_name]['is_vector']:
        for i in range(2):
            f.overwrite_field(var[i,...], mask, var_dict[var_name]['name'][i], level=level, tlevel=1)
    else:
        f.overwrite_field(var, mask, var_dict[var_name]['name'], level=level, tlevel=1)
    f.close()
