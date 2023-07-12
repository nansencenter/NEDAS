from assim_tools import variables, units_convert
from .basic_io import read_data
import glob

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

levels = [0]

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

def get_var(path, **kwargs):
    fname = filename(path, **kwargs)

    assert 'name' in kwargs, 'please specify which variable to get, name=?'
    var_name = kwargs['name']
    assert var_name in var_dict, "variable name "+var_name+" not listed in var_dict"

    var = read_data(fname, var_dict[var_name]['name'])

    if variables[var_name]['is_vector']:
        var = var.reshape((2, -1))

    var = units_convert(variables[var_name]['units'], var_dict[var_name]['units'], var)

    return var

def write_var(path, var, **kwargs):
    pass
