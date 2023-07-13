import numpy as np
import os
import importlib
from datetime import datetime, timedelta
from assim_tools import divide_load, field_info, write_field_info, write_mask, write_field

##state variable definition
##available state variable names, and their properties
variables = {'atmos_surf_wind': {'is_vector':True, 'units':'m/s'},
             'atmos_surf_temp': {'is_vector':False, 'units':'K'},
             'atmos_surf_dew_temp': {'is_vector':False, 'units':'K'},
             'atmos_surf_press': {'is_vector':False, 'units':'Pa'},
             'atmos_precip': {'is_vector':False, 'units':'kg/m2/s'},
             'atmos_snowfall': {'is_vector':False, 'units':'kg/m2/s'},
             'atmos_down_shortwave': {'is_vector':False, 'units':'W/m2'},
             'atmos_down_longwave': {'is_vector':False, 'units':'W/m2'},
             'seaice_conc': {'is_vector':False, 'units':'%'},
             'seaice_thick': {'is_vector':False, 'units':'m'},
             'snow_thick': {'is_vector':False, 'units':'m'},
             'seaice_drift': {'is_vector':True, 'units':'m'},
             'seaice_velocity': {'is_vector':True, 'units':'m/s'},
             'seaice_damage': {'is_vector':False, 'units':'%'},
             'ocean_surf_temp': {'is_vector':False, 'units':'K'},
             'ocean_surf_velocity': {'is_vector':True, 'units':'m/s'},
             'ocean_surf_height': {'is_vector':False, 'units':'m'},
             'ocean_temp': {'is_vector':False, 'units':'K'},
             'ocean_salinity': {'is_vector':False, 'units':'%'},
             'ocean_velocity': {'is_vector':True, 'units':'m/s'},
             }

###units converter
##units: target units used in state variables (defined in variables)
##s_units: source units used in models (defined in var_dict in each module)
def units_convert(units, s_units, var, inverse=False):
    ##  list of available units and methods to convert from/to s_units
    unit_from = {'m/s':     {'km/h':lambda x: x/3.6,
                             'km/day':lambda x: x/86.4,
                            },
                 'm':       {'cm':lambda x: x/100.,
                            },
                 'K':       {'C':lambda x: x+273.15,
                             'F':lambda x: (x-32)*5./9.+273.15,
                            },
                 'Pa':      {'hPa':lambda x: x*100.,
                            },
                 'kg/m2/s': {'Mg/m2/3h':lambda x: x/3/3.6,
                            },
                }
    unit_to   = {'m/s':     {'km/h':lambda x: x*3.6,
                             'km/day':lambda x: x*86.4,
                            },
                 'm':       {'cm':lambda x: x*100.,
                            },
                 'K':       {'C':lambda x: x-273.15,
                             'F':lambda x: (x-273.15)*9./5.+32.,
                            },
                 'Pa':      {'hPa':lambda x: x/100.,
                            },
                 'kg/m2/s': {'Mg/m2/3h':lambda x: x*3*3.6,
                            },
                }
    if units != s_units:
        if s_units not in unit_from[units]:
            raise ValueError("cannot find convert method from "+s_units+" to "+units)
        else:
            if not inverse:
                var = unit_from[units][s_units](var)
            else:
                var = unit_to[units][s_units](var)
    return var

##fill any missing value in fld that is not in masked area
# def fill_missing(fld, mask):

    # return fld

##mask: area in the reference grid that is land or other area that doesn't require analysis,
##      2D fields will have NaN in those area, bin files will only store the ~mask region,
##      the analysis routine will also skip if mask
def prepare_mask(c):
    if c.MASK_FROM = 'nextsim':
        from models import nextsim
        grid = nextsim.get_grid_from_msh(c.MESH_FILE)
        mshgrid.dst_grid = c.ref_grid
        mask  = np.isnan(grid.convert(grid.x))
    ##other options...
    ##save mask to file
    np.save(c.WORK_DIR+'/mask.npy', mask)

##parse state_def, generate field_info,
##then read model output and write 2D fields into bin files:
##   state[nens, nfield, ny, nx], nfield dimension contains nv,nt,nz flattened
##   nv is number of variables, nt is time slices, nz is vertical layers,
##   of course nt,nz vary for each variables, so we stack them in nfield dimension
def prepare_state(c, comm, time):
    ##c: config module
    ##comm: mpi4py communicator
    ##time: analysis time (datetime obj)
    ny, nx = c.ref_grid.x.shape
    mask = np.load(c.WORK_DIR+'/mask.npy')
    binfile = c.WORK_DIR+'/prior.bin' ##TODO: timestr

    if comm.Get_rank() == 0:
        ##generate field info from state_def
        info = field_info(c.STATE_DEF_FILE,
                        current_time,
                        (c.OBS_WINDOW_MIN, c.OBS_WINDOW_MAX),
                        np.arange(c.ZI_MIN, c.ZI_MAX+1),
                        c.NUM_ENS,
                        *c.ref_grid.x.shape, mask)
        write_field_info(binfile, info)
        write_mask(binfile, info, mask)
    else:
        info = None
    info = comm.bcast(info, root=0)

    grid_bank = {}
    for i in divide_load(comm, np.arange(len(info['fields']))):
        rec = info['fields'][i]
        v = rec['var_name']
        t = rec['time']
        m = rec['member']
        z = rec['level']

        ##directory storing model output
        path = c.WORK_DIR + '/models/' + rec['source'] ##+ '/' + current_time.strftime('%Y%m%dT%H%M')

        ##load the module for handling source model
        src = importlib.import_module('models.'+rec['source'])

        ##only need to compute the uniq grids, stored them in bank for later use
        grid_key = (rec['source'],)
        for key in src.uniq_grid:
            grid_key += (rec[key],)
        if grid_key in grid_bank:
            grid = grid_bank[grid_key]
        else:
            grid = src.get_grid(path, name=v, member=m, time=t, level=z)
            grid.dst_grid = c.ref_grid
            grid_bank[grid_key] = grid

        var = src.get_var(path, grid, name=v, member=m, time=t, level=z)
        fld = grid.convert(var, is_vector=rec['is_vector'])

        write_field(binfile, info, mask, i, fld)

