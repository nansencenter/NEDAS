from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta # type: ignore
import os
import warnings
import typing

import cftime # type: ignore
import netCDF4 # type: ignore
import numpy as np
import pyproj # type: ignore

from utils.conversion import t2s
from grid import Grid
from perturb import gen_perturb, apply_perturb, pres_adjusted_wind_perturb, apply_AR1_perturb


_proj:pyproj.Proj = pyproj.Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)

def read_var(fname:str, varnames:list[str]) -> np.ma.MaskedArray:
    """reading a variable from a netcdf file

    Parameters
    ----------
    fname : str
        forcing file name
    varnames : list[str]
        list of variable names
    itime : int
        time index in the forcing file
    """
    with netCDF4.Dataset(fname, 'r') as f:
        # read the variable
        data: list[np.ndarray] = []
        for vname in varnames:
            data.append(f[vname][:])
    return np.ma.array(data)


def write_var(fname:str, varnames: list[str], data: np.ndarray) -> None:
    """Write the perturbed variable back to the forcing file

    Parameters
    ----------
    fname : str
        forcing file name
    varnames : list[str]
        list of variable names
    itime : int
        time index in the forcing file
    """
    # We assume all variables in the forcing file exists
    assert os.path.exists(fname), f'{fname} does not exist; Please copy the forcing file to the correct path first.'
    with netCDF4.Dataset(fname, 'r+') as f:
        for i, vname in enumerate(varnames):
            f[vname][:] = data[i]


def perturb_restart(restart_options:dict, i_ens: int, time: datetime) -> None:
    """perturb the initial conditions in restart files

    Parameters
    ----------
    restart_options : dict
        perturbation options under the section of `perturb` from the yaml file
    i_ens : int
        ensemble index
    time : datetime
        current time
    prev_time : datetime
        previous time
    """

    # perturbation arrays
    pert: np.ndarray[typing.Any, np.dtype[np.float64]]
    # get the restart file name
    file_options = restart_options['file']
    fname:str = file_options['format'].format(i=i_ens, time=time.strftime(file_options['time_format']))
    # get grid object
    with netCDF4.Dataset(fname, 'r') as f:
        grid = Grid(_proj, *_proj(f[file_options['lon_name']], f[file_options['lat_name']]
                                )
                    )

    # get options for perturbing the forcing variables
    options = restart_options['variables']
    for i, varname in enumerate(options['names']):
        if typing.TYPE_CHECKING:
            assert type(varname) == str, 'variable name must be a string'

        # convert the horizontal correlation length scale to grid points
        hcorr:int = np.rint(float(options['hcorr'][i])/grid.dx)
        # do perturbation
        pert = gen_perturb(grid, options['type'][i], float(options['amp'][i]), hcorr)

        # apply perturbations to the variable data
        # in the case of vector fields, we need to split the variable name
        varname_list: list[str] = varname.split(';')
        # read the variable data from forcing file
        data: np.ndarray = read_var(fname, varname_list)
        # apply perturbations to the variable data
        data = apply_perturb(grid, data, pert, options['type'][i])
        # apply lower and upper bounds
        lb = float(options['lower_bounds'][i]) if options['lower_bounds'][i] != 'None' else -np.inf
        ub = float(options['upper_bounds'][i]) if options['upper_bounds'][i] != 'None' else np.inf
        data = np.minimum(np.maximum(data, lb), ub)
        # write the perturbed variable back to the forcing file
        write_var(fname, varname_list, data)
