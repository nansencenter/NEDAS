from datetime import datetime
import os
import typing
import threading

import netCDF4 # type: ignore
import numpy as np
import pyproj # type: ignore

from NEDAS.grid import Grid
from NEDAS.models.nextsim.dg.perturb import gen_perturb, apply_perturb

_proj:pyproj.Proj = pyproj.Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)

thread_lock = threading.Lock()
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
    data: list[np.ndarray] = []
    with thread_lock:
        with netCDF4.Dataset(fname, 'r') as f:
            # read the variable
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
    with thread_lock:
        with netCDF4.Dataset(fname, 'r+') as f:
            for i, vname in enumerate(varnames):
                f[vname][:] = data[i]
            f.sync()


def get_restart_filename(file_options:dict, i_ens: int, time: datetime) -> str:
    # get the restart file name
    fname: str
    try:
        fname = file_options['format'].format(i=i_ens, time=time.strftime(file_options['time_format']))
    except KeyError:
        try:
            fname = file_options['format'].format(i=i_ens, time=time.strftime(file_options['time_format']))
        except KeyError:
            print ('Currently, we only supports keyword of 1. "time_format",'
                   '2. "time_format"+"i".'
                   'See the example yaml file for more information. '
                   'Modified the code if you have other requirements.')
    return fname


def perturb_restart(restart_options:dict, file_options:dict, debug=False) -> None:
    """perturb the initial conditions in restart files

    Parameters
    ----------
    restart_options : dict
        perturbation options under the section of `perturb` from the yaml file
    file_options : dict
        This dictionary is constructed before the function is called.
        It contains the following keys
        - fname : str
            forcing file name.
            This has to be the file that will be perturbed, e.g. in the ensemble directory.
            This is usually derived before it is called.
        - lon_name: str
            name of the longitude variable in the forcing file.
            This is obtained from the files/restart section of the model configuration file.
        - lat_name: str
            name of the latitude variable in the forcing file
            This is obtained from the files/restart section of the model configuration file.
    """

    # perturbation arrays
    pert: np.ndarray[typing.Any, np.dtype[np.float64]]
    # get the restart file name
    fname:str = file_options['fname']
    # get grid object
    with thread_lock:
        with netCDF4.Dataset(fname, 'r') as f:
            lon = f[file_options['lon_name']][:]
            lat = f[file_options['lat_name']][:]
    x, y = _proj(lon, lat)
    grid:Grid = Grid(_proj, x, y)

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
