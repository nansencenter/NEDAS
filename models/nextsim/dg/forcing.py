from datetime import datetime, timedelta
import os
import warnings
import typing

import numpy as np
import netCDF4 # type: ignore
import cftime # type: ignore

from utils.conversion import t2s
from perturb import random_perturb, apply_perturb, pres_adjusted_wind_perturb, apply_AR1_perturb


def get_time_index_from_nc(f: netCDF4.Dataset, time_varname:str, time_units:str, time: datetime) -> int:
    """Get the time index from the netcdf file"""
    # get the start time of the current file
    start_time: datetime = cftime.num2date(f[time_varname][0], units=time_units)
    # get the time step
    time_step: timedelta = cftime.num2date(f[time_varname][1], units=time_units) - start_time
    return np.rint((time - start_time) / time_step)


def read_var(fname:str, varnames:list[str], itime: int) -> np.ma.MaskedArray:
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
            data.append(f[vname][itime])
    return np.ma.array(data)


def write_var(fname:str, varnames: list[str], data: np.ndarray, itime: int) -> None:
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
            f[vname][itime] = data[i]


def geostrophic_perturb(fname:str, options:dict, itime:int, pert:np.ndarray, varname:str) -> None:
    """Perturb the atmosphere wind by the geostrophic balance.
    This applies to horizontal 2D wind fields.

    Parameters
    ----------
    fname : str
        forcing file name
    options : dict
        perturbation options for the geostrophic_wind_adjust section of atmosphere forcing from yaml file
    itime : int
        current time index in the forcing file
    pert : np.ndarray
        perturbation array
    varname : str
        name of the variable to be used to perturb wind fields

    Returns
    -------
    None
    """
    if options['do_adjust'] != True: return

    pres_name: str = options['pres_name']
    if pres_name != varname: return

    # doing wind perturbations by considering the geostrophic balance
    pert_u, pert_v = pres_adjusted_wind_perturb(grid,
                                                options['pres_pert_amp'],
                                                options['wind_pert_amp'],
                                                options['hcorr'], pert)

    uname:str = options['u_name']
    vname:str = options['v_name']
    u: np.ndarray = read_var(fname, [uname,], itime)
    v: np.ndarray = read_var(fname, [vname,], itime)
    u = apply_perturb(grid, u, pert_u, options['type'])
    v = apply_perturb(grid, v, pert_v, options['type'])
    if options['wind_amp_name'] != None:
        wind_amp_name:str = options['wind_amp_name']
        wind_amp = np.sqrt(u**2 + v**2)
        write_var(fname, [wind_amp_name,], wind_amp, itime)
    write_var(fname, [uname, ], u, itime)
    write_var(fname, [vname, ], v, itime)


def perturb_forcing(perturb_options:dict, time: datetime, prev_time: datetime) -> None:
    """perturb the forcing variables"""

    # perturbation arrays
    pert: np.ndarray[typing.Any, np.dtype[np.float64]]
    # path to the saved perturbation file for current time step
    prev_perturb_file: str
    # path to the directory of the perturbation files
    pert_path: str = perturb_options['path']
    # filename of the forcing file
    fname: str

    for forcing_name in ['atmosphere', 'ocean']:
        try:
            perturb_forcing_options:dict = perturb_options[forcing_name]
        except KeyError:
            warnings.warn(f'No {forcing_name} perturbation options found in the yaml file. '
                      f'Please specify the perturbation options in "{forcing_name}" section of nextsim.dg'
                      f' if you\'d like to perturb the {forcing_name} forcing')
            break

        fname = perturb_forcing_options['filename']
        options = perturb_forcing_options['all']
        # get current time index in the forcing file
        itime:int = get_time_index_from_nc(fname, perturb_forcing_options['time_name'], perturb_forcing_options['time_units'], time)
        for i, varname in enumerate(options['names']):
            if typing.TYPE_CHECKING:
                assert type(varname) == str, 'variable name must be a string'
            # get the perturbations
            prev_perturb_file = os.path.join(pert_path, f'perturb_{varname.replace("/", "_")}_{t2s(prev_time)}.npy')
            if os.path.exists(prev_perturb_file):
                pert = np.load(prev_perturb_file)
            else:
                pert = random_perturb(grid, options['type'][i], options['amp'][i], options['hcorr'][i])
            # apply perturbations to the variable data
            # in the case of vector fields, we need to split the variable name
            varname_list: list[str] = varname.split(';')
            # read the variable data from forcing file
            data: np.ndarray = read_var(fname, varname_list, itime)
            # apply perturbations to the variable data
            data = apply_perturb(grid, data, pert, options['type'][i])
            # apply lower and upper bounds
            lb = options['lower_bounds'][i] if options['lower_bounds'][i] != None else -np.inf
            ub = options['upper_bounds'][i] if options['upper_bounds'][i] != None else np.inf
            data = np.minimum(np.maximum(data, lb), ub)
            # write the perturbed variable back to the forcing file
            write_var(fname, varname_list, data, itime)
            # generate wind perturbations and apply them to the atmosphere forcing files based on pressure perturbations
            if forcing_name == 'atmosphere': geostrophic_perturb(fname, perturb_forcing_options['geostrophic_wind_adjust'], itime, pert, varname)
            # generate random perturbations for next time step with AR1 correlation
            pert_new = random_perturb(grid, options['type'][i], options['amp'][i], options['hcorr'][i])
            pert = apply_AR1_perturb(pert_new, options['tcorr'][i], pert)
            # save the perturbations for the next time step
            perturb_file:str = os.path.join(pert_path, f'perturb_{varname.replace("/", "_")}_{t2s(time)}.npy')
            np.save(perturb_file, pert)
