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

# both the topaz and ERA5 data are projected onto the same grid at the moment
_proj:pyproj.Proj = pyproj.Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)

def get_forcing_file_time(current_date: datetime, initial_date:str, interval:str, forcing_file_date_format:str) -> tuple[str, str]:
    """Get the forcing file time based on the current time, the forcing start date and the forcing interval

    Parameters
    ----------
    current_date : datetime
        current date
    initial_date : str
        forcing start date
    interval : str
        forcing interval

    Returns
    -------
    tuple[str, str]
        start date and end date of the forcing file for current time
    """
    # Parse the dates
    initial_date_dt:datetime = datetime.strptime(initial_date, forcing_file_date_format)
    keywords:dict[str, str] = {'y': 'years', 'm': 'months', 'd': 'days'}
    # Initialize start and end dates
    start_date = initial_date_dt
    end_date = initial_date_dt + relativedelta(**{keywords[interval[-1]]: int(interval[:-1])})

    assert current_date >= start_date, \
        f'Current time {current_date} is earlier than the initial forcing date {initial_date}'
    # Calculate the intervals until the current date is within the range
    while end_date <= current_date:
        start_date = end_date
        end_date = start_date + relativedelta(**{keywords[interval[-1]]: int(interval[:-1])})

    # Format the dates back to strings
    start_date_str = start_date.strftime(forcing_file_date_format)
    end_date_str = end_date.strftime(forcing_file_date_format)

    return start_date_str, end_date_str


def get_time_index_from_nc(fname:str, time_varname:str, time_units_name:str, time: datetime) -> int:
    """Get the time index from the netcdf file"""
    with netCDF4.Dataset(fname, 'r') as f:
        time_units = f[time_units_name].units
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


def geostrophic_perturb(fname:str, grid:Grid, options:dict, itime:int, pert:np.ndarray, varname:str) -> None:
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
    if not options['do_adjust']: return

    pres_name: str = options['pres_name']
    if pres_name != varname: return

    # doing wind perturbations by considering the geostrophic balance
    pert_u, pert_v = pres_adjusted_wind_perturb(grid,
                                                float(options['pres_pert_amp']),
                                                float(options['wind_pert_amp']),
                                                float(options['hcorr']), pert)

    uname:str = options['u_name']
    vname:str = options['v_name']
    u: np.ndarray = read_var(fname, [uname,], itime)
    v: np.ndarray = read_var(fname, [vname,], itime)
    u = apply_perturb(grid, u, pert_u, options['type'])
    v = apply_perturb(grid, v, pert_v, options['type'])
    if options['wind_amp_name'] != 'None':
        wind_amp_name:str = options['wind_amp_name']
        wind_amp = np.sqrt(u**2 + v**2)
        write_var(fname, [wind_amp_name,], wind_amp, itime)
    write_var(fname, [uname, ], u, itime)
    write_var(fname, [vname, ], v, itime)


def get_forcing_filename(forcing_file_options:dict, i_ens:int, time:datetime) -> str:
    """Get the forcing file name based on the current time and the forcing file format

    Parameters
    ----------
    forcing_file_options : dict
        forcing file options in the `file` section of the subsections of `perturb` section from the yaml file
    i_ens : int
        ensemble index
    time : datetime
        current time

    Returns
    -------
    str
        forcing file name
    """
    # derive the forcing file name
    # the format of the forcing file name
    file_format:str = forcing_file_options['format']
    # the date of the first forcing file
    forcing_file_initial_date:str = forcing_file_options['initial_date']
    # the length of the each forcing file
    forcing_file_interval:str = forcing_file_options['interval']
    # the length of the each forcing file
    forcing_file_date_format:str = forcing_file_options['datetime_format']
    # get the forcing file time
    forcing_start_date:str
    forcing_end_state:str
    forcing_start_date, forcing_end_state = \
        get_forcing_file_time(time, forcing_file_initial_date, forcing_file_interval, forcing_file_date_format)
    fname: str
    try:
        fname = file_format.format(i=i_ens , start=forcing_start_date, end=forcing_end_state)
    except KeyError:
        try:
            fname = file_format.format(start=forcing_start_date, end=forcing_end_state)
        except KeyError:
            print ('Currently, we only supports keyword of 1. "start"+"end",'
                   '2. "start"+"end"+"i".'
                   'See the example yaml file for more information. '
                   'Modified the code if you have other requirements.')
    return fname


def perturb_forcing(forcing_options:dict, file_options:dict, i_ens: int, time: datetime, prev_time: datetime) -> None:
    """perturb the forcing variables

    Parameters
    ----------
    forcing_options : dict
        perturbation options from the yaml file
    file_options : dict
        forcing file options in the corresponding subsection of the `file` section from the yaml file
        e.g., info in the file/forcing/atmosphere is used in the perturb/forcing/atmosphere section
        Before calling this function, one must add the following keys to the file_options dictionary:
        - fname: the exact filename of the perturbed forcing file under absbolute path

    i_ens : int
        ensemble index
    time : datetime
        current time
    prev_time : datetime
        previous time
    """

    # perturbation arrays
    pert: np.ndarray[typing.Any, np.dtype[np.float64]]
    # path to the saved perturbation file for current time step
    prev_perturb_file: str
    # path to the directory of the perturbation files
    pert_path: str = forcing_options['path']

    for forcing_name in forcing_options.keys():
        # forcing options for each component, e.g., atmosphere or ocean
        forcing_options_comp:dict = forcing_options[forcing_name]
        file_options_comp:dict = file_options[forcing_name]
        # get the forcing file name
        fname:str = file_options_comp['fname']
        # get current time index in the forcing file
        itime:int = get_time_index_from_nc(fname, file_options_comp['time_name'],
                                           file_options_comp['time_units_name'], time
                                           )

        # get grid object
        with netCDF4.Dataset(fname, 'r') as f:
            grid = Grid(_proj, *_proj(f[file_options_comp['lon_name']],
                                      f[file_options_comp['lat_name']]
                                      )
                        )

        # get options for perturbing the forcing variables
        options = forcing_options_comp['variables']
        for i, varname in enumerate(options['names']):
            if typing.TYPE_CHECKING:
                assert type(varname) == str, 'variable name must be a string'

            # convert the horizontal correlation length scale to grid points
            hcorr:int = np.rint(float(options['hcorr'][i])/grid.dx)
            # get the perturbations
            prev_perturb_file = os.path.join(pert_path, f'ensemble_{i_ens}', f'perturb_{varname.replace("/", "_")}_{t2s(prev_time)}.npy')
            if os.path.exists(prev_perturb_file):
                pert = np.load(prev_perturb_file)
            else:
                pert = gen_perturb(grid, options['type'][i], float(options['amp'][i]), hcorr)

            # apply perturbations to the variable data
            # in the case of vector fields, we need to split the variable name
            varname_list: list[str] = varname.split(';')
            # read the variable data from forcing file
            data: np.ndarray = read_var(fname, varname_list, itime)
            # apply perturbations to the variable data
            data = apply_perturb(grid, data, pert, options['type'][i])
            # apply lower and upper bounds
            lb = float(options['lower_bounds'][i]) if options['lower_bounds'][i] != 'None' else -np.inf
            ub = float(options['upper_bounds'][i]) if options['upper_bounds'][i] != 'None' else np.inf
            data = np.minimum(np.maximum(data, lb), ub)
            # write the perturbed variable back to the forcing file
            write_var(fname, varname_list, data, itime)

            # generate wind perturbations and apply them to the atmosphere forcing files based on pressure perturbations
            if forcing_name == 'atmosphere': geostrophic_perturb(fname, grid, forcing_options_comp['geostrophic_wind_adjust'],
                                                                 itime, pert, varname)

            # generate random perturbations for next time step with AR1 correlation
            pert_new = gen_perturb(grid, options['type'][i], float(options['amp'][i]), hcorr)
            pert = apply_AR1_perturb(pert_new, float(options['tcorr'][i]), pert)
            # save the perturbations for the next time step
            perturb_file:str = os.path.join(pert_path, f'ensemble_{i_ens}', f'perturb_{varname.replace("/", "_")}_{t2s(time)}.npy')
            np.save(perturb_file, pert)
