"""This module is used to perturb the forcing variables for the next forecast cycle.

This module is specifically designed for neXtSIM-DG in NEDAS.
The design follows the perturbation strategy in TOAPZ4 where
the perturbation is temporally correlationed as an AR1 process.

Parameters of the perturbation are read from the yaml file.
The perturbation is applied to the forcing variables in the forcing files.
"""
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta # type: ignore
import os
import threading
import typing

import cftime # type: ignore
import netCDF4 # type: ignore
import numpy as np
import pyproj # type: ignore

from NEDAS.utils.conversion import t2s
from NEDAS.grid import Grid
from NEDAS.models.nextsim.dg.perturb import gen_perturb, apply_perturb, pres_adjusted_wind_perturb, apply_AR1_perturb
from NEDAS.models.nextsim.dg import slicing_nc

# both the topaz and ERA5 data are projected onto the same grid at the moment
_proj:pyproj.Proj = pyproj.Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)
# the thread lock for reading and writing netcdf files
thread_lock = threading.Lock()

def get_fname_daterange(current_date: datetime, initial_date:str, interval:str, forcing_file_date_format:str) -> tuple[str, str]:
    """Inferring the date range of the forcing file for the current time,
    the forcing start date and the forcing interval in the initial forcing file given by yaml file.

    Parameters
    ----------
    current_date : datetime
        current date
    initial_date : str
        forcing start date
    interval : str
        forcing interval
    forcing_file_date_format : str
        forcing file date format expressed in strftime format

    Returns
    -------
    tuple[str, str]
        start date and end date of the forcing file for current time
    """
    # Parse the dates
    initial_date_dt:datetime = datetime.strptime(initial_date, forcing_file_date_format)
    initial_date_dt = initial_date_dt.replace(tzinfo=timezone.utc)
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


def get_time_from_nc(fname:str, time_varname:str, time_units_name:str, time: datetime, next_time: datetime, debug:bool=False) -> tuple[np.ndarray, list[datetime]]:
    """Get the indices and corresponding time that includes time and next_time from the netcdf file

    This function is not seeking the exact time and next_time in the forcing file,
    but the time steps that include the time and next_time such that the perturbed forcing
    file can be a bit smaller. Therefore, we allow for a few more time steps in this file.

    Parameters
    ----------
    fname : str
        forcing file name
    time_varname : str
        time variable name in the forcing file
    time_units_name : str
        variable name that gives time units in the forcing file
    time : datetime
        current time
    next_time : datetime
        time at the end of the forecast cycle

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        indices and corresponding time for the next forecast cycle
    """
    with thread_lock:
        with netCDF4.Dataset(fname, 'r') as f:
            time_units = f[time_units_name].units
            # get the start time in the forcing file
            start_time: datetime = cftime.num2date(f[time_varname][0], units=time_units, only_use_cftime_datetimes=False)
            # get the time step in the forcing file
            time_step: timedelta = cftime.num2date(f[time_varname][1], units=time_units, only_use_cftime_datetimes=False) - start_time
            start_time = start_time.replace(tzinfo=timezone.utc)
            # get the indices of the current and next time steps
            it0: int = int(np.rint((time - start_time) / time_step))
            it1: int = int(np.rint((next_time - start_time) / time_step))
            # get total number of time steps in the current file
            nt: int = len(f[time_varname][:])
            # extend the forcing time step by one to ensure all time steps are included
            it0 = max(0, min(it0 - 1, nt - 1))
            it1 = max(0, min(it1 + 1, nt - 1))
            # get the all the time between current time and next time in the forcing file
            file_time: list[datetime] = [cftime.num2date(f[time_varname][it], time_units)
                                         for it in range(it0, it1 + 1)]
            if debug:
                print (f'file: {fname}; 'f'file time: {file_time[0]} to {file_time[-1]},'
                       f'forecast time: {time} to {next_time}')
    return np.arange(it0, it1 + 1), file_time

def get_time_index(fname:str, time_varname:str, time_units_name:str, time:datetime) -> int:
    """
    Get the index of time in a netcdf file
    """
    with thread_lock:
        with netCDF4.Dataset(fname, 'r') as f:
            time_units = f[time_units_name].units
            start_time: datetime = cftime.num2date(f[time_varname][0], units=time_units, only_use_cftime_datetimes=False)
            time_step: timedelta = cftime.num2date(f[time_varname][1], units=time_units, only_use_cftime_datetimes=False) - start_time
            start_time = start_time.replace(tzinfo=timezone.utc)
            ind: int = int(np.rint((time - start_time) / time_step))
    return ind

def get_prev_time_from_nc(fname:str, time_varname:str, time_units_name:str, itime:int) -> datetime:
    """Get the previous time in the netcdf file before the start of the forecast cycle

    Parameters
    ----------
    fname : str
        forcing file name
    time_varname : str
        time variable name in the forcing file
    time_units_name : str
        variable name that gives time units in the forcing file
    itime : int
        current time index in the forcing file
    """
    with thread_lock:
        with netCDF4.Dataset(fname, 'r') as f:
            it :int = max(0, itime - 1)
            time_units:str = f[time_units_name].units
            prev_time: datetime = cftime.num2date(f[time_varname][it], units=time_units, only_use_cftime_datetimes=False)
            prev_time = prev_time.replace(tzinfo=timezone.utc)
    return prev_time


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
    data: list[np.ndarray] = []
    with thread_lock:
        with netCDF4.Dataset(fname, 'r') as f:
            # read the variable
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
    with thread_lock:
        with netCDF4.Dataset(fname, 'r+') as f:
            for i, vname in enumerate(varnames):
                f[vname][itime] = data[i]
                f.sync()


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
    forcing_end_date:str
    forcing_start_date, forcing_end_date = \
        get_fname_daterange(time, forcing_file_initial_date, forcing_file_interval, forcing_file_date_format)
    fname: str
    try:
        fname = file_format.format(i=i_ens , start=forcing_start_date, end=forcing_end_date)
    except KeyError:
        try:
            fname = file_format.format(start=forcing_start_date, end=forcing_end_date)
        except KeyError:
            print ('Currently, we only supports keyword of 1. "start"+"end",'
                   '2. "start"+"end"+"i".'
                   'See the example yaml file for more information. '
                   'Modified the code if you have other requirements.')
    return fname


def perturb_forcing(forcing_options:dict, file_options:dict, i_ens: int, time: datetime, next_time:datetime, debug=False) -> None:
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
        current time as the begining of the forecast cycle
    next_time : datetime
        end time of the next forecast cycle
    """
    # perturbation arrays
    pert: np.ndarray[typing.Any, np.dtype[np.float64]]
    # path to the directory of the perturbation files
    pert_path: str = forcing_options['path']
    # create the directory if it does not exist
    os.makedirs(os.path.join(pert_path, f'ensemble_{i_ens}'), exist_ok=True)
    # time index and time array
    time_index: np.ndarray[typing.Any, np.dtype[np.int64]]
    time_array: list[datetime]

    for forcing_name in forcing_options:
        if forcing_name not in file_options: continue
        # forcing options for each component, e.g., atmosphere or ocean
        forcing_options_comp:dict = forcing_options[forcing_name]
        file_options_comp:dict = file_options[forcing_name]
        # get the forcing file name
        fname:str = file_options_comp['fname']
        # copy forcing files to the ensemble member directory
        # we don't change the filename,
        # but only copy limited time slices of the original forcing file
        time_index, time_array = get_time_from_nc(file_options_comp['fname_src'],
                                                  file_options_comp['time_name'],
                                                  file_options_comp['time_units_name'],
                                                  time, next_time, debug
                                                  )
        # get prev_time
        prev_time:datetime = get_prev_time_from_nc(file_options_comp['fname_src'],
                                                   file_options_comp['time_name'],
                                                   file_options_comp['time_units_name'],
                                                   time_index[0]
                                                   )
        with thread_lock:
            slicing_nc.copy_time_sliced_nc_file(file_options_comp['fname_src'],
                                                fname, time_index,
                                                file_options_comp['time_name'],
                                                time_array[0])

        # get grid object for geometric information
        with thread_lock:
            with netCDF4.Dataset(fname, 'r') as f:
                grid = Grid(_proj, *_proj(f[file_options_comp['lon_name']],
                                        f[file_options_comp['lat_name']]
                                        )
                            )

        for itime, time_f in enumerate(time_array):
            # get options for perturbing the forcing variables
            options = forcing_options_comp['variables']
            for i, varname in enumerate(options['names']):
                if typing.TYPE_CHECKING:
                    assert type(varname) == str, 'variable name must be a string'
                # variable name in saved .npy filename
                varname_f:str = varname.replace("/", "_")
                # get perturbations
                pert_fname:str = os.path.join(pert_path, f'ensemble_{i_ens}',
                                              f'perturb_{varname_f}_{t2s(time_f)}.npy')
                if os.path.exists(pert_fname):
                    pert = np.load(pert_fname)
                else:
                    # convert the horizontal correlation length scale to grid points
                    hcorr:int = np.rint(float(options['hcorr'][i])/grid.dx)
                    # generate new perturbations
                    if prev_time != time_f:
                        prev_perturb_fname:str = os.path.join(pert_path,
                                                          f'ensemble_{i_ens}',
                                                          f'perturb_{varname_f}_{t2s(prev_time)}.npy')
                        pert_prev:np.ndarray = np.load(prev_perturb_fname)
                        # generate random perturbations for the current time step with AR1 correlation
                        pert_new:np.ndarray = gen_perturb(grid,
                                                          options['type'][i], float(options['amp'][i]),
                                                          hcorr)
                        pert = apply_AR1_perturb(pert_new, float(options['tcorr'][i]), pert_prev)
                    else:
                        pert = gen_perturb(grid, options['type'][i], float(options['amp'][i]), hcorr)

                    # save the perturbations for the next time step
                    perturb_file:str = os.path.join(pert_path,
                                                    f'ensemble_{i_ens}',
                                                    f'perturb_{varname_f}_{t2s(time_f)}.npy')
                    np.save(perturb_file, pert)

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
                if forcing_name == 'atmosphere': geostrophic_perturb(fname, grid,
                                                                     forcing_options_comp['geostrophic_wind_adjust'],
                                                                     itime, pert, varname)

            prev_time = time_f
