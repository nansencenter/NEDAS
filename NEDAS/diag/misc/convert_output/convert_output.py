import os
import yaml
import cftime
import numpy as np
from pyproj import Proj
from NEDAS.grid import Grid
from NEDAS.utils.netcdf_lib import nc_write_var
from NEDAS.utils.conversion import ensure_list, proj2dict, s2t, dt1h
from NEDAS.utils.shell_utils import makedir

def get_task_list(c, **kwargs):
    """get a list of tasks to be done, as unique kwargs to be passed to run()"""
    # parse kwargs
    # load the default config first
    default_config_file = os.path.join(os.path.dirname(__file__), 'default.yml')
    with open(default_config_file, 'r') as f:
        kwargs_from_default = yaml.safe_load(f)
    kwargs_from_config_file = {}
    if 'config_file' in kwargs and kwargs['config_file'] is not None:
        with open(kwargs['config_file'], 'r') as f:
            kwargs_from_config_file = yaml.safe_load(f)
    # overwrite with kwargs
    kwargs = {**kwargs_from_default, **kwargs_from_config_file, **kwargs}

    ##generate task list
    tasks = []
    for member in range(c.nens):
        for vname in ensure_list(kwargs['variables']):
            tasks.append({**kwargs, 'variable': vname, 'member': member})
    return tasks

def get_file_list(c, **kwargs):
    """Return a list of files to be opened for collective i/o"""
    if 'time' in kwargs:
        time_start = s2t(kwargs['time'])
    else:
        time_start = c.time

    files = []
    for member in range(c.nens):
        file = kwargs['file'].format(time=time_start, member=member+1)
        if file not in files:
            files.append(file)
    return files

def run(c, **kwargs):
    """
    Run diagnostics: convert model restart variables into netcdf files, given formatting options in kwargs
    """
    vname = kwargs['variable']
    member = kwargs['member']
    model_name = kwargs['model_src']
    model = c.models[model_name]

    grid_def = kwargs['grid_def']
    if grid_def:
        proj = Proj(grid_def['proj'])
        xmin = grid_def['xmin']
        xmax = grid_def['xmax']
        ymin = grid_def['ymin']
        ymax = grid_def['ymax']
        dx = grid_def['dx']
        centered = grid_def.get('centered', False)
        grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, centered)
    else:
        grid = model.grid
    proj_params = proj2dict(grid.proj)
    x = grid.x[0, :] / 1e5
    y = grid.y[:, 0] / 1e5  ##convert to 100km units
    lon, lat = grid.proj(grid.x, grid.y, inverse=True)

    if 'time' in kwargs:
        time_start = s2t(kwargs['time'])
    else:
        time_start = c.time
    dt_hours = kwargs.get("dt_hours", model.output_dt)
    forecast_hours = kwargs.get("forecast_hours", c.cycle_period)
    time_units = kwargs.get('time_units', 'seconds since 1970-01-01T00:00:00+00:00')
    time_calendar = kwargs.get('time_calendar', 'standard')
    t_steps = range(0, forecast_hours, dt_hours)
    path = c.forecast_dir(time_start, model_name)

    for n_step, t_step in enumerate(t_steps):
        t = time_start + t_step * dt1h

        if c.debug:
            print(f"PID {c.pid:4} convert_output on variable '{vname:20}' for {model_name:10} member {member+1:3} at {t}", flush=True)
        # read the variable from the model restart file
        rec = model.variables[vname]
        file = kwargs['file'].format(time=time_start, member=member+1)
        makedir(os.path.dirname(file))

        lon_name: str = kwargs.get('lon_name', 'lon')
        lat_name: str = kwargs.get('lat_name', 'lat')
        x_name: str = kwargs.get('x_name', 'x')
        y_name: str = kwargs.get('y_name', 'y')
        time_name: str = kwargs.get('time_name', 'time')
        recno = {}
        recno[time_name] = n_step
        levels = rec['levels']
        is_vector = rec['is_vector']
        for k in levels:
            # read the field from model restart file
            model.read_grid(path=path, name=vname, time=t, member=member, k=k)
            fld = model.read_var(path=path, name=vname, time=t, member=member, k=k)
            model.grid.set_destination_grid(grid)

            # convert to output grid
            fld_ = model.grid.convert(fld, is_vector=is_vector)
            # build dimension records
            dims = {}
            dims[time_name] = None  ##make time dimension unlimited in nc file
            k_name = kwargs.get('k_name')
            if len(levels) > 1:
                dims[k_name] = None  ##add level dimension (unlimited) if there are multiple levels
            dims[y_name] = grid.ny
            dims[x_name] = grid.nx
            # output the variable
            if len(levels) > 1:
                recno[k_name] = list(levels).index(k)
            # variable attr
            attr = {'standard_name':vname,
                    'units':rec['units'],
                    'grid_mapping': proj_params['projection'],
                    'coordinates': f"{lon_name} {lat_name}",
                    }
            if is_vector:
                for i in range(2):
                    if isinstance(rec['name'], tuple):
                        rec_name = rec['name'][i]
                    else:
                        rec_name = rec['name']+'_'+(x_name, y_name)[i]
                    nc_write_var(file, dims, rec_name, fld_[i,...], recno=recno, attr=attr, comm=c.comm)
            else:
                nc_write_var(file, dims, rec['name'], fld_, recno=recno, attr=attr, comm=c.comm)

        # output the dimension variables
        time = cftime.date2num(t, units=time_units)
        time_attr = {'long_name': 'forecast time', 'units': time_units, 'calendar': time_calendar}
        nc_write_var(file, {time_name:None}, time_name, time, dtype=float, recno=recno, attr=time_attr, comm=c.comm)
        nc_write_var(file, {x_name:grid.nx}, x_name, x, attr={'standard_name':'projection_x_coordinate', 'units':'100 km'}, comm=c.comm)
        nc_write_var(file, {y_name:grid.ny}, y_name, y, attr={'standard_name':'projection_y_coordinate', 'units':'100 km'}, comm=c.comm)
        nc_write_var(file, {y_name:grid.ny, x_name:grid.nx}, lon_name, lon, attr={'standard_name':'longitude', 'units':'degrees_east'}, comm=c.comm)
        nc_write_var(file, {y_name:grid.ny, x_name:grid.nx}, lat_name, lat, attr={'standard_name':'latitude', 'units':'degrees_north'}, comm=c.comm)
        # output projection info
        nc_write_var(file, {}, proj_params['projection'], np.array([1]), attr=proj_params, comm=c.comm)
