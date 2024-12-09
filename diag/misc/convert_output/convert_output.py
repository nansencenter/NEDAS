import yaml
import cftime
from pyproj import Proj
from grid import Grid
from utils.netcdf_lib import nc_write_var
from utils.conversion import ensure_list, proj2dict, s2t, dt1h
from utils.dir_def import forecast_dir

def run(c, **kwargs):
    """
    Run diagnostics: convert model restart variables into netcdf files, given formatting options in kwargs
    """
    # parse kwargs
    # load the default config first
    kwargs_from_config_file = {}
    if 'config_file' in kwargs:
        with open(kwargs['config_file'], 'r') as f:
            kwargs_from_config_file = yaml.safe_load(f)
    # overwrite with kwargs
    kwargs = {**kwargs_from_config_file, **kwargs}

    variables = ensure_list(kwargs['variables'])

    var_names = ensure_list(kwargs['var_names'])
    var_standard_names = ensure_list(kwargs['var_standard_names'])
    var_units = ensure_list(kwargs['var_units'])

    member = kwargs['member']
    
    model_name = kwargs['model_src']
    model = c.model_config[model_name]

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
    x = grid.x[0, :]
    y = grid.y[:, 0]
    lon, lat = grid.proj(grid.x, grid.y, inverse=True)
    
    model.grid.set_destination_grid(grid)

    if 'time_start' in kwargs:
        time_start = s2t(kwargs['time_start'])
    else:
        time_start = c.time_start
    if 'time_end' in kwargs:
        time_end = s2t(kwargs['time_end'])
    else:
        time_end = c.time_end 
    dt = c.cycle_period * dt1h
    time_units = kwargs.get('time_units', 'seconds since 1970-01-01T00:00:00+00:00')
    time_calendar = kwargs.get('time_calendar', 'standard')

    t = time_start
    while t <= time_end:
        path = forecast_dir(c, t, model_name)

        if c.debug:
            print(f"PID {c.pid} running misc.convert_output on variables {variables} for {model_name} member {member+1} at {t}", flush=True)
        
        for v_id, vname in enumerate(variables):
            # read the variable from the model restart file
            rec = model.variables[vname]
            file = kwargs['file'].format(time=t, var_name=var_names[v_id], member=member)

            levels = rec['levels']
            is_vector = rec['is_vector']
            for k in levels:
                # read the field from model restart file
                fld = model.read_var(path=path, name=vname, time=t, member=member)
                # convert to output grid
                fld_ = model.grid.convert(fld, is_vector=is_vector)
                # build dimension records
                dims = {}
                time_name = kwargs.get('time_name', 'time')
                dims[time_name] = None  ##make time dimension unlimited in nc file
                k_name = kwargs.get('k_name')
                if k_name: ##add multi-level field index 
                    dims[k_name] = len(levels)
                x_name = kwargs.get('x_name', 'x')
                y_name = kwargs.get('y_name', 'y')
                dims[y_name] = grid.ny
                dims[x_name] = grid.nx
                lon_name = kwargs.get('lon_name', 'lon')
                lat_name = kwargs.get('lat_name', 'lat')
                # output the variable
                recno = {time_name:0}  ##TODO:only process analysis time output, no forecast lead times yet
                # variable attr
                attr = {'standard_name':var_standard_names[v_id],
                        'units':var_units[v_id],
                        'grid_mapping': proj_params['projection'],
                        **proj_params, }
                if is_vector:
                    for i in range(2):
                        nc_write_var(file, dims, var_names[v_id][i], fld_[i,...], recno=recno, attr=attr, comm=c.comm)
                else:
                    nc_write_var(file, dims, var_names[v_id], fld_, recno=recno, attr=attr, comm=c.comm)

            # output the dimension variables
            time = cftime.date2num(t, units=time_units)
            time_attr = {'long_name': 'forecast time', 'units': time_units, 'calendar': time_calendar}
            nc_write_var(file, {time_name:None}, time_name, time, recno=recno, attr=time_attr, comm=c.comm)
            nc_write_var(file, {x_name:grid.nx}, x_name, x)
            nc_write_var(file, {y_name:grid.ny}, y_name, y)
            nc_write_var(file, {y_name:grid.ny, x_name:grid.nx}, lon_name, lon, attr={'standard_name':'longitude', 'units':'degrees_east'})
            nc_write_var(file, {y_name:grid.ny, x_name:grid.nx}, lat_name, lat, attr={'standard_name':'latitude', 'units':'degrees_north'})
        
        t += dt
