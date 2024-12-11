import os
import argparse
import pygrib
import glob
import numpy as np
from datetime import datetime, timedelta
from grid import Grid
from pyproj import Proj
from utils.conversion import units_convert
from utils.netcdf_lib import nc_write_var
from models.topaz.abfile import ABFileForcing
from models.topaz.time_format import dayfor
from models.topaz.v5 import Model
model = Model()

grib_path = "/cluster/work/users/yingyue/data/ACCIBEREG/EC_grib_files"
nens = 50  ##ensemble members in file
dt_hours = 6  #interval hours of records in file
variables = {
    'atmos_surf_velocity': {'name':('wndewd', 'wndnwd'), 'is_vector':True, 'units':'m/s'},
    'atmos_surf_temp':     {'name':'airtmp', 'is_vector':False, 'units':'C'},
    'atmos_surf_dewpoint': {'name':'dewpt', 'is_vector':False, 'units':'K'},
    'atmos_surf_press':    {'name':'mslprs', 'is_vector':False, 'units':'Pa'},
    'atmos_precip':        {'name':'precip', 'is_vector':False, 'units':'m/s'},
    'atmos_down_longwave': {'name':'radflx', 'is_vector':False, 'units':'W/m2'},
    'atmos_down_shortwave': {'name':'shwflx', 'is_vector':False, 'units':'W/m2'},
    }
native_variables = {
    'airtmp': {'name':"2 metre temperature", 'units':'K'},
    'dewpt':  {'name':"2 metre dewpoint temperature", 'units':'K'},
    'mslprs': {'name':"Mean sea level pressure", 'units':'Pa'},
    'cloud':  {'name':"Total cloud cover", 'units':'%'},
    'precip': {'name':"Total precipitation", 'units':'m/s'},
    'radflx': {'name':"Surface long-wave (thermal) radiation downwards", 'units':'W/m2'},
    'shwflx': {'name':"Surface short-wave (solar) radiation downwards", 'units':'W/m2'},
    'wndewd': {'name':"10 metre U wind component", 'units':'m/s'},
    'wndnwd': {'name':"10 metre V wind component", 'units':'m/s'},
    }

##default output path
forcing_path = "/cluster/work/users/yingyue/data/ecmwf_fcsts"

def filename_analysis(path, t, ensemble):
    if ensemble:
        ##probablistic forecasts
        search = os.path.join(path, f"{t:%Y}-{t:%m}", f"aciceberg_pf_{t:%Y}_{t:%m}*.grb")
    else:
        ##deterministic forecast
        search = os.path.join(path, f"{t:%Y}-{t:%m}", f"aciceberg_{t:%Y}_{t:%m}*.grb")

    file_list = glob.glob(search)
    for file in file_list:
        # Extract the date range from the file name
        date_range = file.split(".")[0].split("_")[-1]
        tmp = date_range.split("-")
        day_start, day_end = int(tmp[0]), int(tmp[1])
        if t.day in [day for day in range(day_start, day_end+1)]:
            return file, t.day
    raise RuntimeError(f"could not find file that contain time {t}")

def filename_forecast(path, t, ensemble):
    search = os.path.join(path, f"{t:%Y}-{t:%m}", "FORECAST", f"fc_aciceberg_{t:%Y}-{t:%m}*.grb")
    forecast_days = 10
    file_list = glob.glob(search)
    for file in file_list:
        day_start = int(file.split(".")[0].split('-')[-1])
        if t.day in range(day_start, day_start + forecast_days + 1):
            return file, day_start
    raise RuntimeError(f"could not find file that contain time {t}")

def get_record_id_lookup(grbs):
    lookup = {}
    for i in range(grbs.messages):
        grb = grbs.message(i+1)
        variable_name = grb.name
        forecast_hours = int(grb.stepRange)
        start_date = grb.analDate
        member = grb.perturbationNumber
        key = (variable_name, start_date, forecast_hours, member)
        lookup[key] = i+1
    return lookup

def read_var(grbs, lookup, t_start, t, member, vname, units):
    ##get search key
    forecast_hours = int((t - t_start) / timedelta(hours=1))
    key = (native_variables[vname]['name'], t_start, forecast_hours, member)
    assert key in lookup, f"failed to search for record {key} in grib file {grbs.name}"

    ##look up the message id
    rec_id = lookup[key]
    ##read the message into var
    grb = grbs.message(rec_id)
    var = grb.values
    ##convert units
    var = units_convert(units, native_variables[vname]['units'], var)
    return var

def read_grid(grb):
    lat, lon = grb.latlons()
    grid = Grid(Proj("+proj=longlat"), lon, lat, cyclic_dim='x', pole_dim='y', pole_index=(0,))
    grid.set_destination_grid(model.grid)
    return grid

def fill_missing(var):
    ##there will be a pole hole after converting from reduced_gg to topaz_grid
    ##just fill it with surrounding values
    j, i = np.where(np.isnan(var))
    d = 2
    for n in range(len(j)):
        var[j[n], i[n]] = np.nanmean(var[j[n]-d:j[n]+d, i[n]-d:i[n]+d])
    return var

def output_abfile(f, filename, name, t, var):
    ##make sure directory exists
    os.system("mkdir -p "+os.path.dirname(filename))
    print(f"output to {filename}.a")

    idm = model.grid.nx
    jdm = model.grid.ny
    cline1 = "ecmwf"
    cline2 = f"{name} ({native_variables[name]['units']})"
    dtime1 = dayfor(model.yrflag, t.year, int(t.strftime('%j')), t.hour)
    rdtime = dt_hours / 24

    ##open the file handle
    if f[name] is None:
        f[name] = ABFileForcing(filename, 'w', idm=idm, jdm=jdm, cline1=cline1, cline2=cline2)

    ##write var to file
    f[name].write_field(var, None, name, dtime1, rdtime)

def output_ncfile(filename, name, member, t, tstart, var):
    print("output to "+filename)
    ny, nx = var.shape
    t_step = int((t - tstart) / (dt_hours*timedelta(hours=1)))
    nc_write_var(filename, {'time':None, 'member':None, 'y':ny, 'x':nx}, name, var, recno={'time':t_step, 'member':member})

def process(grbs, lookup, day_start, t, field_type, member):
    t_start = datetime(t.year, t.month, day_start)

    for varname, rec in variables.items():
        print(f"\nprocess forcing variable '{varname}' at {t} for member {member}")

        if rec['is_vector']:
            u = read_var(grbs, lookup, t_start, t, member, rec['name'][0], rec['units'])
            v = read_var(grbs, lookup, t_start, t, member, rec['name'][1], rec['units'])
            var = np.array([u, v])
        else:
            var = read_var(grbs, lookup, t_start, t, member, rec['name'], rec['units'])
            ##convert accumulative variables to flux
            if rec['name'] in ['precip', 'radflx', 'shwflx'] and t>t_start:
                t_prev = t - dt_hours * timedelta(hours=1)
                var -= read_var(grbs, lookup, t_start, t_prev, member, rec['name'], rec['units'])
                var /= 3600. * dt_hours

        print("convert to topaz grid")
        var_topaz = grid.convert(var, is_vector=rec['is_vector'])

        if member == 0:
            mem = None
        else:
            mem = member - 1
        forcing_file = model.filename(path=forcing_path, name=varname, member=mem, time=t)
        forcing_file_nc = os.path.join(forcing_path, f"forcing_{field_type}_mem{member:03d}.nc")
        if rec['is_vector']:
            for i in range(2):
                fill_missing(var_topaz[i,...])
                output_abfile(f, forcing_file+'.'+rec['name'][i], rec['name'][i], t, var_topaz[i,...])
                output_ncfile(forcing_file_nc, rec['name'][i], member, t, time_start, var_topaz[i,...])
        else:
            fill_missing(var_topaz)
            output_abfile(f, forcing_file+'.'+rec['name'], rec['name'], t, var_topaz)
            output_ncfile(forcing_file_nc, rec['name'], member, t, time_start, var_topaz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('time_start', help="Start time: format 'ccyymmddHHMM'")
    parser.add_argument('time_end', help="End time: format 'ccyymmddHHMM'")
    parser.add_argument('field_type', default='forecast', help="Type of forcing fields to extract: 'analysis' or 'forecast'")
    parser.add_argument('ensemble', type=bool, default=True, help="Ensemble or deterministic forecast (default True)")
    parser.add_argument('-m', '--member', default=0, help=f"Ensemble member to be processed (default: 0)")
    parser.add_argument('-o', '--output', default=forcing_path, help=f"Output directory (default: {forcing_path})")
    args = parser.parse_args()

    time_start = datetime.strptime(args.time_start, '%Y%m%d%H%M')
    time_end = datetime.strptime(args.time_end, '%Y%m%d%H%M')
    field_type = args.field_type
    ensemble = args.ensemble
    member = int(args.member)
    forcing_path = args.output

    grid = None
    file_list = []
    f = {}
    for name in native_variables.keys():
        f[name] = None

    t = time_start
    while t <= time_end:

        if field_type == 'analysis':
            filename = filename_analysis
        elif field_type == 'forecast':
            filename = filename_forecast
        file, day_start = filename(grib_path, t, ensemble)

        if file not in file_list:
            print(f"\n\nopening {file}")
            file_list.append(file)
            grbs = pygrib.open(file)

        lookup = get_record_id_lookup(grbs)

        if grid is None:
            grid = read_grid(grbs.message(1))

        process(grbs, lookup, day_start, t, field_type, member=member)

        t += dt_hours * timedelta(hours=1)

    for name in native_variables.keys():
        f[name].close()
