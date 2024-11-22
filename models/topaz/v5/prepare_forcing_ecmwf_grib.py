import sys
import os
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

grib_path = "/nird/projects/NS2993K/METNO_2_NERSC/ACCIBEREG/EC_grib_files"
forcing_path = "/cluster/projects/nn2993k/yingyue/ecmwf_fcsts"
dt = 6
variables = {
    'atmos_surf_velocity': {'name':('wndewd', 'wndnwd'), 'is_vector':True, 'units':'m/s'},
    'atmos_surf_temp':     {'name':'airtmp', 'is_vector':False, 'units':'C'},
    'atmos_surf_dewpoint': {'name':'dewpt', 'is_vector':False, 'units':'K'},
    'atmos_surf_press':    {'name':'mslprs', 'is_vector':False, 'units':'Pa'},
    'atmos_precip':        {'name':'precip', 'is_vector':False, 'units':'precip m/s'},
    'atmos_down_longwave': {'name':'radflx', 'is_vector':False, 'units':'W/m2'},
    'atmos_down_shortwave': {'name':'shwflx', 'is_vector':False, 'units':'W/m2'},
    }
native_variables = {
    'airtmp': {'id':1, 'units':'K'},
    'dewpt':  {'id':2, 'units':'K'},
    'mslprs': {'id':3, 'units':'Pa'},
    'precip': {'id':7, 'units':'m/6h'},
    'radflx': {'id':10, 'units':'J/m2/6h'},
    'shwflx': {'id':9, 'units':'J/m2/6h'},
    'wndewd': {'id':5, 'units':'m/s'},
    'wndnwd': {'id':6, 'units':'m/s'},
    }

def filename(path, t, member):
    year = t.year
    month = t.month
    if member is None:
        ##deterministic forecast
        search = os.path.join(path, f"{year}-{month}", f"aciceberg_{year}_{month}*.grb")
    else:
        ##probablistic forecasts
        search = os.path.join(path, f"{year}-{month}", f"aciceberg_pf_{year}_{month}*.grb")

    file_list = glob.glob(search)
    for file in file_list:
        # Extract the date range from the file name
        date_range = file.split(".")[0].split("_")[-1]
        day1, day2 = date_range.split("-")
        if t.day in [day for day in range(int(day1), int(day2)+1)]:
            filename = file
            day_start = int(day1)
            break
    return file, day_start

def read_var(grbs, day_start, t, member, varname):
    nvar = 10
    t_start = datetime(t.year, t.month, day_start)
    t_step = int((t - t_start) / (dt * timedelta(hours=1)))
    if member is None:
        nens = 1
        member = 0
    else:
        nens = 50
    var_id = native_variables[varname]['id']
    rec_id = nens*nvar*t_step + nvar*member + var_id
    grb = grbs.message(rec_id)
    return grb.values

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
    rdtime = dt / 24

    ##open the file handle
    if f[name] is None:
        f[name] = ABFileForcing(filename, 'w', idm=idm, jdm=jdm, cline1=cline1, cline2=cline2)

    ##write var to file
    f[name].write_field(var, None, name, dtime1, rdtime)

def output_ncfile(filename, name, member, t, tstart, var):
    print("output to "+filename)
    ny, nx = var.shape
    t_step = int((t - tstart) / (dt*timedelta(hours=1)))
    if member is None:
        member = 0
    else:
        member = member + 1
    nc_write_var(filename, {'t':None, 'mem':None, 'y':ny, 'x':nx}, name, var, recno={'t':t_step, 'mem':member})

if __name__ == '__main__':
    ##usage: python prepare_forcing_ecmwf_grib.py time_start time_end member
    time_start = datetime.strptime(sys.argv[1], '%Y%m%d%H%M')
    time_end = datetime.strptime(sys.argv[2], '%Y%m%d%H%M')
    member = int(sys.argv[3])
    if member == 0:
        member = None
    else:
        member = member - 1

    grid = None
    file_list = []
    f = {}
    for name in native_variables.keys():
        f[name] = None

    t = time_start
    while t <= time_end:
        print(f"\nforcing at {t}")
        file, day_start = filename(grib_path, t, member)
        if file not in file_list:
            print(f"opening {file}")
            file_list.append(file)
            grbs = pygrib.open(file)

        if grid is None:
            grid = read_grid(grbs.message(1))

        for varname, rec in variables.items():
            if member is None:
                print(f"getting {varname}")
            else:
                print(f"getting {varname} for member {member+1}")

            if rec['is_vector']:
                u = read_var(grbs, day_start, t, member, rec['name'][0])
                v = read_var(grbs, day_start, t, member, rec['name'][1])
                var = np.array([u, v])
            else:
                var = read_var(grbs, day_start, t, member, rec['name'])
                ##accumulated variables
                if rec['name'] in ['precip', 'radflx', 'shwflx'] and t.hour>0:
                    prev_t = t - dt * timedelta(hours=1)
                    var -= read_var(grbs, day_start, prev_t, member, rec['name'])

            print("convert to topaz grid")
            var_topaz = grid.convert(var, is_vector=rec['is_vector'])

            ##convert units

            forcing_file = model.filename(path=forcing_path, name=varname, member=member, time=t)
            if rec['is_vector']:
                for i in range(2):
                    fill_missing(var_topaz[i,...])
                    output_abfile(f, forcing_file+'.'+rec['name'][i], rec['name'][i], t, var_topaz[i,...])
                    output_ncfile(forcing_path+'/forcing.nc', rec['name'][i], member, t, time_start, var_topaz[i,...])
            else:
                fill_missing(var_topaz)
                output_abfile(f, forcing_file+'.'+rec['name'], rec['name'], t, var_topaz)
                output_ncfile(forcing_path+'/forcing.nc', rec['name'], member, t, time_start, var_topaz)

        t += dt * timedelta(hours=1)

    for name in native_variables.keys():
        f[name].close()

