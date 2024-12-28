import os
import argparse
from datetime import datetime, timedelta
import numpy as np
from utils.netcdf_lib import nc_write_var
from models.topaz.abfile import ABFileForcing
from models.topaz.time_format import dayfor
from models.topaz.v5 import Model
from dataset.ecmwf.forecast import Dataset

variables = {
    'atmos_surf_velocity': {'name':('wndewd', 'wndnwd'), 'is_vector':True, 'units':'m/s'},
    'atmos_surf_temp':     {'name':'airtmp', 'is_vector':False, 'units':'C'},
    'atmos_surf_dewpoint': {'name':'dewpt', 'is_vector':False, 'units':'K'},
    'atmos_surf_press':    {'name':'mslprs', 'is_vector':False, 'units':'Pa'},
    'atmos_precip':        {'name':'precip', 'is_vector':False, 'units':'m/s'},
    'atmos_down_longwave': {'name':'radflx', 'is_vector':False, 'units':'W/m2'},
    'atmos_down_shortwave': {'name':'shwflx', 'is_vector':False, 'units':'W/m2'},
    }

##default output path
forcing_path = "/cluster/work/users/yingyue/data/ecmwf_fcsts"

def fill_missing(var):
    ##there will be a pole hole after converting from reduced_gg to topaz_grid
    ##just fill it with surrounding values
    j, i = np.where(np.isnan(var))
    d = 2
    for n in range(len(j)):
        var[j[n], i[n]] = np.nanmean(var[j[n]-d:j[n]+d, i[n]-d:i[n]+d])
    return var

def output_abfile(f, filename, name, t, dt, var, units):
    ##make sure directory exists
    os.system("mkdir -p "+os.path.dirname(filename))
    print(f"output to {filename}.a")

    idm = topaz.grid.nx
    jdm = topaz.grid.ny
    cline1 = "ecmwf"
    cline2 = f"{name} ({units})"
    dtime1 = dayfor(topaz.yrflag, t.year, int(t.strftime('%j')), t.hour)
    rdtime = dt / 24

    ##open the file handle
    if f[name] is None:
        f[name] = ABFileForcing(filename, 'w', idm=idm, jdm=jdm, cline1=cline1, cline2=cline2)

    ##write var to file
    f[name].write_field(var, None, name, dtime1, rdtime)

def output_ncfile(filename, name, member, t, dt, tstart, var):
    print("output to "+filename)
    ny, nx = var.shape
    t_step = int((t - tstart) / (dt*timedelta(hours=1)))
    nc_write_var(filename, {'time':None, 'y':ny, 'x':nx}, name, var, recno={'time':t_step})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('time_start', help="Start time: format 'ccyymmddHHMM'")
    parser.add_argument('time_end', help="End time: format 'ccyymmddHHMM'")
    parser.add_argument('-m', '--member', default=0, help=f"Ensemble member to be processed (default: 0)")
    parser.add_argument('-o', '--output', default=forcing_path, help=f"Output directory (default: {forcing_path})")
    args = parser.parse_args()

    time_start = datetime.strptime(args.time_start, '%Y%m%d%H%M')
    time_end = datetime.strptime(args.time_end, '%Y%m%d%H%M')
    member = int(args.member)
    forcing_path = args.output

    ecmwf = Dataset(time_start=time_start)
    dt = ecmwf.dt_hours
    topaz = Model()

    file_list = []
    f = {}
    for name in [name for r in variables.values() for name in (r['name'] if isinstance(r['name'], tuple) else [r['name']])]:
        f[name] = None

    t = time_start
    while t <= time_end:

        for varname, rec in variables.items():
            print(f"\nprocess forcing variable '{varname}' at {t} for member {member}")

            ecmwf.read_grid(name=varname, time=t)
            var = ecmwf.read_var(name=varname, member=member, time=t)

            print("convert to topaz grid")
            ecmwf.grid.set_destination_grid(topaz.grid)
            var_topaz = ecmwf.grid.convert(var, is_vector=rec['is_vector'])

            path = os.path.join(forcing_path, f"{ecmwf.time_start:%Y%m%d%H%M}")
            forcing_file = topaz.filename(path=path, name=varname, member=member, time=t)
            forcing_file_nc = os.path.join(path, f"forcing_mem{member:03d}.nc")
            if rec['is_vector']:
                for i in range(2):
                    fill_missing(var_topaz[i,...])
                    output_abfile(f, forcing_file+'.'+rec['name'][i], rec['name'][i], t, dt, var_topaz[i,...], rec['units'])
                    output_ncfile(forcing_file_nc, rec['name'][i], member, t, dt, time_start, var_topaz[i,...])
            else:
                fill_missing(var_topaz)
                output_abfile(f, forcing_file+'.'+rec['name'], rec['name'], t, dt, var_topaz, rec['units'])
                output_ncfile(forcing_file_nc, rec['name'], member, t, dt, time_start, var_topaz)

        t += dt * timedelta(hours=1)

    for name in variables.keys():
        if f[name] is not None:
            f[name].close()
