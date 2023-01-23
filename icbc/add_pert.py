import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
import datetime
import sys
import os

def filename(t):
    return "generic_ps_atm_{:04d}{:02d}{:02d}.nc".format(t.year, t.month, t.day)

def pert_file(m_index, t_index):
    return cc.SCRATCH+"/perturbation/{:03d}/perturb_{:04d}.nc".format(m_index, t_index)

###translation of varname in perturbation files
varname_pert = {'x_wind_10m':'x_wind_10m',
                'y_wind_10m':'y_wind_10m'}

t_start = datetime.datetime.strptime(cc.DATE_START, '%Y%m%d%H%M')
t_end = datetime.datetime.strptime(cc.DATE_END, '%Y%m%d%H%M')
cp = datetime.timedelta(hours=cc.CYCLE_PERIOD/60)
n_per_day = int(np.maximum(1, 24*60/cc.CYCLE_PERIOD))

m_index = int(sys.argv[1])
pert_coef = 1.0 #float(sys.argv[2])
orig_dir = cc.SCRATCH+"/data/GENERIC_PS_ATM/from_ERA5"
out_dir = cc.SCRATCH+"/data/GENERIC_PS_ATM/wind10m_err{}/{:03d}".format(pert_coef, m_index)

t = t_start
while t < t_end:
    ##copy original forcing file
    os.system("cp "+orig_dir+"/"+filename(t)+" "+out_dir+"/"+filename(t))
    dat_out = dict()
    for v in varname_pert:
        dat_out[v] = nc.read(out_dir+'/'+filename(t), v)

    ##forcing file stored daily, perturb each time step within
    for n in range(n_per_day):
        t1 = t + n*cp
        n_index = int((t1-t_start)/cp)  ##perturbation index in time
        for v in varname_pert:
            dat_out[v][n, :, :] += pert_coef * nc.Dataset(pert_file(m_index, n_index))[varname_pert[v]][0, :, :]

    ##write output
    for v in varname_pert:
        nc.write(out_dir+'/'+filename(t), {'time':0, 'y':cc.NY, 'x':cc.NX}, v, dat_out[v])

    t += datetime.timedelta(days=1)
