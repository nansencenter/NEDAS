import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
import grid.multiscale as ms
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

###scale component params
domain_size = np.maximum(cc.NX, cc.NY) * cc.DX/1e3
krange = domain_size/np.array([3600, 1800, 700, 320, 160, 80, 48, 20])
ns = len(krange)
growth_period = np.array([336, 288, 240, 96, 72, 36, 24, 12])

t_start = datetime.datetime.strptime(cc.DATE_START, '%Y%m%d%H%M')
t_end = datetime.datetime.strptime(cc.DATE_END, '%Y%m%d%H%M')
cp = datetime.timedelta(hours=cc.CYCLE_PERIOD/60)
n_per_day = int(np.maximum(1, 24*60/cc.CYCLE_PERIOD))

m_index = int(sys.argv[1])
relax_factor = float(sys.argv[2])  ##0: no pert; 1: full pert

orig_dir = cc.SCRATCH+"/data/GENERIC_PS_ATM/from_ERA5"
out_dir = cc.SCRATCH+"/data/GENERIC_PS_ATM/"+cc.EXP_NAME+"_{}/{:03d}".format(relax_factor, m_index)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

t = t_start
while t < t_end:
    ##copy original forcing file
    os.system("cp "+orig_dir+"/"+filename(t)+" "+out_dir+"/"+filename(t))
    dat = dict()
    for v in varname_pert:
        dat[v] = nc.read(out_dir+'/'+filename(t), v)

    dat_out = dat.copy()
    ##forcing file stored daily, perturb each time step within
    for n in range(n_per_day):
        t1 = t + n*cp
        n_index = int((t1-t_start)/cp)  ##perturbation index in time
        for v in varname_pert:
            mean = np.zeros((1, cc.NY, cc.NX))
            for s in range(ns):
                elapse_time = n_index*cc.CYCLE_PERIOD/60
                frac = np.maximum(0.0, 1.0 - elapse_time/growth_period[s])
                mean[0, :, :] += ms.get_scale(dat[v][n, :, :], krange, s) * frac
            pert = nc.Dataset(pert_file(m_index, n_index))[varname_pert[v]][0, :, :]
            dat_out[v][n, :, :] = relax_factor * (mean + pert - dat[v][n, :, :]) + dat[v][n, :, :]

    ##write output
    for v in varname_pert:
        nc.write(out_dir+'/'+filename(t), {'time':0, 'y':cc.NY, 'x':cc.NX}, v, dat_out[v])

    t += datetime.timedelta(days=1)
