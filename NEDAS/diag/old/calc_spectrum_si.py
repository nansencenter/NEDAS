import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
from grid import pwrspec2d
import datetime
import sys
import os

t1 = datetime.datetime(2007, 2, 1, 6, 0, 0)

outdir = sys.argv[1]
member = int(sys.argv[2])
nt = int(sys.argv[3])
dt = int(sys.argv[4])
nup = int(sys.argv[5])


si_ke_spec = np.zeros((nup, nt+1))
si_ke_pert_spec = np.zeros((nup, nt+1))

for n in range(nt):
    t = t1+n*datetime.timedelta(hours=dt*6)
    tstr = t.strftime('%Y%m%dT%H%M%SZ')
    filename = outdir+'/{:03d}'.format(member)+'/output/'+tstr+'.nc'
    u = nc.read(filename, 'siu')[0, :, :]
    v = nc.read(filename, 'siv')[0, :, :]
    u[np.where(np.isnan(u))] = 0.
    v[np.where(np.isnan(v))] = 0.
    wn, pwr_u = pwrspec2d(u)
    wn, pwr_v = pwrspec2d(v)
    si_ke_spec[:, n] = 0.5*(pwr_u+pwr_v)[:]
    np.save(outdir+'/{:03d}'.format(member)+'/output/si_ke_spec.npy', si_ke_spec)

    filename = outdir+'/mean/output/'+tstr+'.nc'
    um = nc.read(filename, 'siu')[0, :, :]
    vm = nc.read(filename, 'siv')[0, :, :]
    um[np.where(np.isnan(um))] = 0.
    vm[np.where(np.isnan(vm))] = 0.
    wn, pwr_u = pwrspec2d(u-um)
    wn, pwr_v = pwrspec2d(v-vm)
    si_ke_pert_spec[:, n] = 0.5*(pwr_u+pwr_v)[:]
    np.save(outdir+'/{:03d}'.format(member)+'/output/si_ke_pert_spec.npy', si_ke_pert_spec)


