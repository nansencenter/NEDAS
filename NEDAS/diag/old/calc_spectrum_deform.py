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


deform_spec = np.zeros((nup, nt+1))
deform_pert_spec = np.zeros((nup, nt+1))

for n in range(nt):
    t = t1+n*datetime.timedelta(hours=dt*6)
    tstr = t.strftime('%Y%m%dT%H%M%SZ')
    filename = outdir+'/{:03d}'.format(member)+'/output/'+tstr+'.nc'
    deform = nc.read(filename, 'deform')[0, :, :]
    deform[np.where(np.isnan(deform))] = 0.
    wn, pwr = pwrspec2d(deform)
    deform_spec[:, n] = pwr[:]
    np.save(outdir+'/{:03d}'.format(member)+'/output/deform_spec.npy', deform_spec)

    filename = outdir+'/mean/output/'+tstr+'.nc'
    deformm = nc.read(filename, 'deform')[0, :, :]
    deformm[np.where(np.isnan(deformm))] = 0.
    wn, pwr = pwrspec2d(deform-deformm)
    deform_pert_spec[:, n] = pwr[:]
    np.save(outdir+'/{:03d}'.format(member)+'/output/deform_pert_spec.npy', deform_pert_spec)


