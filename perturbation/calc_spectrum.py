import numpy as np
from netCDF4 import Dataset
from grid.multiscale import pwrspec2d
import grid.io.netcdf as nc
import sys

pert_path = sys.argv[1] #'/cluster/work/users/yingyue/perturb_param/sample_ECMWF/'
n_sample = int(sys.argv[2]) #47
nt = int(sys.argv[3]) #40
dt = int(sys.argv[4]) #4
nup = int(sys.argv[5]) #917

ke_spec = np.zeros((n_sample, nup, nt+1))

for sample in range(n_sample):
    print(sample+1)

    for t in range(1, nt+1):
        filename = pert_path+'/{:03d}/{:04d}/perturb.nc'.format(sample+1, t*dt)
        u = nc.read(filename, 'x_wind_10m')[0, :, :]
        v = nc.read(filename, 'y_wind_10m')[0, :, :]
        u[np.where(np.isnan(u))] = 0.
        v[np.where(np.isnan(v))] = 0.
        wn, pwr_u = pwrspec2d(u)
        wn, pwr_v = pwrspec2d(v)
        ke_spec[sample, :, t] = 0.5*(pwr_u+pwr_v)[:]

    np.save(pert_path+'/ke_spec.npy', ke_spec)
