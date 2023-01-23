import numpy as np
from netCDF4 import Dataset
from grid.multiscale import pwrspec2d
import grid.io.netcdf as nc
import sys

pert_path = sys.argv[1]
sample = int(sys.argv[2])
nt = int(sys.argv[3])
dt = int(sys.argv[4])
nup = int(sys.argv[5])

ke_spec = np.zeros((nup, nt+1))

for t in range(nt):
    filename = pert_path+'/{:03d}/perturb_{:04d}.nc'.format(sample, t*dt)
    u = nc.read(filename, 'x_wind_10m')[0, :, :]
    v = nc.read(filename, 'y_wind_10m')[0, :, :]
    u[np.where(np.isnan(u))] = 0.
    v[np.where(np.isnan(v))] = 0.
    wn, pwr_u = pwrspec2d(u)
    wn, pwr_v = pwrspec2d(v)
    ke_spec[:, t] = 0.5*(pwr_u+pwr_v)[:]

    np.save(pert_path+'/{:03d}/ke_spec.npy'.format(sample), ke_spec)
