import numpy as np
import grid.io.netcdf as nc
import config.constants as cc
import os
import sys


domain_size = np.maximum(cc.NX, cc.NY) * cc.DX/1e3
ns = cc.PERTURB_NUM_SCALE
nens = cc.NUM_ENS
nens_batch = cc.PERTURB_NUM_ENS
varname = ('uwind', 'vwind')
varname_out = ('x_wind_10m', 'y_wind_10m')

m = int(sys.argv[1])-1
nt = int(sys.argv[2])

for t in range(nt):
    for v in range(len(varname)):
        pert = np.zeros((1, cc.NY, cc.NX))
        for s in range(ns):
            pert_file = cc.WORK_DIR+'/run/perturbation/mem{:03d}/scale{}/perturb_{:04d}.nc'.format(int(m/nens_batch)+1, s+1, t)
            pert += nc.read(pert_file, varname[v])[:, np.mod(m, nens_batch), :, :]

        out_path = cc.SCRATCH+'/perturbation/{:03d}'.format(m+1)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        out_file = out_path+'/perturb_{:04d}.nc'.format(t)
        nc.write(out_file, {'t':0, 'y':cc.NY, 'x':cc.NX}, varname_out[v], pert)
