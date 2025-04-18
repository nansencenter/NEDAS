import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
from pynextsim import NextsimBin
from datetime import datetime, timedelta
import sys
import os

##date range
t1 = datetime(2007, 1, 1, 6, 0, 0)
dt = timedelta(hours=6)
nt = 84
x = grid.x_ref
y = grid.y_ref
nx, ny = x.shape

outdir = sys.argv[1]
v = int(sys.argv[2]) ##var id

##output variable
vname = ('x_wind_10m', 'y_wind_10m')[v]
voname = ('x_wind_10m', 'y_wind_10m')[v]

##read bin files
out = np.zeros((nx, ny))

for n in range(nt):
    t = t1 + n*dt

    t0 = datetime(t.year, t.month, t.day, 0, 0, 0)
    t_index = int((t - t0)/dt)
    infile = outdir+'/data/GENERIC_PS_ATM/generic_ps_atm_'+t.strftime('%Y%m%d')+'.nc'
    f = nc.Dataset(infile)
    out = f[vname][t_index, :, :]

    ##output nc file
    out_path = outdir+'/output/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_dat = np.zeros((1, ny, nx))
    out_dat[0, :, :] = out
    nc.write(out_path+t.strftime('%Y%m%dT%H%M%SZ')+'.nc', {'t':0, 'y':ny, 'x':nx}, voname, out_dat)

