import numpy as np
import os
from models.qg import Model
from models.qg.util import read_data_bin, spec2grid
from utils.conversion import t2s, s2t, dt1h
from utils.netcdf_lib import nc_write_var

mem_list = np.arange(20)
model = Model()

kmax = 127
nx = 2*(kmax+1)
nz = 2
path = '/cluster/work/users/yingyue/qg/cycle/202301011200/qg'
t = s2t('202301011200')

vnames = ['temperature', 'streamfunc', 'vorticity']

for m in mem_list:
    print(m)
    for n in [0]:
        tout = t + n*12*dt1h
        for name in vnames:
            field = np.zeros((nz, nx, nx))
            for k in range(nz):
                field[k,...] = model.read_var(name=name, path=path, member=m, time=tout, k=k)
            nc_write_var(path+'/ens.nc', {'t':None, 'm':None, 'z':nz, 'y':nx, 'x':nx}, name, field, recno={'t':n, 'm':m})

