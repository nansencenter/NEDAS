import numpy as np
import os
import sys
from NEDAS.models.qg import Model
from NEDAS.models.qg.util import read_data_bin, spec2grid
from NEDAS.utils.conversion import t2s, s2t, dt1h
from NEDAS.utils.netcdf_lib import nc_write_var

def main():
    model_name = sys.argv[1] ##'qg'
    nens = int(sys.argv[2]) ##20
    tstr = sys.argv[3]  ##'202301021200'

    mem_list = np.arange(nens)
    model = Model()

    kmax = 127
    nx = 2*(kmax+1)
    nz = 2
    path = '/cluster/work/users/yingyue/'+model_name+f'.n{nens}'+'/cycle/'+tstr+'/'+model_name
    t = s2t(tstr)

    vnames = ['temperature', 'streamfunc', 'vorticity']

    for m in mem_list:
        print(m)
        for n in [0, 1]:
            tout = t + n*12*dt1h
            for name in vnames:
                field = np.zeros((nz, nx, nx))
                for k in range(nz):
                    field[k,...] = model.read_var(name=name, path=path, member=m, time=tout, k=k)
                nc_write_var(path+'/ens.nc', {'t':None, 'm':None, 'z':nz, 'y':nx, 'x':nx}, name, field, recno={'t':n, 'm':m})

if __name__ == '__main__':
    main()

