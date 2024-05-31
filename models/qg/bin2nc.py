import numpy as np
from utils.parallel import Comm, distribute_tasks
from models.qg.util import read_data_bin, spec2grid
from netcdf_lib import nc_write_var

comm = Comm()

mem_list = np.arange(100)

pid = comm.Get_rank()
mem_list_pid = distribute_tasks(comm, mem_list)

kmax = 127
nx = 2*(kmax+1)
nz = 2

for m in mem_list_pid[pid]:
    print(m, flush=True)
    for t in range(50, 151):
        psi = np.zeros((nz, nx, nx))
        for k in range(nz):
            fieldk = read_data_bin('{:04d}/psi.bin'.format(m+1), kmax, nz, k, t)
            field = spec2grid(fieldk).T
            psi[k, ...] = field
        nc_write_var('/cluster/work/users/yingyue/qg_output/{:04d}/{:03d}.nc'.format(m+1, t-50), {'t':None, 'z':nz, 'y':nx, 'x':nx}, 'psi', psi, recno={'t':0})

