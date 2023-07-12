import numpy as np
from datetime import datetime
import importlib
from mpi4py import MPI
from assim_tools.basic_io import *
import sys
import config as c
current_time = datetime(2007, 1, 2)

## state[nens, nfield, ny, nx], nfield dimension contains nv,nt,nz flattened
## nv is number of variables, nt is time slices, nz is vertical layers,
## of course nt,nz vary for each variables, so we stack them in nfield dimension

## parallel scheme: given nproc
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
proc_id = comm.Get_rank()

ny, nx = c.ref_grid.x.shape

##mask for analysis grid
mask = np.load(c.WORK_DIR+'/mask.npy')

binfile = c.WORK_DIR+'/prior.bin'

if proc_id == 0:
    ##generate field info from state_def
    info = field_info(c.STATE_DEF_FILE,
                      current_time,
                      (c.OBS_WINDOW_MIN, c.OBS_WINDOW_MAX),
                      np.arange(c.NZ_MIN, c.NZ_MAX+1),
                      c.NUM_ENS,
                      *c.ref_grid.x.shape, mask)
    write_field_info(binfile, info)
    write_mask(binfile, info, mask)
else:
    info = None
info = comm.bcast(info, root=0)

grid_bank = {}

nfield = len(info['fields'])
nbatch = nfield//nproc
i_start = proc_id*nbatch
i_end = (proc_id+1)*nbatch  ##TODO: handle remainers

for i in range(i_start, i_end):
    rec = info['fields'][i]
    v = rec['var_name']
    t = rec['time']
    m = rec['member']
    z = rec['level']
    print(v, m, t, z)
    sys.stdout.flush()
    path = c.WORK_DIR + '/' + rec['source'].replace('.', '/')
    src = importlib.import_module(rec['source'])

    ##only need to compute the uniq grids, stored them in bank for later use
    grid_key = (rec['source'],)
    for key in src.uniq_grid:
        grid_key += (rec[key],)
    if grid_key in grid_bank:
        grid = grid_bank[grid_key]
    else:
        grid = src.get_grid(path, name=v, member=m, time=t, level=z)
        grid.dst_grid = c.ref_grid
        grid_bank[grid_key] = grid

    var = src.get_var(path, grid, name=v, member=m, time=t, level=z)
    fld = grid.convert(var, is_vector=rec['is_vector'])

    write_field(binfile, info, mask, i, fld)

