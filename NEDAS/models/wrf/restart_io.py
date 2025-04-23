import sys
import os
import argparse
import threading
import numpy as np
from netCDF4 import Dataset
from NEDAS.utils.parallel import distribute_tasks, Comm

def read_chunks(filename, chk_list, var_list):
    ##get the total dimensions for each variable
    var_dims = {}
    chunks = {}
    with Dataset(filename+'_0000', 'r') as f:
        for vname in var_list:
            dims = f[vname].dimensions
            if 'west_east' in dims:
                ni = f.getncattr('WEST-EAST_GRID_DIMENSION') - 1
                dim_i = 'west_east'
            elif 'west_east_stag' in dims:
                ni = f.getncattr('WEST-EAST_GRID_DIMENSION')
                dim_i = 'west_east_stag'
            else:
                raise ValueError("west_east dimension not found in "+vname)

            if 'south_north' in dims:
                nj = f.getncattr('SOUTH-NORTH_GRID_DIMENSION') - 1
                dim_j = 'south_north'
            elif 'south_north_stag' in dims:
                nj = f.getncattr('SOUTH-NORTH_GRID_DIMENSION')
                dim_j = 'south_north_stag'
            else:
                raise ValueError("south_north dimension not found in "+vname)

            if 'bottom_top' in dims:
                nk = f.getncattr('BOTTOM-TOP_GRID_DIMENSION') - 1
                dim_k = 'bottom_top'
            elif 'bottom_top_stag' in dims:
                nk = f.getncattr('BOTTOM-TOP_GRID_DIMENSION')
                dim_k = 'bottom_top_stag'
            else:
                nk = 1
                dim_k = 'surface'
            var_dims[vname] = (ni, nj, nk, dim_i, dim_j, dim_k)
            chunks[vname] = {}

    ##each processor read subset of chunks
    for chk_id in chk_list:
        with Dataset(filename+f'_{chk_id:04d}', 'r') as f:
            for vname in var_list:
                dims = f[vname].dimensions
                if 'west_east' in dims:
                    i1 = f.getncattr('WEST-EAST_PATCH_START_UNSTAG') - 1
                    i2 = f.getncattr('WEST-EAST_PATCH_END_UNSTAG')
                elif 'west_east_stag' in dims:
                    i1 = f.getncattr('WEST-EAST_PATCH_START_STAG') - 1
                    i2 = f.getncattr('WEST-EAST_PATCH_END_STAG')

                if 'south_north' in dims:
                    j1 = f.getncattr('SOUTH-NORTH_PATCH_START_UNSTAG') - 1
                    j2 = f.getncattr('SOUTH-NORTH_PATCH_END_UNSTAG')
                elif 'south_north_stag' in dims:
                    j1 = f.getncattr('SOUTH-NORTH_PATCH_START_STAG') - 1
                    j2 = f.getncattr('SOUTH-NORTH_PATCH_END_STAG')

                if 'bottom_top' in dims:
                    k1 = f.getncattr('BOTTOM-TOP_PATCH_START_UNSTAG') - 1
                    k2 = f.getncattr('BOTTOM-TOP_PATCH_END_UNSTAG')
                elif 'bottom_top_stag' in dims:
                    k1 = f.getncattr('BOTTOM-TOP_PATCH_START_STAG') - 1
                    k2 = f.getncattr('BOTTOM-TOP_PATCH_END_STAG')
                else:
                    k1 = 0
                    k2 = 1
                chunks[vname][chk_id] = (i1,i2, j1,j2, k1,k2, f[vname][0, ...])
    return var_dims, chunks

def write_chunks(filename, chk_list, var_list, chunks):
    for chk_id in chk_list:
        with Dataset(filename+f'_{chk_id:04d}', 'r+') as f:
            for vname in var_list:
                _,_, _,_, _,_, chk = chunks[vname][chk_id]
                f[vname][0, ...] = chk

def transpose_chunks_to_fields(comm, chk_list_pid, var_list_pid, var_dims, chunks):
    fields = {}
    pid = comm.Get_rank()
    nproc = comm.Get_size()

    nv_max = np.max([len(lst) for p,lst in var_list_pid.items()])
    for v in range(nv_max):
        if v < len(var_list_pid[pid]):
            ##prepare empty field for receiving chunks
            vname = var_list_pid[pid][v]
            ni, nj, nk, _,_,_ = var_dims[vname]
            fields[vname] = np.full((nk, nj, ni), np.nan)

        for dst_pid in np.arange(0, pid):
            if v < len(var_list_pid[dst_pid]):
                dst_vname = var_list_pid[dst_pid][v]
                comm.send(chunks[dst_vname], dest=dst_pid, tag=v)

        if v < len(var_list_pid[pid]):
            for src_pid in np.mod(np.arange(nproc)+pid, nproc):
                if src_pid == pid:
                    chks = chunks[vname]
                else:
                    chks = comm.recv(source=src_pid, tag=v)

                ##unpack chunks into the full field
                for chk_id in chk_list_pid[src_pid]:
                    i1,i2,j1,j2,k1,k2, chk = chks[chk_id]
                    fields[vname][k1:k2, j1:j2, i1:i2] = chk

        for dst_pid in np.arange(pid+1, nproc):
            if v < len(var_list_pid[dst_pid]):
                dst_vname = var_list_pid[dst_pid][v]
                comm.send(chunks[dst_vname], dest=dst_pid, tag=v)

    return fields

def transpose_fields_to_chunks():

    pass

def read_fields_bin(comm, filename, var_list, var_dims):

    return fields

def write_fields_bin(comm, filename, var_list, var_dims, fields):
    pid = comm.Get_rank()
    if pid == 0:
        with open(filename, 'w') as f:
            pass
    comm.Barrier()
    for vname in var_list:
        with open(filename, 'r+b') as f:
            f.seek()
            f.write()

def read_fields_nc(comm, filename, var_list, var_dims):

    return fields

def write_fields_nc(comm, filename, var_list, var_dims, fields):
    """output a joined nc file for wrfrst, one variable per file"""
    for vname in var_list:
        with Dataset(filename+'_'+vname+'.nc', 'w') as f:
            ni,nj,nk, dim_i,dim_j,dim_k = var_dims[vname]
            if dim_i not in f.dimensions:
                f.createDimension(dim_i, ni)
            if dim_j not in f.dimensions:
                f.createDimension(dim_j, nj)
            if dim_k not in f.dimensions:
                f.createDimension(dim_k, nk)
            f.createVariable(vname, float, (dim_k, dim_j, dim_i))
            f[vname][...] = fields[vname]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to read chunks of wrf restart files into a binary file, or reverse')
    parser.add_argument('mode', choices=['join', 'split'], help='operation mode: join or split')
    parser.add_argument('filename', help='wrf restart file name: wrfrst_d01_<time>')
    parser.add_argument('nchk', default=1, type=int, help='number of processors (chunks)')
    args = parser.parse_args()

    comm = Comm()
    pid = comm.Get_rank()

    chk_list = np.arange(args.nchk)
    chk_list_pid = distribute_tasks(comm, chk_list)

    var_list = ['U_1', 'V_1', 'W_1', 'T', 'PH_1', 'P', 'MU_1', 'QVAPOR', 'QCLOUD']
    var_list_pid = distribute_tasks(comm, var_list)
    full_file = os.path.join(os.path.dirname(args.filename), 'restart')

    if args.mode == 'join':
        var_dims, chunks = read_chunks(args.filename, chk_list_pid[pid], var_list)
        fields = transpose_chunks_to_fields(comm, chk_list_pid, var_list_pid, var_dims, chunks)
        write_fields_nc(comm, full_file, var_list_pid[pid], var_dims, fields)

    # elif args.mode == 'split':
    #     split_chunks(args.filename, chk_list_pid[pid], var_list, full_file)

