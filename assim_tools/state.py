import numpy as np
import sys
import struct
import importlib
from datetime import datetime, timedelta
from .conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from .log import message, progress_bar
from .parallel import distribute_tasks

##Note: The analysis is performed on a regular grid.
## The entire state has dimensions: member, variable, time,  z,  y,  x
##                      indexed by: mem_id,        v,    t,  k,  j,  i
##                       with size:   nens,       nv,   nt, nz, ny, nx
##
## To parallelize workload, we group the dimensions into 3 indices:
## mem_id indexes the ensemble members
## rec_id indexes the uniq 2D fields with (v, t, k), since nz and nt may vary
##          for different variables, we stack these dimensions in the 'record'
##          dimension with size nrec
## loc_id indexes the location (j, i) in the 2D field.
##
## The entire state is distributed across the memory of many processors,
## at any moment, a processor only stores a subset of state in its memory:
## either having all the mem_id,rec_id but only a subset of loc_id (we call this
## ensemble-complete), or having all the loc_id but a subset of mem_id,rec_id
## (we call this field-complete).
## It is easier to perform i/o and pre/post processing on field-complete state,
## while easier to run assimilation algorithms with ensemble-complete state.


## parses info for the nrec fields in the state.
## input: c: config module with the environment variables
## returns: info dict with some dimensions and list of uniq field records
def parse_state_info(c):
    info = {'nx':c.nx, 'ny':c.ny, 'size':0, 'fields':{}}
    rec_id = 0   ##record id for a 2D field
    pos = 0      ##seek position for rec_id

    ##loop through variables in state_def
    for vrec in c.state_def:
        vname = vrec['name']
        ##some properties of the variable is defined in its source module
        src = importlib.import_module('models.'+vrec['source'])
        assert vname in src.variables, 'variable '+vname+' not defined in models.'+vrec['source']+'.variables'

        #now go through time and zlevels to form a uniq field record
        for time in s2t(c.time) + c.state_ts*timedelta(hours=1):  ##time slices
            for k in src.variables[vname]['levels']:  ##vertical levels
                rec = {'name': vname,
                       'source': vrec['source'],
                       'dtype': src.variables[vname]['dtype'],
                       'is_vector': src.variables[vname]['is_vector'],
                       'units': src.variables[vname]['units'],
                       'err_type': vrec['err_type'],
                       'time': time,
                       'dt': c.t_scale,
                       'k': k,
                       'pos': pos, }
                info['fields'][rec_id] = rec

                ##update seek position and field id
                nv = 2 if rec['is_vector'] else 1
                fld_size = np.sum((~c.mask).astype(int))
                pos += nv * fld_size * type_size[rec['dtype']]
                rec_id += 1

    info['size'] = pos ##size of a complete state for one memeber

    return info


##write info to a .dat file accompanying the .bin file
def write_state_info(binfile, info):
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        ##first line: some dimension sizes
        f.write('{} {} {}\n'.format(info['nx'], info['ny'], info['size']))

        ##followed by nfield lines: each for a field record
        for i, rec in info['fields'].items():
            name = rec['name']
            source = rec['source']
            dtype = rec['dtype']
            is_vector = int(rec['is_vector'])
            units = rec['units']
            err_type = rec['err_type']
            time = t2h(rec['time'])
            dt = rec['dt']
            k = rec['k']
            pos = rec['pos']
            f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(name, source, dtype, is_vector, units, err_type, time, dt, k, pos))


##read info from .dat file
def read_state_info(binfile):
    with open(binfile.replace('.bin','.dat'), 'r') as f:
        lines = f.readlines()

        ss = lines[0].split()
        info = {'nx':int(ss[0]), 'ny':int(ss[1]), 'size':int(ss[2]), 'fields':{}}

        ##records for uniq fields
        rec_id = 0
        for lin in lines[1:]:
            ss = lin.split()
            rec = {'name': ss[0],
                   'source': ss[1],
                   'dtype': ss[2],
                   'is_vector': bool(int(ss[3])),
                   'units': ss[4],
                   'err_type': ss[5],
                   'time': h2t(np.float32(ss[6])),
                   'dt': np.float32(ss[7]),
                   'k': int(ss[8]),
                   'pos': int(ss[9]), }
            info['fields'][rec_id] = rec
            rec_id += 1

    return info


##write a field fld with mem_id,rec_id to binfile
def write_field(binfile, info, mask, mem_id, rec_id, fld):
    ny = info['ny']
    nx = info['nx']
    rec = info['fields'][rec_id]

    fld_shape = (2, ny, nx) if rec['is_vector'] else (ny, nx)
    assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

    if rec['is_vector']:
        fld_ = fld[:, ~mask].flatten()
    else:
        fld_ = fld[~mask]

    with open(binfile, 'r+b') as f:
        f.seek(mem_id*info['size'] + rec['pos'])
        f.write(struct.pack(fld_.size*type_dic[rec['dtype']], *fld_))


##read a field from binfile, given mem_id, rec_id
def read_field(binfile, info, mask, mem_id, rec_id):
    ny = info['ny']
    nx = info['nx']
    rec = info['fields'][rec_id]
    nv = 2 if rec['is_vector'] else 1

    fld_shape = (2, ny, nx) if rec['is_vector'] else (ny, nx)
    fld_size = np.sum((~mask).astype(int))

    with open(binfile, 'rb') as f:
        f.seek(mem_id*info['size'] + rec['pos'])
        fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec['dtype']]),
                        f.read(nv*fld_size*type_size[rec['dtype']])))
        fld = np.full(fld_shape, np.nan)
        if rec['is_vector']:
            fld[:, ~mask] = fld_.reshape((2, -1))
        else:
            fld[~mask] = fld_
        return fld


##functions to prepare the state variables, some common inputs:
##  c: config module with env variables
##  comm: mpi4py communicator for multiple processors
##  state_info: dict with information on field records
##  fld_list_proc: list of field records to work on for each pid,
##                 each item is (mem_id,rec_id) referring to a uniq field
##  loc_list_proc: list of spatial locations to work on for each pid,
##                 each item is (istart,iend,di,jstart,jend,dj) that
##                 describes a slice of the domain

##prepare_state collects fields from model restart files, convert them to
##    the analysis grid, preprocess (coarse-graining etc), save to fields
##    with key (mem_id, rec_id) pointing to uniq fields
def prepare_state(c, comm, state_info, fld_list_proc):
    pid = comm.Get_rank()
    nproc = comm.Get_size()

    message(comm, 'prepare state by reading fields from model restart\n', 0)
    fields = {}
    grid_bank = {}
    z_bank = {}

    ##process the fields, each proc gets its own workload as a subset of fld_list:
    ##fld_list_proc[pid] points to the list of tasks for each pid
    ntask_max = np.max([len(lst) for pid,lst in fld_list_proc.items()])

    ##all proc goes through their own task list simultaneously
    for task in range(ntask_max):
        message(comm, progress_bar(task, ntask_max), 0)

        ##process a field record if not at the end of task list
        if task < len(fld_list_proc[pid]):
            ##this is the field to process in this task
            mem_id, rec_id = fld_list_proc[pid][task]
            rec = state_info['fields'][rec_id]
            # message(comm, '   {:15s} t={} k={:5d} member={:3d} on proc{}\n'.format(rec['name'], rec['time'], rec['k'], mem_id+1, pid))

            ##directory storing model output
            path = c.work_dir+'/forecast/'+c.time+'/'+rec['source']

            ##load the module for handling source model
            src = importlib.import_module('models.'+rec['source'])

            ##only need to generate the uniq grid objs, stored them in memory bank
            member = mem_id if 'member' in src.uniq_grid_key else None
            time = rec['time'] if 'time' in src.uniq_grid_key else None
            k = rec['k'] if 'k' in src.uniq_grid_key else None
            grid_key = (rec['source'], member, time, k)
            if grid_key in grid_bank:
                grid = grid_bank[grid_key]
            else:
                grid = src.read_grid(path, **rec)
                grid.dst_grid = c.grid
                grid_bank[grid_key] = grid

            # if rec['name'] == 'z_coords':
            #     ##only need to compute the uniq z_coords, stored them in bank
            #     member = rec['member'] if 'member' in src.uniq_z_key else None
            #     time = rec['time'] if 'time' in src.uniq_z_key else None
            #     z_key = (rec['source'], rec['units'], member, time)
            #     if z_key in z_bank:
            #         fld = z_bank[z_key]
            #     else:
            #         var = src.z_coords(path, grid, **rec)
            #         fld = grid.convert(var, method='linear', coarse_grain=True)

            var = src.read_var(path, grid, member=mem_id, **rec)
            fld = grid.convert(var, is_vector=rec['is_vector'], method='linear', coarse_grain=True)
            fields[mem_id, rec_id] = fld

    message(comm, ' Done.\n', 0)

    return fields


##transpose_field_to_state send chunks of field owned by a pid to other pid
##  so that the field-complete fields get transposed into ensemble-complete state
##  with keys (mem_id, rec_id) pointing to the slices in loc_list_proc
def transpose_field_to_state(c, comm, state_info, fld_list_proc, loc_list_proc, fields):
    pid = comm.Get_rank()
    nproc = comm.Get_size()

    message(comm, 'transpose state from field-complete to ensemble-complete\n', 0)
    state = {}

    ntask_max = np.max([len(lst) for pid,lst in fld_list_proc.items()])

    ##all proc goes through their own task list simultaneously
    for task in range(ntask_max):

        ##prepare the fld for sending if not at the end of task list
        if task < len(fld_list_proc[pid]):
            mem_id, rec_id = fld_list_proc[pid][task]
            rec = state_info['fields'][rec_id]
            fld = fields[mem_id, rec_id]

        ## - for each source proc id (src_pid) with fld_list item (mem_id, rec_id),
        ##   send chunk of fld[..., jstart:jend:dj, istart:iend:di] to destination
        ##   proc id (dst_pid) with its corresponding slice in loc_list_proc
        ## - every pid needs to send/recv to/from every pid, so we use cyclic
        ##   coreography here to prevent deadlock

        ## 1) receive fld_chk from src_pid, for src_pid<pid first
        for src_pid in np.arange(0, pid):
            if task < len(fld_list_proc[src_pid]):
                src_mem_id, src_rec_id = fld_list_proc[src_pid][task]
                state[src_mem_id, src_rec_id] = comm.recv(source=src_pid)

        ## 2) send my fld chunk to a list of dst_pid, send to dst_pid>=pid first
        ##    because they wait to receive before being able to send their own stuff;
        ##    when finished with dst_pid>=pid, cycle back to send to dst_pid<pid,
        ##    i.e., dst_pid list = [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
        if task < len(fld_list_proc[pid]):
            for dst_pid in np.mod(np.arange(0, nproc)+pid, nproc):
                fld_chk = {}
                for loc in range(len(loc_list_proc[dst_pid])):
                    ##slice for this loc
                    istart,iend,di,jstart,jend,dj = loc_list_proc[dst_pid][loc]
                    ##save the unmasked points in slice to fld_chk for this loc
                    mask_chk = c.mask[jstart:jend:dj, istart:iend:di]
                    if rec['is_vector']:
                        fld_chk[loc] = fld[:, jstart:jend:dj, istart:iend:di][:, ~mask_chk]
                    else:
                        fld_chk[loc] = fld[jstart:jend:dj, istart:iend:di][~mask_chk]

                if dst_pid == pid:
                    ##same pid, so just write to state
                    state[mem_id, rec_id] = fld_chk
                else:
                    ##send fld_chk to dst_pid's state
                    comm.send(fld_chk, dest=dst_pid)

        ## 3) finish receiving fld_chk from src_pid, for src_pid>pid now
        for src_pid in np.arange(pid+1, nproc):
            if task < len(fld_list_proc[src_pid]):
                src_mem_id, src_rec_id = fld_list_proc[src_pid][task]
                state[src_mem_id, src_rec_id] = comm.recv(source=src_pid)

        if task < len(fld_list_proc[pid]):
            del fields[mem_id, rec_id]   ##free up memory

    return state


##transpose_state_to_field transposes back the state to field-complete fields
def transpose_state_to_field(c, comm, state_info, fld_list_proc, loc_list_proc, state):
    pid = comm.Get_rank()
    nproc = comm.Get_size()

    message(comm, 'transpose state from ensemble-complete to field-complete\n', 0)
    fields = {}

    ntask_max = np.max([len(lst) for pid,lst in fld_list_proc.items()])

    ##all proc goes through their own task list simultaneously
    for task in range(ntask_max):

        ##prepare an empty fld for receiving if not at the end of task list
        if task < len(fld_list_proc[pid]):
            mem_id, rec_id = fld_list_proc[pid][task]
            rec = state_info['fields'][rec_id]
            if rec['is_vector']:
                fld = np.full((2, c.ny, c.nx), np.nan)
            else:
                fld = np.full((c.ny, c.nx), np.nan)

        ##this is just the reverse transpose, see comments in transpose_field_to_state
        ## here, we take the exact steps, but swap send and recv operations
        ##
        ## 1) send my fld_chk to dst_pid, for dst_pid<pid first
        for dst_pid in np.arange(0, pid):
            if task < len(fld_list_proc[dst_pid]):
                dst_mem_id, dst_rec_id = fld_list_proc[dst_pid][task]
                comm.send(state[dst_mem_id, dst_rec_id], dest=dst_pid)
                del state[dst_mem_id, dst_rec_id]   ##free up memory

        ## 2) receive fld_chk from a list of src_pid, receive from src_pid>=pid first
        ##    because they wait to send stuff before being able to receive themselves,
        ##    cycle back to receive from src_pid<pid then.
        if task < len(fld_list_proc[pid]):
            for src_pid in np.mod(np.arange(0, nproc)+pid, nproc):
                if src_pid == pid:
                    ##same pid, so just copy fld_chk from state
                    fld_chk = state[mem_id, rec_id]
                else:
                    ##receive fld_chk from src_pid's state
                    fld_chk = comm.recv(source=src_pid)

                ##unpack the fld_chk to form a complete field
                for loc in range(len(loc_list_proc[src_pid])):
                    istart,iend,di,jstart,jend,dj = loc_list_proc[src_pid][loc]
                    mask_chk = c.mask[jstart:jend:dj, istart:iend:di]
                    fld[..., jstart:jend:dj, istart:iend:di][..., ~mask_chk] = fld_chk[loc]

                fields[mem_id, rec_id] = fld

        ## 3) finish sending fld_chk to dst_pid, for dst_pid>pid now
        for dst_pid in np.arange(pid+1, nproc):
            if task < len(fld_list_proc[dst_pid]):
                dst_mem_id, dst_rec_id = fld_list_proc[dst_pid][task]
                comm.send(state[dst_mem_id, dst_rec_id], dest=dst_pid)
                del state[dst_mem_id, dst_rec_id]   ##free up memory

    return fields


def output_state(c, comm, state_info, fld_list_proc, fields, state_file):
    pid = comm.Get_rank()
    nproc = comm.Get_size()

    message(comm, 'save state to '+state_file+'\n', 0)

    if pid == 0:
        ##if file doesn't exist, create the file
        open(state_file, 'wb')
        ##write state_info to the accompanying .dat file
        write_state_info(state_file, state_info)
    comm.Barrier()

    ntask_max = np.max([len(lst) for pid,lst in fld_list_proc.items()])
    for task in range(ntask_max):
        message(comm, progress_bar(task, ntask_max), 0)

        if task < len(fld_list_proc[pid]):
            ##get the field record for output
            mem_id, rec_id = fld_list_proc[pid][task]
            fld = fields[mem_id, rec_id]

            ##write the data to binary file
            write_field(state_file, state_info, c.mask, mem_id, rec_id, fld)

    message(comm, ' Done.\n', 0)


