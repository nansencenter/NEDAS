import numpy as np
import sys
import struct
import importlib
from datetime import datetime, timedelta
import config as c
from conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from log import message, progress_bar
from parallel import bcast_by_root, distribute_tasks

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
@bcast_by_root(c.comm)
def parse_state_info(c):
    info = {'nx':c.nx, 'ny':c.ny, 'size':0, 'fields':{}, 'scalars':[]}
    rec_id = 0   ##record id for a 2D field
    pos = 0      ##seek position for rec

    ##loop through variables in state_def
    for vrec in c.state_def:
        vname = vrec['name']

        if vrec['state_type'] == 'field':
            ##this is a state variable 'field' with dimensions t, z, y, x
            ##some properties of the variable is defined in its source module
            src = importlib.import_module('models.'+vrec['source'])
            assert vname in src.variables, 'variable '+vname+' not defined in models.'+vrec['source']+'.variables'

            #now go through time and zlevels to form a uniq field record
            for time in s2t(c.time) + c.state_ts*timedelta(hours=1):  ##time slices
                for k in src.variables[vname]['levels']:  ##vertical levels
                    rec = { 'name': vname,
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

                    ##update seek position
                    nv = 2 if rec['is_vector'] else 1
                    fld_size = np.sum((~c.mask).astype(int))
                    pos += nv * fld_size * type_size[rec['dtype']]
                    rec_id += 1

        if vrec['state_type'] == 'scalar':
            ##this is a scalar (model parameter, etc.) to be updated
            ##since there is no difficulty storing the scalars on 1 proc
            ##we don't bother with parallelization (no rec_id needed)
            for time in s2t(c.time) + c.state_ts*timedelta(hours=1):  ##time slices
                rec = {'name': vname,
                       'source': vrec['source'],
                       'err_type': vrec['err_type'],
                       'time': time,
                      }
                info['scalars'].append(rec)

    message(c.comm, 'number of unique field records, nrec={}\n'.format(len(info['fields'])), 0)

    info['size'] = pos ##size of a complete state (fields) for 1 memeber

    return info


##write info to a .dat file accompanying the .bin file
def write_field_info(binfile, info):
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
def read_field_info(binfile):
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


##functions to prepare the state variables

def build_state_tasks(c):
    ##list of mem_id as tasks
    c.mem_list = distribute_tasks(c.comm_mem, [m for m in range(c.nens)])

    ##list rec_id as tasks
    rec_list_full = [i for i in c.state_info['fields'].keys()]
    rec_size = np.array([2 if r['is_vector'] else 1 for i,r in c.state_info['fields'].items()])
    c.rec_list = distribute_tasks(c.comm_rec, rec_list_full, rec_size)

    ##collect (mem_id, rec_id) together in field_list
    ##to make the task loop easier to read
    c.field_list = {}
    ntask = []
    for p in range(c.nproc):
        lst = [(m, r)
               for m in c.mem_list[p%c.nproc_mem]
               for r in c.rec_list[p//c.nproc_mem] ]
        ntask.append(len(lst))
        c.field_list[p] = lst
    c.field_ntask_max = np.max(ntask)


##prepare_state collects fields from model restart files, convert them to
##    the analysis grid, preprocess (coarse-graining etc), save to fields
##    with key (mem_id, rec_id) pointing to uniq fields
def prepare_state(c):

    message(c.comm, 'prepare state by reading fields from model restart\n', 0)
    fields = {}
    z_coords = {}
    grid_bank = {}
    z_bank = {}

    ##process the fields, each proc gets its own workload as a subset of
    ##field_list[pid] pointing to the list of tasks for each pid
    ##all pid goes through their own task list simultaneously
    for task in range(c.field_ntask_max):

        ##process a field record if not at the end of task list
        if task < len(c.field_list[c.pid]):
            ##this is the field to process in this task
            mem_id, rec_id = c.field_list[c.pid][task]
            rec = c.state_info['fields'][rec_id]
            # message(c.comm, '   {:15s} t={} k={:5d} member={:3d} on proc{}\n'.format(rec['name'], rec['time'], rec['k'], mem_id+1, c.pid), c.pid)

            ##directory storing model output
            path = c.work_dir+'/forecast/'+c.time+'/'+rec['source']

            ##load the module for handling source model
            src = importlib.import_module('models.'+rec['source'])

            ##only need to generate the uniq grid objs, stored them in memory bank
            member = mem_id if 'member' in src.uniq_grid_key else None
            var_name = rec['name'] if 'variable' in src.uniq_grid_key else None
            time = rec['time'] if 'time' in src.uniq_grid_key else None
            k = rec['k'] if 'k' in src.uniq_grid_key else None
            grid_key = (member, rec['source'], var_name, time, k)
            if grid_key in grid_bank:
                grid = grid_bank[grid_key]
            else:
                grid = src.read_grid(path, **rec)
                grid.dst_grid = c.grid
                grid_bank[grid_key] = grid

            ##read field and save to dict
            var = src.read_var(path, grid, member=mem_id, **rec)
            fld = grid.convert(var, is_vector=rec['is_vector'], method='linear', coarse_grain=True)
            fields[mem_id, rec_id] = fld

            ##read z_coords and save to dict
            ##only need to generate the uniq z coords, store in bank
            member = mem_id if 'member' in src.uniq_z_key else None
            var_name = rec['name'] if 'variable' in src.uniq_z_key else None
            time = rec['time'] if 'time' in src.uniq_z_key else None
            k = rec['k'] if 'k' in src.uniq_z_key else None
            z_key = (member, rec['source'], var_name, time, k)
            if z_key in z_bank:
                z = z_bank[z_key]
            else:
                zvar = src.z_coords(path, grid, member=mem_id, time=rec['time'], k=rec['k'])
                z = grid.convert(zvar, is_vector=False, method='linear', coarse_grain=True)
                z_bank[z_key] = z

            if rec['is_vector']:
                z_coords[mem_id, rec_id] = np.array([z, z])
            else:
                z_coords[mem_id, rec_id] = z

        message(c.comm, progress_bar(task, c.field_ntask_max), 0)

    message(c.comm, ' done.\n', 0)

    return fields, z_coords


##transpose_field_to_state send chunks of field owned by a pid to other pid
##  so that the field-complete fields get transposed into ensemble-complete state
##  with keys (mem_id, rec_id) pointing to the slices in loc_list
def transpose_field_to_state(c, fields):

    message(c.comm, 'transpose field to state\n', 0)
    state = {}

    ##all pid goes through their own task list simultaneously
    for task in range(c.field_ntask_max):

        ##prepare the fld for sending if not at the end of task list
        if task < len(c.field_list[c.pid]):
            mem_id, rec_id = c.field_list[c.pid][task]
            rec = c.state_info['fields'][rec_id]
            fld = fields[mem_id, rec_id]

        ## - for each source proc id (src_pid) with field_list item (mem_id, rec_id),
        ##   send chunk of fld[..., jstart:jend:dj, istart:iend:di] to destination
        ##   proc id (dst_pid) with its corresponding slice in loc_list
        ## - every pid needs to send/recv to/from every pid, so we use cyclic
        ##   coreography here to prevent deadlock

        ## 1) receive fld_chk from src_pid, for src_pid<pid first
        for src_pid in np.arange(0, c.pid):
            if task < len(c.field_list[src_pid]):
                src_mem_id, src_rec_id = c.field_list[src_pid][task]
                state[src_mem_id, src_rec_id] = c.comm.recv(source=src_pid)

        ## 2) send my fld chunk to a list of dst_pid, send to dst_pid>=pid first
        ##    because they wait to receive before being able to send their own stuff;
        ##    when finished with dst_pid>=pid, cycle back to send to dst_pid<pid,
        ##    i.e., dst_pid list = [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
        if task < len(c.field_list[c.pid]):
            for dst_pid in np.mod(np.arange(0, c.nproc)+c.pid, c.nproc):
                fld_chk = {}
                for loc in range(len(c.loc_list[dst_pid])):
                    ##slice for this loc
                    istart,iend,di,jstart,jend,dj = c.loc_list[dst_pid][loc]
                    ##save the unmasked points in slice to fld_chk for this loc
                    mask_chk = c.mask[jstart:jend:dj, istart:iend:di]
                    if rec['is_vector']:
                        fld_chk[loc] = fld[:, jstart:jend:dj, istart:iend:di][:, ~mask_chk]
                    else:
                        fld_chk[loc] = fld[jstart:jend:dj, istart:iend:di][~mask_chk]

                if dst_pid == c.pid:
                    ##same pid, so just write to state
                    state[mem_id, rec_id] = fld_chk
                else:
                    ##send fld_chk to dst_pid's state
                    c.comm.send(fld_chk, dest=dst_pid)

        ## 3) finish receiving fld_chk from src_pid, for src_pid>pid now
        for src_pid in np.arange(c.pid+1, c.nproc):
            if task < len(c.field_list[src_pid]):
                src_mem_id, src_rec_id = c.field_list[src_pid][task]
                state[src_mem_id, src_rec_id] = c.comm.recv(source=src_pid)

        if task < len(c.field_list[c.pid]):
            del fields[mem_id, rec_id]   ##free up memory

        message(c.comm, progress_bar(task, c.field_ntask_max), 0)

    message(c.comm, ' done.\n', 0)

    return state


##transpose_state_to_field transposes back the state to field-complete fields
def transpose_state_to_field(c, state):

    message(c.comm, 'transpose state to field\n', 0)
    fields = {}

    ##all pid goes through their own task list simultaneously
    for task in range(c.field_ntask_max):

        ##prepare an empty fld for receiving if not at the end of task list
        if task < len(c.field_list[c.pid]):
            mem_id, rec_id = c.field_list[c.pid][task]
            rec = c.state_info['fields'][rec_id]
            if rec['is_vector']:
                fld = np.full((2, c.ny, c.nx), np.nan)
            else:
                fld = np.full((c.ny, c.nx), np.nan)

        ##this is just the reverse transpose, see comments in transpose_field_to_state
        ## here, we take the exact steps, but swap send and recv operations
        ##
        ## 1) send my fld_chk to dst_pid, for dst_pid<pid first
        for dst_pid in np.arange(0, c.pid):
            if task < len(c.field_list[dst_pid]):
                dst_mem_id, dst_rec_id = c.field_list[dst_pid][task]
                c.comm.send(state[dst_mem_id, dst_rec_id], dest=dst_pid)
                del state[dst_mem_id, dst_rec_id]   ##free up memory

        ## 2) receive fld_chk from a list of src_pid, receive from src_pid>=pid first
        ##    because they wait to send stuff before being able to receive themselves,
        ##    cycle back to receive from src_pid<pid then.
        if task < len(c.field_list[c.pid]):
            for src_pid in np.mod(np.arange(0, c.nproc)+c.pid, c.nproc):
                if src_pid == c.pid:
                    ##same pid, so just copy fld_chk from state
                    fld_chk = state[mem_id, rec_id]
                else:
                    ##receive fld_chk from src_pid's state
                    fld_chk = c.comm.recv(source=src_pid)

                ##unpack the fld_chk to form a complete field
                for loc in range(len(c.loc_list[src_pid])):
                    istart,iend,di,jstart,jend,dj = c.loc_list[src_pid][loc]
                    mask_chk = c.mask[jstart:jend:dj, istart:iend:di]
                    fld[..., jstart:jend:dj, istart:iend:di][..., ~mask_chk] = fld_chk[loc]

                fields[mem_id, rec_id] = fld

        ## 3) finish sending fld_chk to dst_pid, for dst_pid>pid now
        for dst_pid in np.arange(c.pid+1, c.nproc):
            if task < len(c.field_list[dst_pid]):
                dst_mem_id, dst_rec_id = c.field_list[dst_pid][task]
                c.comm.send(state[dst_mem_id, dst_rec_id], dest=dst_pid)
                del state[dst_mem_id, dst_rec_id]   ##free up memory

        message(c.comm, progress_bar(task, c.field_ntask_max), 0)

    message(c.comm, ' done.\n', 0)

    return fields


##parallel output the fields to the binary state_file
def output_state(c, fields, state_file):

    message(c.comm, 'save state to '+state_file+'\n', 0)

    if c.pid == 0:
        ##if file doesn't exist, create the file
        open(state_file, 'wb')
        ##write state_info to the accompanying .dat file
        write_field_info(state_file, c.state_info)
    c.comm.Barrier()

    for task in range(c.field_ntask_max):

        if task < len(c.field_list[c.pid]):
            ##get the field record for output
            mem_id, rec_id = c.field_list[c.pid][task]
            fld = fields[mem_id, rec_id]

            ##write the data to binary file
            write_field(state_file, c.state_info, c.mask, mem_id, rec_id, fld)

        message(c.comm, progress_bar(task, c.field_ntask_max), 0)

    message(c.comm, ' done.\n', 0)


##compute ensemble mean of a field stored distributively on all pid_mem
##collect means on pid_mem=0, and output to mean_file
def output_ens_mean(c, fields, mean_file):

    message(c.comm, 'compute ensemble mean, save to '+mean_file+'\n', 0)
    if c.pid == 0:
        open(mean_file, 'wb')
        write_field_info(mean_file, c.state_info)
    c.comm.Barrier()

    for rec_id in c.rec_list[c.pid_rec]:

        ##initialize a zero field with right dimensions for rec_id
        if c.state_info['fields'][rec_id]['is_vector']:
            sum_fld_pid = np.zeros((2, c.ny, c.nx))
        else:
            sum_fld_pid = np.zeros((c.ny, c.nx))

        ##sum over all fields locally stored on pid
        for mem_id in c.mem_list[c.pid_mem]:
            sum_fld_pid += fields[mem_id, rec_id]

        ##sum over all field sums on different pids together to get the total sum
        sum_fld = c.comm_mem.reduce(sum_fld_pid, root=0)

        if c.pid_mem == 0:
            mean_fld = sum_fld / c.nens
            write_field(mean_file, c.state_info, c.mask, 0, rec_id, mean_fld)

        message(c.comm, progress_bar(rec_id, len(c.rec_list[c.pid_rec])), 0)

    message(c.comm, ' done.\n', 0)


