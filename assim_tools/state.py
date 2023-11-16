import numpy as np
import sys
import struct
import importlib
from datetime import datetime, timedelta
from .common import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from .parallel import distribute_tasks, message

##top-level routine to prepare the state variables, getting them from model restart files,
##convert to the analysis grid, perform coarse-graining, and save to ens state in memory
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallelization
def process_state(c, comm, state_file):

    proc_id = comm.Get_rank()
    nproc = comm.Get_size()

    message(comm, 'process_state: parsing and generating field_info\n', 0)
    ##parse config and generate field_info, this is done by the first processor and broadcast
    if proc_id == 0:
        field_info = parse_field_info(c)
    else:
        field_info = None
    field_info = comm.bcast(field_info, root=0)

    nfld = len(field_info['fields'])
    nens = c.nens
    ny, nx = c.grid.x.shape

    ##one field at a time when read/write, a field have dims [nv, ny, nx]
    ##nfld is the number of fields

    ##each proc process part of the (nens, nfld), distribute_tasks gives proc_id:list of (mem_id, fld_id)
    fld_list = [(m, f) for m in range(nens) for f in range(nfld)]
    fld_list_proc = distribute_tasks(comm, fld_list)

    ##for each fld_id the field belongs to part of nfld in the final ens state [nens, nfld, ntile, nloc]

    ##the x,y domain is divided into square tiles with nx_tile grid points in each direction
    ##each tile contains [jstart:jend, istart:iend] as part of [0:ny, 0:nx]
    ##each proc process part of the ntile, distribute_tasks gives proc_id:list of (istart,iend,jstart,jend)
    nx_tile = int(np.round(np.sqrt(nx*ny/nproc/3)))
    tile_list = [(i, np.minimum(i+nx_tile, nx), j, np.minimum(j+nx_tile, ny))
                 for j in np.arange(0, ny, nx_tile) for i in np.arange(0, nx, nx_tile)]

    nloc_tile = np.array([np.sum((~c.mask[jstart:jend, istart:iend]).astype(int)) for istart,iend,jstart,jend in tile_list])

    nlobs_tile = 0 ###

    load_on_tile = np.maximum(nloc_tile, 1) * np.maximum(nlobs_tile, 1)
    tile_list_proc = distribute_tasks(comm, tile_list, load_on_tile)

    ##now start processing the fields, each processor gets its own workload as a subset of nfield
    message(comm, 'process_state: reading state varaible for each field record\n', 0)
    state = {}
    grid_bank = {}
    z_bank = {}

    nbatch_max = np.max([len(fld_lst) for pid,fld_lst in fld_list_proc.items()])
    for batch_id in range(nbatch_max):

        if batch_id < len(fld_list_proc[proc_id]):
            mem_id, fld_id = fld_list_proc[proc_id][batch_id]
            rec = field_info['fields'][fld_id]
            # message(comm, '   {:15s} t={} k={:5d} member={:3d} on proc{}\n'.format(rec['name'], rec['time'], rec['k'], mem_id+1, proc_id))

            ##directory storing model output
            path = c.work_dir+'/forecast/'+c.time+'/'+rec['source']

            ##load the module for handling source model
            src = importlib.import_module('models.'+rec['source'])

            ##only need to compute the uniq grids, stored them in bank for later use
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
            #     ##only need to compute the uniq z_coords, stored them in bank for later use
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


        ##for each proc_id with (mem_id, fld_id), send field[nv, ystart:yend, xstart:xend] to proc_id with (tile_i, tile_j)
        ##all proc_id send/recv to all, so use cyclic coreography
        for src_proc_id in np.arange(0, proc_id):
            ##receive tile from src_proc_id's fld
            if batch_id < len(fld_list_proc[src_proc_id]):
                src_mem_id, src_fld_id = fld_list_proc[src_proc_id][batch_id]
                for tile_id in range(len(tile_list_proc[proc_id])):
                    istart,iend,jstart,jend = tile_list_proc[proc_id][tile_id]  ##bounding box for tile
                    tile_j, tile_i = np.where(~c.mask[jstart:jend, istart:iend])  ##points unmasked to be saved in state vector
                    state[src_mem_id, src_fld_id, tile_id] = comm.recv(source=src_proc_id)

        if batch_id < len(fld_list_proc[proc_id]):
            for dst_proc_id in np.mod(np.arange(0, nproc)+proc_id, nproc):
                for tile_id in range(len(tile_list_proc[dst_proc_id])):
                    istart,iend,jstart,jend = tile_list_proc[dst_proc_id][tile_id]  ##bounding box for tile
                    tile_j, tile_i = np.where(~c.mask[jstart:jend, istart:iend])  ##points unmasked to be saved in state vector
                    fld_tile = fld[..., jstart:jend, istart:iend]
                    if dst_proc_id == proc_id:
                        ##just copy tile to state
                        state[mem_id, fld_id, tile_id] = fld_tile[..., tile_j, tile_i]
                    else:
                        ##send tile to dst_proc_id's state
                        comm.send(fld_tile[..., tile_j, tile_i], dest=dst_proc_id)

        for src_proc_id in np.arange(proc_id+1, nproc):
            ##receive tile from src_proc_id's fld
            if batch_id < len(fld_list_proc[src_proc_id]):
                src_mem_id, src_fld_id = fld_list_proc[src_proc_id][batch_id]
                for tile_id in range(len(tile_list_proc[proc_id])):
                    istart,iend,jstart,jend = tile_list_proc[proc_id][tile_id]  ##bounding box for tile
                    tile_j, tile_i = np.where(~c.mask[jstart:jend, istart:iend])  ##points unmasked to be saved in state vector
                    state[src_mem_id, src_fld_id, tile_id] = comm.recv(source=src_proc_id)

        ##show progress
        # message(comm, '\r|{:{}}| {:.0f}%'.format('='*int(np.ceil(batch_id/(nbatch_max-1)*50)), 50, (100/(nbatch_max-1)*batch_id)), 0)

    message(comm, ' Done.\n', 0)
    # np.savez('/cluster/work/users/yingyue/dat.{:04d}.npz'.format(proc_id), state=state)

    return state


##parse info for the nfld 2D fields in the state
##The entire state has dimensions: variable, time, z, y, x
##to organize the tasks of i/o and filter update, we consider 3 indices: mem_id, fld_id, loc_id
##mem_id indexes the rank in the ensemble
##loc_id indexes the horizontal position in a given 2D field defined on analysis grid
##fld_id indexes the remaining [variable, time, z] stacked into one dimension, nt and nz vary for
##   different variables, stacking them in one dimension helps better distribute i/o tasks
def parse_field_info(c):
    info = {'nx':c.nx, 'ny':c.ny, 'nens':c.nens, 'size':0, 'fields':{}}

    fld_id = 0   ##field id
    pos = 0      ##seek position for fld_id

    ##loop through variables in state_def
    for vname, vrec in c.state_def.items():
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
                info['fields'][fld_id] = rec

                ##update seek position and field id
                nv = 2 if src.variables[vname]['is_vector'] else 1
                fld_size = np.sum((~c.mask).astype(int))
                pos += nv * fld_size * type_size[src.variables[vname]['dtype']]
                fld_id += 1

    info['size'] = pos

    return info



##write field_info to a .dat file accompanying the bin file
def write_field_info(binfile, info):
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        ##first line: some dimensions
        f.write('{} {} {}\n'.format(info['nx'], info['ny'], info['nens']))

        ##followed by nfield lines: each for a field record
        for i, rec in info['field'].items():
            name = rec['name']
            source = rec['source']
            dtype = rec['dtype']
            is_vector = int(rec['is_vector'])
            units = rec['units']
            err_type = rec['err_type']
            member = 'None' if rec['member'] is None else rec['member']
            time = 'None' if rec['time'] is None else t2h(rec['time'])
            dt = 'None' if rec['dt'] is None else rec['dt']
            k = 'None' if rec['k'] is None else rec['k']
            pos = rec['pos']
            f.write('{} {} {} {} {} {} {} {} {} {} {}\n'.format(name, source, dtype, is_vector, units, err_type, member, time, dt, k, pos))


##read field_info from .dat file
def read_field_info(binfile):
    with open(binfile.replace('.bin','.dat'), 'r') as f:
        lines = f.readlines()

        ss = lines[0].split()
        info = {'nx':int(ss[0]), 'ny':int(ss[1]), 'nens':int(ss[2]), 'field':{}}

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
                   'member': None if ss[6]=='None' else int(ss[6]),
                   'time': None if ss[7]=='None' else h2t(np.float32(ss[7])),
                   'dt': None if ss[8]=='None' else np.float32(ss[8]),
                   'k': None if ss[9]=='None' else int(ss[9]),
                   'pos': int(ss[10]), }
            info['field'][rec_id] = rec
            rec_id += 1

    return info


##write a field fld with mem_id,fld_id to the binfile
def write_field(binfile, info, mask, mem_id, fld_id, fld):
    nx = info['nx']
    ny = info['ny']
    rec = info['fields'][fld_id]
    is_vector = rec['is_vector']
    fld_shape = (2, ny, nx) if is_vector else (ny, nx)
    assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

    fld_ = fld[:, ~mask].flatten() if is_vector else fld[~mask]
    with open(binfile, 'r+b') as f:
        f.seek(mem_id*info['size'] + rec['pos'])
        f.write(struct.pack(fld_.size*type_dic[rec['dtype']], *fld_))


##read a field from binfile, given mem_id, fld_id
def read_field(binfile, info, mask, mem_id, fld_id):
    nx = info['nx']
    ny = info['ny']
    rec = info['fields'][fld_id]
    nv = 2 if rec['is_vector'] else 1
    fld_shape = (2, ny, nx) if rec['is_vector'] else (ny, nx)
    fld_size = np.sum((~mask).astype(int))

    with open(binfile, 'rb') as f:
        f.seek(mem_id*info['size'] + rec['pos'])
        fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec['dtype']]),
                        f.read(nv*fld_size*type_size[rec['dtype']])))
        fld = np.full(fld_shape, np.nan)
        if rec['is_vector']:
            fld[:, ~mask] = fld_.reshape((2,-1))
        else:
            fld[~mask] = fld_
        return fld


#unmasked horizontal locale indices
# def loc_inds(mask):
#     ny, nx = mask.shape
#     ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
#     inds = jj * nx + ii
#     return inds[~mask]


##uniq field records for one member
# def uniq_fields(info):
#     ufid = 0
#     fields = {}
#     for fid, rec in info['fields'].items():
#         if rec['member'] == 0 and rec['name']!='z_coords':
#             for i in range(2 if rec['is_vector'] else 1):
#                 fields[ufid] = rec
#                 ufid += 1
#     return fields


##read the entire ensemble in local state space [nens, ufields, local_inds]
##local_inds: list of inds for horizontal locales [y,x]
##  Note: when used in parallelization, local_inds shall be continous chunks of inds
##        to avoid conflict in read/write of binfile; otherwise in single call, it is
##        fine to read a discontiguous chunk of field with arbitrary list of local_inds.
##nfield indexes the uniq fields at each locale with key (name, time, k)
##return: dict (nfield, local_inds): state[nens], and coordinates name,time,z,y,x
# def read_local_state(comm, binfile, info, mask, local_inds):
#     # from mpi4py import MPI

#     inds = loc_inds(mask)  ##horizontal locale indices for the entire field

#     fld_size = np.sum((~mask).astype(int))  ##size of the field rec in binfile
#     seek_inds = np.searchsorted(inds, local_inds)  ##seek pos in binfile for the local inds
#     chk_size = seek_inds[-1] - seek_inds[0] + 1   ##size of chunk to read from field rec

#     ##some dimensions for the local state
#     nens = info['nens']
#     nlocal = len(local_inds)
#     ufields = uniq_fields(info)  ##dict fid: uniq field rec for one member
#     nfield = len(ufields)

#     local_state = {'state': np.full((nens, nfield, nlocal), np.nan),
#                    'name': [r['name'] for r in ufields.values()],
#                    'time': [t2h(r['time']) for r in ufields.values()],
#                    'k': [r['k'] for r in ufields.values()],
#                    'z': np.full((nens, nfield, nlocal), np.nan), }
#                     ##TODO: there can be more than one z_coords for each uniq z_units

#     # f = MPI.File.Open(comm, binfile, MPI.MODE_RDONLY)
#     f = open(binfile, 'r+b')

#     ##loop through each field rec in binfile
#     for rec in info['fields'].values():
#         if comm.Get_rank()==0:
#             print(rec['pos'])
#             sys.stdout.flush()

#         ##read the chunk covering the local_inds
#         for ic in range(2 if rec['is_vector'] else 1):  ##vector fields have 2 components

#             seek_pos = rec['pos'] + (ic*fld_size+seek_inds[0])*type_size[rec['dtype']]
#             # chunk = np.empty(chk_size, dtype=type_convert[rec['dtype']])
#             # f.Read_at_all(seek_pos, chunk)
#             f.seek(seek_pos)
#             chunk = np.array(struct.unpack((chk_size*type_dic[rec['dtype']]), f.read(chk_size*type_size[rec['dtype']])))

#             if rec['name'] == 'z_coords':
#                 ##if this is a z_coords rec, assign its local_inds chunk to the corresponding z array
#                 for fid in [i for i,r in ufields.items() if r['time']==rec['time'] and r['k']==rec['k']]:
#                     local_state['z'][rec['member'], fid, :] = chunk[seek_inds-seek_inds[0]]

#             else:
#                 ##if this is a variable rec, assign the chunk to the state array
#                 fid_list = [i for i,r in ufields.items() if r['name']==rec['name'] and r['time']==rec['time'] and r['k']==rec['k']]
#                 local_state['state'][rec['member'], fid_list[ic], :] = chunk[seek_inds-seek_inds[0]]

#     f.close()
#     # f.Close()
#     # MPI.Finalize()

#     return local_state


# ##write the updated local_state [nens, nfield, local_inds] back to the binfile
# def write_local_state(comm, binfile, info, mask, local_inds, local_state):
#     # from mpi4py import MPI

#     inds = loc_inds(mask) ##horizontal locale indices for the entire field

#     fld_size = np.sum((~mask).astype(int))
#     seek_inds = np.searchsorted(inds, local_inds[0])
#     chk_size = len(local_inds) ##seek_inds[-1] - seek_inds[0] + 1

#     nens = info['nens']
#     nlocal = len(local_inds)
#     ufields = uniq_fields(info)  ##dict fid: uniq field rec for one member
#     nfield = len(ufields)
#     local_state_shape = local_state['state'].shape
#     assert local_state_shape==(nens, nfield, nlocal), f'local_state shape incorrect: expected {(nens, nfield, nlocal)}, got {local_state_shape}'

#     ##write the local local_state to binfiles
#     f = open(binfile, 'r+b')

#     for rec in info['fields'].values():
#         if comm.Get_rank()==0:
#             print(rec['pos'])
#             sys.stdout.flush()

#         for ic in range(2 if rec['is_vector'] else 1):

#             # if chk_size == nlocal:  ##chunk is contiguous, make an empty array
#             #     chunk = np.full(nlocal, np.nan)
#             # else:   ##chunk is discontiguous, first read the chunk from file
#             #     f.seek(rec['pos'] + (ic*fld_size+seek_inds[0])*type_size[rec['dtype']])
#             #     chunk = np.array(struct.unpack((chk_size*type_dic[rec['dtype']]), f.read(chk_size*type_size[rec['dtype']])))
#             # chunk = np.empty(nlocal, dtype=type_dic[rec['dtype']])

#             if rec['name'] == 'z_coords':
#                 pass ##we don't need to output z_coords, since they are not updated by local analysis
#             else:
#                 ##update value in chunk given the new local_state
#                 fid_list = [i for i,r in ufields.items() if r['name']==rec['name'] and r['time']==rec['time'] and r['k']==rec['k']]
#                 chunk = local_state['state'][rec['member'], fid_list[ic], :]
#                 ##write the chunk back to file
#                 f.seek(rec['pos'] + (ic*fld_size+seek_inds)*type_size[rec['dtype']])
#                 f.write(struct.pack((chk_size*type_dic[rec['dtype']]), *chunk))

#     f.close()


def output_state(comm, state, state_file):
    pass


