import numpy as np
import struct
import importlib
from datetime import datetime, timedelta
from .common import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from .parallel import distribute_tasks, message

##top-level routine to prepare the state variables, getting them from model restart files,
##convert to the analysis grid, perform coarse-graining, and save to prior_state.bin
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallelization
def process_state(c, comm, prior_binfile, post_binfile):
    proc_id = comm.Get_rank()
    message(comm, 'process_state: parsing and generating field_info', 0)
    ##parse config and generate field_info, this is done by the first processor and broadcast
    if proc_id == 0:
        info = field_info(c)
        for binfile in [prior_binfile, post_binfile]:
            with open(binfile, 'wb'):  ##initialize file in case doesn't exist
                pass
            write_field_info(binfile, info)
            write_header(binfile, info, c.grid.x, c.grid.y, c.mask)
    else:
        info = None
    info = comm.bcast(info, root=0)

    grid_bank = {}
    z_bank = {}

    ##now start processing the fields, each processor gets its own workload as a subset of nfield
    message(comm, 'process_state: reading state varaible for each field record', 0)
    for fld_id in distribute_tasks(comm, np.arange(len(info['fields'])))[proc_id]:
        rec = info['fields'][fld_id]

        ##directory storing model output
        path = c.work_dir+'/forecast/'+c.time+'/'+rec['source']

        ##load the module for handling source model
        src = importlib.import_module('models.'+rec['source'])

        ##only need to compute the uniq grids, stored them in bank for later use
        member = rec['member'] if 'member' in src.uniq_grid_key else None
        time = rec['time'] if 'time' in src.uniq_grid_key else None
        k = rec['k'] if 'k' in src.uniq_grid_key else None
        grid_key = (rec['source'], member, time, k)
        if grid_key in grid_bank:
            grid = grid_bank[grid_key]
        else:
            grid = src.read_grid(path, **rec)
            grid.dst_grid = c.grid
            grid_bank[grid_key] = grid

        if rec['name'] == 'z_coords':
            ##only need to compute the uniq z_coords, stored them in bank for later use
            member = rec['member'] if 'member' in src.uniq_z_key else None
            time = rec['time'] if 'time' in src.uniq_z_key else None
            z_key = (rec['source'], rec['units'], member, time)
            if z_key in z_bank:
                fld = z_bank[z_key]
            else:
                var = src.z_coords(path, grid, **rec)
                fld = grid.convert(var, method='linear', coarse_grain=True)
        else:
            var = src.read_var(path, grid, **rec)
            fld = grid.convert(var, is_vector=rec['is_vector'], method='linear', coarse_grain=True)

        for binfile in [prior_binfile, post_binfile]:
            write_field(binfile, info, c.mask, fld_id, fld)

        # message(comm, '   {:15s} t={} k={:5d} member={:3d}'.format(rec['name'], rec['time'], rec['k'], rec['member']+1))


##generate info for the nens*nfield 2D fields in the state
##The entire ensemble state has dimensions: member, variable, time, z, y, x
##to organize the tasks of i/o and filter update, we consider 3 indices: member, field, locale
##member indexes the rank in the ensemble
##locale indexes the horizontal position in a given 2D field defined on analysis grid
##field indexes the remaining [variable, time, z] stacked into one dimension, nt and nz vary for
##   different variables, stacking them in one dimension helps better distribute i/o tasks
def field_info(c):
    info = {'nx':c.nx, 'ny':c.ny, 'nens': c.nens, 'fields':{}}

    fld_id = 0   ##field id
    pos = 0      ##f.seek position
    pos += c.ny * c.nx * (2*type_size['float'] + type_size['int'])  ##x,y,mask are the first 3 records

    ##loop through variables in state_def
    for name, rec in c.state_def.items():
        ##some properties of the variable is defined in its source module
        src = importlib.import_module('models.'+rec['source'])
        assert name in src.variables, 'variable '+name+' not defined in models.'+rec['source']+'.variables'

        #now go through member, time, z to form a uniq field record
        for member in range(c.nens):  ##ensemble members
            for time in s2t(c.time) + c.state_ts*timedelta(hours=1):  ##time slices
                for k in src.variables[name]['levels']:  ##vertical levels
                    kwargs = {'name': name,
                              'source': rec['source'],
                              'dtype': src.variables[name]['dtype'],
                              'is_vector': src.variables[name]['is_vector'],
                              'units': src.variables[name]['units'],
                              'err_type': rec['err_type'],
                              'member': member,
                              'time': time,
                              'dt': c.t_scale,
                              'k': k,
                              'pos': pos, }
                    info['fields'][fld_id] = kwargs

                    ##update f.seek position
                    nv = 2 if src.variables[name]['is_vector'] else 1
                    fld_size = np.sum((~c.mask).astype(int))
                    pos += nv * fld_size * type_size[src.variables[name]['dtype']]

                    fld_id += 1

    z_units_list = set()  ##uniq z_units for obs variables in obs_def
    for obs_name in c.obs_def:
        obs_src = importlib.import_module('dataset.'+c.obs_def[obs_name]['source'])
        z_units = obs_src.variables[obs_name]['z_units']
        if z_units is not None:
            z_units_list.add(z_units)

    ##loop over all field records, add z_coords record if uniq
    z_keys = set()
    for rec in info['fields'].copy().values():
        for z_units in z_units_list:
            kwargs = {'name': 'z_coords',
                      'source': rec['source'],
                      'dtype': 'float',
                      'is_vector': False,
                      'units': z_units,
                      'err_type': 'normal',
                      'member': rec['member'],
                      'time': rec['time'],
                      'dt' : rec['dt'],
                      'k': rec['k'],
                      'pos': pos, }
            key = (rec['source'], z_units, rec['member'], rec['time'], rec['k'])
            if key not in z_keys:
                info['fields'][fld_id] = kwargs
                z_keys.add(key)
                pos += np.sum((~c.mask).astype(int)) * type_size['float']
                fld_id += 1

    return info


##write field_info to a .dat file accompanying the bin file
def write_field_info(binfile, info):
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        ##first line: some dimensions
        f.write('{} {} {}\n'.format(info['nx'], info['ny'], info['nens']))

        ##followed by nfield lines: each for a field record
        for i, rec in info['fields'].items():
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
        info = {'nx':int(ss[0]), 'ny':int(ss[1]), 'nens':int(ss[2]), 'fields':{}}

        ##records for uniq fields
        fld_id = 0
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
            info['fields'][fld_id] = rec
            fld_id += 1

    return info


##write x,y,mask to the first records in binfile
def write_header(binfile, info, x, y, mask):
    nx = info['nx']
    ny = info['ny']
    assert x.shape == (ny, nx), f'x shape incorrect: expected {(ny, nx)}, got {x.shape}'
    assert y.shape == (ny, nx), f'y shape incorrect: expected {(ny, nx)}, got {y.shape}'
    assert mask.shape == (ny, nx), f'x shape incorrect: expected {(ny, nx)}, got {mask.shape}'

    with open(binfile, 'r+b') as f:
        f.write(struct.pack(ny*nx*type_dic['float'], *x.flatten().astype(type_convert['float'])))
        f.write(struct.pack(ny*nx*type_dic['float'], *y.flatten().astype(type_convert['float'])))
        f.write(struct.pack(ny*nx*type_dic['int'], *mask.flatten().astype(type_convert['int'])))


##read x,y,mask from binfile
def read_header(binfile, info):
    nx = info['nx']
    ny = info['ny']
    with open(binfile, 'rb') as f:
        x = np.array(struct.unpack((ny*nx*type_dic['float']), f.read(ny*nx*type_size['float'])))
        y = np.array(struct.unpack((ny*nx*type_dic['float']), f.read(ny*nx*type_size['float'])))
        mask = np.array(struct.unpack((ny*nx*type_dic['int']), f.read(ny*nx*type_size['int'])))
        return x.reshape((ny, nx)), y.reshape((ny, nx)), mask.astype(bool).reshape((ny, nx))


##write a field with fld_id to the binfile
def write_field(binfile, info, mask, fld_id, fld):
    nx = info['nx']
    ny = info['ny']
    rec = info['fields'][fld_id]
    is_vector = rec['is_vector']
    fld_shape = (2, ny, nx) if is_vector else (ny, nx)
    assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

    fld_ = fld[:, ~mask].flatten() if is_vector else fld[~mask]
    with open(binfile, 'r+b') as f:
        f.seek(rec['pos'])
        f.write(struct.pack(fld_.size*type_dic[rec['dtype']], *fld_))


##read a field from binfile, given fld_id
def read_field(binfile, info, mask, fld_id):
    nx = info['nx']
    ny = info['ny']
    rec = info['fields'][fld_id]
    nv = 2 if rec['is_vector'] else 1
    fld_shape = (2, ny, nx) if rec['is_vector'] else (ny, nx)
    fld_size = np.sum((~mask).astype(int))

    with open(binfile, 'rb') as f:
        f.seek(rec['pos'])
        fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec['dtype']]),
                        f.read(nv*fld_size*type_size[rec['dtype']])))
        fld = np.full(fld_shape, np.nan)
        if rec['is_vector']:
            fld[:, ~mask] = fld_.reshape((2,-1))
        else:
            fld[~mask] = fld_
        return fld


##unmasked horizontal locale indices
def xy_inds(mask):
    ny, nx = mask.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    inds = jj * nx + ii
    return inds[~mask]


##uniq field records for one member
def uniq_fields(info):
    ufid = 0
    fields = {}
    for fid, rec in info['fields'].items():
        if rec['member'] == 0 and rec['name']!='z_coords':
            for i in range(2 if rec['is_vector'] else 1):
                fields[ufid] = rec
                ufid += 1
    return fields


##read the entire ensemble in a local state space [nens, ufields, local_inds]
##local_inds: list of inds for horizontal locales [y,x]
##  Note: when used in parallelization, local_inds shall be continous chunks of inds
##        to avoid conflict in read/write of binfile; otherwise in single call, it is
##        fine to read a discontiguous chunk of field with arbitrary list of local_inds.
##nfield indexes the uniq fields at each locale with key (name, time, k)
##return: dict (nfield, local_inds): state[nens], and coordinates name,time,z,y,x
def read_local_state(binfile, info, mask, local_inds):

    inds = xy_inds(mask)  ##horizontal locale indices for the entire field

    fld_size = np.sum((~mask).astype(int))  ##size of the field rec in binfile
    seek_inds = np.searchsorted(inds, local_inds)  ##seek pos in binfile for the local inds
    chk_size = seek_inds[-1] - seek_inds[0] + 1   ##size of chunk to read from field rec

    ##some dimensions for the local state
    nens = info['nens']
    nlocal = len(local_inds)
    ufields = uniq_fields(info)  ##dict fid: uniq field rec for one member
    nfield = len(ufields)

    local_state = {'state': np.full((nens, nfield, nlocal), np.nan),
                   'name': [r['name'] for r in ufields.values()],
                   'time': [t2h(r['time']) for r in ufields.values()],
                   'k': [r['k'] for r in ufields.values()],
                   'z': np.full((nens, nfield, nlocal), np.nan), }
                    ##TODO: there can be more than one z_coords for each uniq z_units

    with open(binfile, 'rb') as f:
        ##loop through each field rec in binfile
        for rec in info['fields'].values():

            ##read the chunk covering the local_inds
            for ic in range(2 if rec['is_vector'] else 1):  ##vector fields have 2 components

                f.seek(rec['pos'] + (ic*fld_size+seek_inds[0])*type_size[rec['dtype']])
                chunk = np.array(struct.unpack((chk_size*type_dic[rec['dtype']]), f.read(chk_size*type_size[rec['dtype']])))

                if rec['name'] == 'z_coords':
                    ##if this is a z_coords rec, assign its local_inds chunk to the corresponding z array
                    for fid in [i for i,r in ufields.items() if r['time']==rec['time'] and r['k']==rec['k']]:
                        local_state['z'][rec['member'], fid, :] = chunk[seek_inds-seek_inds[0]]

                else:
                    ##if this is a variable rec, assign the chunk to the state array
                    fid_list = [i for i,r in ufields.items() if r['name']==rec['name'] and r['time']==rec['time'] and r['k']==rec['k']]
                    local_state['state'][rec['member'], fid_list[ic], :] = chunk[seek_inds-seek_inds[0]]

    return local_state


##write the updated local_state [nens, nfield, local_inds] back to the binfile
def write_local_state(binfile, info, mask, local_inds, local_state):

    inds = xy_inds(mask) ##horizontal locale indices for the entire field

    fld_size = np.sum((~mask).astype(int))
    seek_inds = np.searchsorted(inds, local_inds)
    chk_size = seek_inds[-1] - seek_inds[0] + 1

    nens = info['nens']
    nlocal = len(local_inds)
    ufields = uniq_fields(info)  ##dict fid: uniq field rec for one member
    nfield = len(ufields)
    local_state_shape = local_state['state'].shape
    assert local_state_shape==(nens, nfield, nlocal), f'local_state shape incorrect: expected {(nens, nfield, nlocal)}, got {local_state_shape}'

    ##write the local local_state to binfiles
    with open(binfile, 'r+b') as f:
        for rec in info['fields'].values():

            for ic in range(2 if rec['is_vector'] else 1):

                if chk_size == nlocal:  ##chunk is contiguous, make an empty array
                    chunk = np.full(nlocal, np.nan)
                else:   ##chunk is discontiguous, first read the chunk from file
                    f.seek(rec['pos'] + (ic*fld_size+seek_inds[0])*type_size[rec['dtype']])
                    chunk = np.array(struct.unpack((chk_size*type_dic[rec['dtype']]), f.read(chk_size*type_size[rec['dtype']])))

                if rec['name'] == 'z_coords':
                    pass ##we don't need to output z_coords, since they are not updated by local analysis
                else:
                    ##update value in chunk given the new local_state
                    fid_list = [i for i,r in ufields.items() if r['name']==rec['name'] and r['time']==rec['time'] and r['k']==rec['k']]
                    chunk[seek_inds-seek_inds[0]] = local_state['state'][rec['member'], fid_list[ic], :]
                    ##write the chunk back to file
                    f.seek(rec['pos'] + (ic*fld_size+seek_inds[0])*type_size[rec['dtype']])
                    f.write(struct.pack((chk_size*type_dic[rec['dtype']]), *chunk))


