import numpy as np
import struct
import importlib
import sys
from datetime import datetime, timedelta
from .common import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from .parallel import distribute_tasks

##top-level routine to prepare the state variables, getting them from model restart files,
##convert to the analysis grid, perform coarse-graining, and save to the binfile
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallelization
def process_state(c, comm):
    ##name of the binfile for the current c.time and c.scale
    ##make a separate dir for each scale if more than 1 scale
    s_dir = f'/s{c.scale+1}' if c.nscale>1 else ''
    binfile = c.work_dir+'/analysis/'+c.time+s_dir+'/state.bin'

    ##parse config and generate field_info, this is done by the first processor and broadcast
    if comm.Get_rank() == 0:
        info = field_info(c)
        with open(binfile, 'wb'):  ##initialize the binfile in case it doesn't exist
            pass
        write_field_info(binfile, info)
        write_header(binfile, info, c.grid.x, c.grid.y, c.mask)
    else:
        info = None
    info = comm.bcast(info, root=0)

    grid_bank = {}

    ##now start processing the fields, each processor gets its own workload as a subset of nfield
    for i in distribute_tasks(comm, np.arange(len(info['fields']))):
        rec = info['fields'][i]
        # print('processing', rec)
        # sys.stdout.flush()

        ##directory storing model output
        path = c.work_dir+'/forecast/'+c.time+'/'+rec['source']

        ##load the module for handling source model
        src = importlib.import_module('models.'+rec['source'])

        #if available model output has more levels and time slices, average them to the target level/time box.
        vtimes = np.arange(t2h(s2t(c.time_start)), t2h(s2t(c.time_end)), src.variables[rec['name']]['dt'])
        #[h2t(h) for h in vtimes if h>=t2h(t) and h<t2h(t)+c.dt],

        ##only need to compute the uniq grids, stored them in bank for later use
        grid_key = (rec['source'],)
        for key in src.uniq_grid_key:
            grid_key += (rec[key],)
        if grid_key in grid_bank:
            grid = grid_bank[grid_key]
        else:
            grid = src.read_grid(path, **rec)
            grid.dst_grid = c.ref_grid
            grid_bank[grid_key] = grid

        var = src.read_var(path, grid, **rec)
        fld = grid.convert(var, is_vector=rec['is_vector'])

        write_field(binfile, info, mask, i, fld)

    ##clean up
    del grid_bank


##generate info for the nens*nfield 2D fields in the state
##The entire ensemble state has dimensions: member, variable, time, z, y, x
##to organize the tasks of i/o and filter update, we consider 3 indices: member, field, locale
##member indexes the rank in the ensemble
##locale indexes the horizontal position in a given 2D field defined on analysis grid
##field indexes the remaining [variable, time, z] stacked into one dimension, nt and nz vary for
##   different variables, stacking them in one dimension helps better distribute i/o tasks
def field_info(c):
    info = {'nx':c.nx, 'ny':c.ny,
            'nens': c.nens,
            'nfield': 0, 'fields':{},
            'nuz':0, 'z_coords':{}}

    fld_id = 0   ##field id
    nfield = 0   ##uniq fields count for one member
    pos = 0   ##f.seek position
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
                              'dt': np.minimum(c.t_scale, c.obs_dt),
                              'k': k,
                              'pos': pos, }
                    info['fields'][fld_id] = kwargs

                    ##update f.seek position
                    nv = 2 if src.variables[name]['is_vector'] else 1
                    fld_size = np.sum((~c.mask).astype(int))
                    pos += nv * fld_size * type_size[src.variables[name]['dtype']]

                    fld_id += 1
                    if member==0:
                        nfield += 1
    info['nfield'] = nfield

    ##loop through obs variables and get set of z_types
    z_types = set()
    for obs_name in c.obs_def:
        obs_src = importlib.import_module('dataset.'+c.obs_def[obs_name]['source'])
        z_types.add(obs_src.variables[obs_name]['z_type'])

    z_id = 0 ##uniq z level id
    for z_type in z_types:
        ##loop over all field records, find uniq key for this z_coords, add the record to info
        for fld_id, rec in info['fields'].items():
            src = importlib.import_module('models.'+rec['source'])

            member = rec['member'] if 'member' in src.uniq_z_key else None
            time = rec['time'] if 'time' in src.uniq_z_key else None
            z_key = (z_type, rec['source'], member, time, rec['k'])

            if z_key not in info['z_coords']:
                info['z_coords'][z_key] = {'pos':pos}
                pos += np.sum((~c.mask).astype(int)) * type_size['float']
                z_id += 1
    info['nuz'] = z_id

    return info


##write field_info to a .dat file accompanying the bin file
def write_field_info(binfile, info):
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        ##first line: some dimensions
        f.write('{} {} {} {} {}\n'.format(info['nx'], info['ny'], info['nens'], info['nfield'], info['nuz']))

        ##followed by nfield lines: each for a field record
        for i, rec in info['fields'].items():
            f.write('{} {} {} {} {} {} {} {} {} {} {}\n'.format(rec['name'], rec['source'], rec['dtype'], int(rec['is_vector']), rec['units'], rec['err_type'], rec['member'], t2h(rec['time']), rec['dt'], rec['k'], rec['pos']))

        ##followed by nuz lines: each for a uniq z_coords record
        for key, rec in info['z_coords'].items():
            z_type, source, member, time, k = key
            time_h = None if time is None else t2h(time)
            f.write('{} {} {} {} {} {}\n'.format(z_type, source, member, time_h, k, rec['pos']))


##read field_info from .dat file
def read_field_info(binfile):
    with open(binfile.replace('.bin','.dat'), 'r') as f:
        lines = f.readlines()

        ss = lines[0].split()
        info = {'nx':int(ss[0]), 'ny':int(ss[1]),
                'nens':int(ss[2]),
                'nfield':int(ss[3]), 'fields':{},
                'nuz':int(ss[4]), 'z_coords':{}}

        ##records for uniq fields
        ln = 1
        for fld_id in range(info['nfield']*info['nens']):
            ss = lines[ln].split()
            rec = {'name': ss[0],
                   'source': ss[1],
                   'dtype': ss[2],
                   'is_vector': bool(int(ss[3])),
                   'units': ss[4],
                   'err_type': ss[5],
                   'member': int(ss[6]),
                   'time': h2t(np.float32(ss[7])),
                   'dt': np.float32(ss[8]),
                   'k': int(ss[9]),
                   'pos': int(ss[10]), }
            info['fields'][fld_id] = rec
            ln += 1

        ##records for uniq z coordinates
        for z_id in range(info['nuz']):
            ss = lines[ln].split()
            z_key = (ss[0],   ##z_type
                     ss[1],   ##source
                     int(ss[2]) if ss[2]!='None' else None,  ##member
                     h2t(np.float32(ss[3])) if ss[3]!='None' else None, ##time
                     int(ss[4]),  ##k
                     )
            info['z_coords'][z_key] = {'pos': int(ss[5])}
            ln += 1

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
    is_vector = rec['is_vector']
    nv = 2 if is_vector else 1
    fld_shape = (2, ny, nx) if is_vector else (ny, nx)
    fld_size = np.sum((~mask).astype(int))

    with open(binfile, 'rb') as f:
        f.seek(rec['pos'])
        fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec['dtype']]),
                        f.read(nv*fld_size*type_size[rec['dtype']])))
        fld = np.full(fld_shape, np.nan)
        if is_vector:
            fld[:, ~mask] = fld_.reshape((2,-1))
        else:
            fld[~mask] = fld_
        return fld


##write a z_coords with z_key to the binfile
def write_z_coords(binfile, info, mask, z_coords, z_key):
    nx = info['nx']
    ny = info['ny']
    assert z_coords.shape == (ny, nx), f'z_coords shape incorrect: expected {(ny, nx)}, got {z_coords.shape}'
    rec = info['z_coords'][z_key]
    z_ = z_coords[~mask]
    with open(binfile, 'r+b') as f:
        f.seek(rec['pos'])
        f.write(struct.pack(z_.size*type_dic['float'], *z_))


##read a z_coords from binfile given z_key
def read_z_coords(binfile, info, mask, z_key):
    nx = info['nx']
    ny = info['ny']
    rec = info['z_coords'][z_key]
    z_coords = np.full((ny, nx), np.nan)
    z_size = np.sum((~mask).astype(int))
    with open(binfile, 'rb') as f:
        f.seek(rec['pos'])
        z_coords[~mask] = np.array(struct.unpack((z_size*type_dic['float']), f.read(z_size*type_size['float'])))
        return z_coords


##read the entire ensemble in a local state space [nens, nfield, local_inds]
##local_inds: list of inds for locales [y,x] for a processor to perform local analysis
def read_local_state(binfile, info, mask, local_inds):
    ny = info['ny']
    nx = info['nx']
    nfield = info['nfield']
    nens = info['nens']
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    inds = jj * nx + ii

    ##read the local state ensemble, local_state[member, field, local_inds]
    local_state = np.full((nens, nfield, len(local_inds)), np.nan)
    with open(binfile, 'rb') as f:
        for m in range(nens):
            n = 0
            for rec in [rec for i,rec in info['fields'].items() if rec['member']==m]:
                fld_size = np.sum((~mask).astype(int))
                seek_inds = np.searchsorted(inds[~mask], local_inds)
                chk_size = seek_inds[-1] - seek_inds[0] + 1
                for i in range(2 if rec['is_vector'] else 1):
                    ##read the chunck from binfile covering local_inds
                    f.seek(rec['pos'] + (i*fld_size+seek_inds[0])*type_size[rec['dtype']])
                    chunck = np.array(struct.unpack((chk_size*type_dic[rec['dtype']]), f.read(chk_size*type_size[rec['dtype']])))
                    ##collect the local_inds from chunck and assign to local_state
                    local_state[m, n, :] = chunck[seek_inds-seek_inds[0]]
                    n += 1
    return local_state


##write the updated local_state [nens, nfield, local_inds] back to the binfile
def write_local_state(binfile, info, mask, local_inds, local_state):
    ny = info['ny']
    nx = info['nx']
    nfield = info['nfield']
    nens = info['nens']
    assert local_state.shape == (nens, nfield, len(local_inds)), f'local_state shape incorrect: expected {(nens, nfield, len(local_inds))}, got {local_state.shape}'

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    inds = jj * nx + ii

    ##write the local local_state to binfiles
    with open(binfile, 'r+b') as f:
        for m in range(nens):
            n = 0
            for rec in [rec for i,rec in info['fields'].items() if rec['member']==m]:
                fld_size = np.sum((~mask).astype(int))
                seek_inds = np.searchsorted(inds[~mask], local_inds)
                chk_size = seek_inds[-1] - seek_inds[0] + 1
                for i in range(2 if rec['is_vector'] else 1):
                    ##first read the chunck
                    f.seek(rec['pos'] + (i*fld_size+seek_inds[0])*type_size[rec['dtype']])
                    chunck = np.array(struct.unpack((chk_size*type_dic[rec['dtype']]), f.read(chk_size*type_size[rec['dtype']])))
                    ##update value in chunck given the new local_state
                    chunck[seek_inds-seek_inds[0]] = local_state[m, n, :]
                    ##write the chunck back to file
                    f.seek(rec['pos'] + (i*fld_size+seek_inds[0])*type_size[rec['dtype']])
                    f.write(struct.pack((chk_size*type_dic[rec['dtype']]), *chunck))
                    n += 1


