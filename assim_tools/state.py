import numpy as np
from netCDF4 import Dataset
import struct
from .common import type_convert, type_dic, type_size, t2h, h2t

##state variable definition
##available state variable names, and their properties
state_variables = {'atmos_surf_wind': {'is_vector':True, 'units':'m/s'},
                   'atmos_surf_temp': {'is_vector':False, 'units':'K'},
                   'atmos_surf_dew_temp': {'is_vector':False, 'units':'K'},
                   'atmos_surf_press': {'is_vector':False, 'units':'Pa'},
                   'atmos_precip': {'is_vector':False, 'units':'kg/m2/s'},
                   'atmos_snowfall': {'is_vector':False, 'units':'kg/m2/s'},
                   'atmos_down_shortwave': {'is_vector':False, 'units':'W/m2'},
                   'atmos_down_longwave': {'is_vector':False, 'units':'W/m2'},
                   'seaice_conc': {'is_vector':False, 'units':'%'},
                   'seaice_thick': {'is_vector':False, 'units':'m'},
                   'snow_thick': {'is_vector':False, 'units':'m'},
                   'seaice_drift': {'is_vector':True, 'units':'m/s'},
                   'seaice_damage': {'is_vector':False, 'units':'%'},
                   'ocean_surf_temp': {'is_vector':False, 'units':'K'},
                   'ocean_surf_velocity': {'is_vector':True, 'units':'m/s'},
                   'ocean_surf_height': {'is_vector':False, 'units':'m'},
                   'ocean_layer_thick': {'is_vector':False, 'units':'Pa'},
                   'ocean_temp': {'is_vector':False, 'units':'K'},
                   'ocean_salinity': {'is_vector':False, 'units':'%'},
                   'ocean_velocity': {'is_vector':True, 'units':'m/s'},
                   }

##parse and generate field_info dict
def field_info(state_def_file, ###state_def filename, see config module
               time,         ##datetime obj for the current analysis time
               t_offsets,    ##analysis window defined by (-a, b) hours
               zlevels,      ##vertical levels (indices) in analysis
               nens, ny, nx, ##ensemble size, and 2d field dimensions
               mask,         ##landmask (True for land)
               ):
    info = {'ny':ny, 'nx':nx, 'fields':{}}

    with open(state_def_file, 'r') as f:
        fld_id = 0   ##record number for this field
        pos = 0      ##f.seek position
        pos += ny * nx * type_size['int']  ##the landmask is the first record
        for lin in f.readlines():
            ss = lin.split()
            assert len(ss) == 6, 'state_def format error, should be "varname, src, dtype, dt, zi_min, zi_max"'
            ##variable name
            var_name = ss[0]
            ##which module handles i/o for this variable
            var_source = ss[1]
            ##data type int/float/double
            var_dtype = ss[2]
            ##time interval (hours) available from model
            var_dt = np.float32(ss[3])
            ##vertical index min,max
            ##Note: 0 means surface variables, negative indices for ocean layers
            ##      positive indices for atmos layers
            var_zi_min = np.int32(ss[4])
            var_zi_max = np.int32(ss[5])

            for m in range(nens):  ##loop over ensemble members
                for n in np.arange(int(t_offsets[0]/var_dt), int(t_offsets[1]/var_dt)+1):  ##time slices
                    for level in [z for z in zlevels if var_zi_min<=z<=var_zi_max]:  ##vertical levels
                        rec = {'var_name':var_name,
                               'source':var_source,
                               'dtype':var_dtype,
                               'is_vector':state_variables[var_name]['is_vector'],
                               'pos':pos,
                               'member':m,
                               'time':h2t(t2h(time) + n*var_dt),
                               'level':level}
                        info['fields'].update({fld_id:rec})  ##add this record to fields
                        ##f.seek position
                        nv = 2 if state_variables[var_name]['is_vector'] else 1
                        fsize = np.sum((~mask).astype(int))
                        pos += nv * fsize * type_size[var_dtype]
                        fld_id += 1
    return info

def read_field_info(filename):
    info = {}
    with open(filename.replace('.bin','.dat'), 'r') as f:
        lines = f.readlines()

        ss = lines[0].split()
        ny = int(ss[0])
        nx = int(ss[1])
        info.update({'ny':ny, 'nx':nx, 'fields':{}})

        ##following lines of variable records
        fld_id = 0
        for lin in lines[1:]:
            ss = lin.split()
            var_name = ss[0]
            source = ss[1]
            dtype = ss[2]
            is_vector = bool(int(ss[3]))
            pos = int(ss[4])
            member = int(ss[5])
            time = h2t(np.float32(ss[6]))
            level = np.int32(ss[7])
            field = {'var_name':var_name,
                     'source':source,
                     'dtype':dtype,
                     'is_vector':is_vector,
                     'pos':pos,
                     'member':member,
                     'time':time,
                     'level':level}
            info['fields'].update({fld_id:field})
            fld_id += 1
    return info

def write_field_info(filename, info):
    ##line 1: ny, nx
    ##line 2:end: list of variables (one per line):
    ##     var_name, source_module, data_type, seek_position, member, time, level
    with open(filename.replace('.bin','.dat'), 'wt') as f:
        f.write('%i %i\n' %(info['ny'], info['nx']))
        for i, rec in info['fields'].items():
            f.write('%s %s %s %i %i %i %f %i\n' %(rec['var_name'], rec['source'], rec['dtype'], int(rec['is_vector']), rec['pos'], rec['member'], t2h(rec['time']), rec['level']))

##mask off land area, only store data where mask=False
def read_mask(filename, info):
    nx = info['nx']
    ny = info['ny']
    with open(filename, 'rb') as f:
        mask = np.array(struct.unpack((ny*nx*type_dic['int']), f.read(ny*nx*type_size['int'])))
        return mask.astype(bool).reshape((ny, nx))

def write_mask(filename, info, mask):
    ny, nx = mask.shape
    assert ny == info['ny'] and nx == info['nx'], 'mask shape incorrect'
    with open(filename, 'wb'):  ##initialize the file if it doesn't exist
        pass
    with open(filename, 'r+b') as f:
        f.write(struct.pack(ny*nx*type_dic['int'], *mask.flatten().astype(int)))

##mask: area in the reference grid that is land or other area that doesn't require analysis,
##      2D fields will have NaN in those area, bin files will only store the ~mask region,
##      the analysis routine will also skip if mask
def prepare_mask(c):
    if c.MASK_FROM == 'nextsim':
        from models import nextsim
        grid = nextsim.get_grid_from_msh(c.MESH_FILE)
        grid.dst_grid = c.ref_grid
        mask  = np.isnan(grid.convert(grid.x))
    ##other options...
    ##save mask to file
    np.save(c.WORK_DIR+'/mask.npy', mask)

##read/write a 2D field from/to binfile, given fid
def read_field(filename, info, mask, fid):
    nx = info['nx']
    ny = info['ny']
    rec = info['fields'][fid]
    is_vector = rec['is_vector']
    nv = 2 if is_vector else 1
    fld_shape = (2, ny, nx) if is_vector else (ny, nx)
    fsize = np.sum((~mask).astype(int))

    with open(filename, 'rb') as f:
        f.seek(rec['pos'])
        fld_ = np.array(struct.unpack((nv*fsize*type_dic[rec['dtype']]),
                        f.read(nv*fsize*type_size[rec['dtype']])))
        fld = np.full(fld_shape, np.nan)
        if is_vector:
            fld[:, ~mask] = fld_.reshape((2,-1))
        else:
            fld[~mask] = fld_
        return fld

def write_field(filename, info, mask, fid, fld):
    ny = info['ny']
    nx = info['nx']
    rec = info['fields'][fid]
    is_vector = rec['is_vector']
    fld_shape = (2, ny, nx) if is_vector else (ny, nx)
    assert fld.shape == fld_shape, 'fld shape incorrect'

    fld_ = fld[:, ~mask].flatten() if is_vector else fld[~mask]
    with open(filename, 'r+b') as f:
        f.seek(rec['pos'])
        f.write(struct.pack(fld_.size*type_dic[rec['dtype']], *fld_))

##parse state_def, generate field_info, then read model output and write 2D fields into bin files:
##   state[nens, nfield, ny, nx], nfield dimension contains nv,nt,nz flattened
##   nv is number of variables, nt is time slices, nz is vertical layers,
##   of course nt,nz vary for each variables, so we stack them in nfield dimension
def prepare_state(c, comm, time):
    ##c: config module
    ##comm: mpi4py communicator
    ##time: analysis time (datetime obj)
    import importlib
    import sys
    from .parallel import distribute_tasks

    ny, nx = c.ref_grid.x.shape
    mask = np.load(c.WORK_DIR+'/mask.npy')
    binfile = c.WORK_DIR+'/prior.bin' ##TODO: timestr

    if comm.Get_rank() == 0:
        ##generate field info from state_def
        info = field_info(c.STATE_DEF_FILE,
                        time,
                        (c.OBS_WINDOW_MIN, c.OBS_WINDOW_MAX),
                        np.arange(c.ZI_MIN, c.ZI_MAX+1),
                        c.NUM_ENS,
                        *c.ref_grid.x.shape, mask)
        write_field_info(binfile, info)
        write_mask(binfile, info, mask)
    else:
        info = None
    info = comm.bcast(info, root=0)

    grid_bank = {}
    for i in distribute_tasks(comm, np.arange(len(info['fields']))):
        rec = info['fields'][i]
        v = rec['var_name']
        t = rec['time']
        m = rec['member']
        z = rec['level']
        print('processing', v, t, m, z)
        sys.stdout.flush()

        ##TODO: coarse-graining in z and t dimensions
        #if available model output has more levels and time slices, average them to the target level/time box.

        ##directory storing model output
        path = c.WORK_DIR + '/models/' + rec['source'] ##+ '/' + time.strftime('%Y%m%dT%H%M')

        ##load the module for handling source model
        src = importlib.import_module('models.'+rec['source'])

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

def get_dims(info):
    nens = len(list(set(rec['member'] for i,rec in info['fields'].items())))
    nfield = 0
    for rec in [rec for i,rec in info['fields'].items() if rec['member']==0]:
        for i in range(2 if rec['is_vector'] else 1):
            nfield += 1
    return nens, nfield

##read/write the entire ensemble in a local state space [nens, nfield, inds]
##local_inds: list of inds for local analysis
def read_local_ens(filename, info, mask, local_inds):
    nens, nfield = get_dims(info)
    ny = info['ny']
    nx = info['nx']
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    inds = jj * nx + ii

    ##read the local state ensemble, state_ens[member, field, local_inds]
    state_ens = np.full((nens, nfield, len(local_inds)), np.nan)
    with open(filename, 'rb') as f:
        for m in range(nens):
            n = 0
            for rec in [rec for i,rec in info['fields'].items() if rec['member']==m]:
                fsize = np.sum((~mask).astype(int))
                seek_inds = np.searchsorted(inds[~mask], local_inds)
                csize = seek_inds[-1] - seek_inds[0] + 1
                for i in range(2 if rec['is_vector'] else 1):
                    ##read the chunck from binfile covering local_inds
                    f.seek(rec['pos'] + (i*fsize+seek_inds[0])*type_size[rec['dtype']])
                    chunck = np.array(struct.unpack((csize*type_dic[rec['dtype']]), f.read(csize*type_size[rec['dtype']])))
                    ##collect the local_inds from chunck and assign to state_ens
                    state_ens[m, n, :] = chunck[seek_inds-seek_inds[0]]
                    n += 1
    return state_ens

def write_local_ens(filename, info, mask, local_inds, state_ens):
    nens, nfield = get_dims(info)
    assert state_ens.shape == (nens, nfield, len(local_inds)), 'state_ens shape incorrect: expected ({},{},{}), got ({},{},{})'.format(nens, nfield, len(local_inds), *state_ens.shape)
    ny = info['ny']
    nx = info['nx']
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    inds = jj * nx + ii

    ##write the local state_ens to binfiles
    with open(filename, 'r+b') as f:
        for m in range(nens):
            n = 0
            for rec in [rec for i,rec in info['fields'].items() if rec['member']==m]:
                fsize = np.sum((~mask).astype(int))
                seek_inds = np.searchsorted(inds[~mask], local_inds)
                csize = seek_inds[-1] - seek_inds[0] + 1
                for i in range(2 if rec['is_vector'] else 1):
                    ##first read the chunck
                    f.seek(rec['pos'] + (i*fsize+seek_inds[0])*type_size[rec['dtype']])
                    chunck = np.array(struct.unpack((csize*type_dic[rec['dtype']]), f.read(csize*type_size[rec['dtype']])))
                    ##update value in chunck given the new state_ens
                    chunck[seek_inds-seek_inds[0]] = state_ens[m, n, :]
                    ##write the chunck back to file
                    f.seek(rec['pos'] + (i*fsize+seek_inds[0])*type_size[rec['dtype']])
                    f.write(struct.pack((csize*type_dic[rec['dtype']]), *chunck))
                    n += 1

