import os
import numpy as np
from netCDF4 import Dataset
import struct
from datetime import datetime, timedelta
from .state import variables

##netcdf file io
def nc_write_var(filename, dim, varname, dat, recno=None, attr=None):
    ###write gridded data to netcdf file
    ###filename: path/name of the output file
    ###dim: dict{'dimension name': number of grid points in dimension (int); None for unlimited}
    ###varname: name for the output variable
    ###dat: data for output, number of dimensions must match dim (excluding unlimited dims)
    ###recno: dict{'dimension name': record_number, ...}, each unlimited dim should have a recno entry
    ###attr: attribute dict{'name of attr':'value of attr'}
    f = Dataset(filename, 'a', format='NETCDF4')
    ndim = len(dim)
    s = ()  ##slice for each dimension
    d = 0
    for i, name in enumerate(dim):
        if dim[name] is None:
            assert(name in recno), "unlimited dimension "+name+" doesn't have a recno"
            s += (recno[name],)
        else:
            s += (slice(None),)
            assert(dat.shape[d] == dim[name]), "dat size for dimension "+name+" mismatch with dim["+name+"]={}".format(dim[name])
            d += 1
        if name in f.dimensions:
            if dim[name] is not None:
                assert(f.dimensions[name].size==dim[name]), "dimension "+name+" size ({}) mismatch with file ({})".format(dim[name], f.dimensions[name].size)
            else:
                assert(f.dimensions[name].isunlimited())
        else:
            f.createDimension(name, size=dim[name])
    if varname not in f.variables:
        f.createVariable(varname, np.float32, dim.keys())

    f[varname][s] = dat  ##write dat to file
    if attr is not None:
        for akey in attr:
            f[varname].setncattr(akey, attr[akey])
    f.close()

def nc_read_var(filename, varname):
    ###read gridded data from netcdf file
    f = Dataset(filename, 'r')
    assert(varname in f.variables)
    dat = f[varname][...].data
    f.close()
    return dat

##binary file io for state variable fields
type_convert = {'double':np.float64, 'float':np.float32, 'int':np.int32}
type_dic = {'double':'d', '8':'d', 'single':'f', 'float':'f', '4':'f', 'int':'i'}
type_size = {'double':8, 'float':4, 'int':4}

##if a model has landmask, we apply the mask to reduce state dimension
is_masked = {'nextsim':True,
             'topaz':True,
             'wrf':False, }

def t2h(t):
    ##convert datetime obj to hours since 1900-1-1 00:00
    return (t - datetime(1900,1,1))/timedelta(hours=1)

def h2t(h):
    ##convert hours since 1900-1-1 00:00 to datetime obj
    return datetime(1900,1,1) + timedelta(hours=1) * h

##parse .dat file
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
            level = int(ss[7])
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

##parse and generate field_info dict
def field_info(state_def_file, ###state_def filename, see config module
               time,         ##datetime obj for the current analysis time
               t_offsets,    ##analysis window defined by (-a, b) hours
               zlevels,      ##vertical levels (indices) in analysis
               nens, ny, nx, ##ensemble size, and 2d field dimensions
               mask,         ##landmask (True for land)
               ):
    info = {}
    info.update({'ny':ny, 'nx':nx, 'fields':{}})

    with open(state_def_file, 'r') as f:
        fields = {}
        fld_id = 0   ##record number for this field
        pos = 0      ##f.seek position
        pos += ny * nx * type_size['int']  ##the landmask is the first record
        for lin in f.readlines():
            ss = lin.split()
            assert len(ss) == 6, 'state_def format error, should be "varname, src, dtype, dt, zmin, zmax"'
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
            var_zmin = np.int32(ss[4])
            var_zmax = np.int32(ss[5])

            for m in range(nens):  ##loop over ensemble members
                for n in np.arange(int(t_offsets[0]/var_dt), int(t_offsets[1]/var_dt)+1):  ##time slices
                    for level in [z for z in zlevels if var_zmin<=z<=var_zmax]:  ##vertical levels
                        field = {'var_name':var_name,
                                 'source':var_source,
                                 'dtype':var_dtype,
                                 'is_vector':variables[var_name]['is_vector'],
                                 'pos':pos,
                                 'member':m,
                                 'time':time + n*var_dt*timedelta(hours=1),
                                 'level':level}
                        info['fields'].update({fld_id:field})  ##add this record to fields
                        ##f.seek position
                        nv = 2 if variables[var_name]['is_vector'] else 1
                        fsize = np.sum((~mask).astype(np.int32)) if is_masked[var_source] else ny*nx
                        pos += nv * fsize * type_size[var_dtype]
                        fld_id += 1
    return info

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
    if not os.path.exists(filename):  ##initialize the file if it doesn't exist
        with open(filename, 'wb'):
            pass
    with open(filename, 'r+b') as f:
        f.write(struct.pack(ny*nx*type_dic['int'], *mask.flatten().astype(int)))

##read/write a 2D field from/to binfile, given fid
def read_field(filename, info, mask, fid):
    nx = info['nx']
    ny = info['ny']
    rec = info['fields'][fid]
    is_vector = rec['is_vector']
    nv = 2 if is_vector else 1
    fld_shape = (2, ny, nx) if is_vector else (ny, nx)
    fsize = np.sum((~mask).astype(int)) if is_masked[rec['source']] else ny*nx

    with open(filename, 'rb') as f:
        f.seek(rec['pos'])
        fld_ = np.array(struct.unpack((nv*fsize*type_dic[rec['dtype']]),
                        f.read(nv*fsize*type_size[rec['dtype']])))
        if is_masked[rec['source']]:
            fld = np.full(fld_shape, np.nan)
            if is_vector:
                fld[:, ~mask] = fld_.reshape((2,-1))
            else:
                fld[~mask] = fld_
        else:
            fld = fld_.reshape(fld_shape)
        return fld

def write_field(filename, info, mask, fid, fld):
    ny = info['ny']
    nx = info['nx']
    rec = info['fields'][fid]
    is_vector = rec['is_vector']
    fld_shape = (2, ny, nx) if is_vector else (ny, nx)
    assert fld.shape == fld_shape, 'fld shape incorrect'

    if is_masked[rec['source']]:
        fld_ = fld[:, ~mask].flatten() if is_vector else fld[~mask]
    else:
        fld_ = fld.flatten()

    with open(filename, 'r+b') as f:
        f.seek(rec['pos'])
        f.write(struct.pack(fld_.size*type_dic[rec['dtype']], *fld_))

##read/write the entire ensemble state[nens, nfields] for a local region
##corresponding to spatial index inds
def get_local_dims(info):
    ##ensemble size
    nens = len(list(set(rec['member'] for i,rec in info['fields'].items())))
    ##number of unique fields
    nfield = 0
    var_names = []
    times = []
    levels = []
    for rec in [rec for i,rec in info['fields'].items() if rec['member']==0]:
        if rec['is_vector']:
            comp = ('_x', '_y')
            for i in range(2):
                nfield += 1
                var_names.append(rec['var_name']+comp[i])
                times.append(rec['time'])
                levels.append(rec['level'])
        else:
            nfield += 1
            var_names.append(rec['var_name'])
            times.append(rec['time'])
            levels.append(rec['level'])
    return nens, nfield, var_names, times, levels

def read_local_ens(filename, info, mask, inds):
    nens, nfield, _, _, _ = get_local_dims(info)

    ny = info['ny']
    nx = info['nx']
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    inds_full = jj * nx + ii

    ##read the local states from corresponding fields
    state_ens = np.full((nens, nfield), np.nan)
    with open(filename, 'rb') as f:
        for m in range(nens):
            n = 0
            for rec in [rec for i,rec in info['fields'].items() if rec['member']==m]:
                if is_masked[rec['source']]:
                    fsize = np.sum((~mask).astype(int))
                    seek_inds = np.searchsorted(inds_full[~mask], inds)
                else:
                    fsize = ny*nx
                    seek_inds = inds
                if rec['is_vector']:
                    for i in range(2):
                        f.seek(rec['pos'] + (i*fsize+seek_inds)*type_size[rec['dtype']])
                        state_ens[m, n] = np.array(struct.unpack((1*type_dic[rec['dtype']]), f.read(1*type_size[rec['dtype']])))
                        n += 1
                else:
                    f.seek(rec['pos'] + seek_inds*type_size[rec['dtype']])
                    state_ens[m, n] = np.array(struct.unpack((1*type_dic[rec['dtype']]), f.read(1*type_size[rec['dtype']])))
                    n+= 1
    return state_ens, var_names, times, levels

def write_local_ens(filename, info, mask, idx, state_ens):
    nens, nfield, _, _, _ = get_local_dims(info)
    assert dat_ens.shape == (nens, nfield), 'dat_ens shape incorrect: expected ({},{}), got ({},{})'.format(nens, nfield, *dat_ens.shape)
    ny = info['ny']
    nx = info['nx']
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    inds_full = jj * nx + ii

    ##write the local state_ens to binfiles
    with open(filename, 'r+b') as f:
        for m in range(nens):
            n = 0
            for rec in [rec for i,rec in info['fields'].items() if rec['member']==m]:
                if is_masked[rec['source']]:
                    fsize = np.sum((~mask).astype(int))
                    seek_inds = np.searchsorted(inds_full[~mask], inds)
                else:
                    fsize = ny*nx
                    seek_inds = inds
                if rec['is_vector']:
                    for i in range(2):
                        f.seek(rec['pos'] + (i*fsize+seek_inds)*type_size[rec['dtype']])
                        f.write(struct.pack((1*type_dic[rec['dtype']]), *state_ens[m, n]))
                        n += 1
                else:
                    f.seek(rec['pos'] + seek_inds*type_size[rec['dtype']])
                    f.write(struct.pack((1*type_dic[rec['dtype']]), *state_ens[m, n]))
                    n += 1

