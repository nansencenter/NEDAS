import os
import numpy as np
from netCDF4 import Dataset
import struct
from datetime import datetime, timedelta
from assim_tools import variables

##netcdf file io
def nc_write_var(filename, dim, varname, dat, attr=None):
    ###write gridded data to netcdf file
    ###filename: path/name of the output file
    ###dim: dict{'dimension name': number of grid points in dimension (int)}
    ###varname: name for the output variable
    ###dat: data for output, number of dimensions must match dim
    f = Dataset(filename, 'a', format="NETCDF4_CLASSIC")
    ndim = len(dim)
    assert(len(dat.shape)==ndim)
    for key in dim:
        if key in f.dimensions:
            if dim[key]!=0:
                assert(f.dimensions[key].size==dim[key])
        else:
            f.createDimension(key, size=dim[key])
    if varname in f.variables:
        assert(f.variables[varname].shape==dat.shape)
    else:
        f.createVariable(varname, float, dim.keys())
    f[varname][...] = dat
    if attr != None:
        for akey in attr:
            f[varname].setncattr(akey, attr[akey])
    f.close ()

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
is_masked = {'models.nextsim':True,
             'models.topaz':True,
             'models.wrf':False, }

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
        nens = int(ss[0])
        ny = int(ss[1])
        nx = int(ss[2])

        var_names = lines[1].split()
        var_sources = lines[2].split()

        ss = lines[3].split()
        time = h2t(np.float32(ss[0]))
        t_offsets = (np.float32(ss[1]), np.float32(ss[2]))

        ss = lines[4].split()
        zlevels = [np.int32(z) for z in ss]

        info.update({'nens':nens,
                     'ny':ny,
                     'nx':nx,
                     'var_names':var_names,
                     'var_sources':var_sources,
                     'time':time,
                     'time_window':(t_offsets[0], t_offsets[1]),
                     'levels':zlevels,
                     'fields':{}})

        ##following lines of variable records
        fld_id = 0
        for lin in lines[5:]:
            ss = lin.split()
            var_name = ss[0]
            source = ss[1]
            dtype = ss[2]
            pos = int(ss[3])
            member = int(ss[4])
            time = h2t(np.float32(ss[5]))
            level = np.int32(ss[6])
            field = {'var_name':var_name,
                     'source':source,
                     'dtype':dtype,
                     'pos':pos,
                     'member':member,
                     'time':time,
                     'level':level}
            info['fields'].update({fld_id:field})
            fld_id += 1

    return info

def write_field_info(filename, info):
    ##line 1: nens, ny, nx
    ##line 2: list of variable names
    ##line 3: list of variable sources
    ##line 4: time (hours since 1900-1-1), offset_left, offset_right (hours)
    ##line 5: list of vertical level indices
    ##line 6:end: list of variables (one per line):
    ##     var_name, source_module, data_type, seek_position, member, time, level
    with open(filename.replace('.bin','.dat'), 'wt') as f:
        f.write('%i %i %i\n' %(info['nens'], info['ny'], info['nx']))
        f.write(' '.join(info['var_names'])+'\n')
        f.write(' '.join(info['var_sources'])+'\n')
        f.write('%f %f %f\n' %(t2h(info['time']), *info['time_window']))
        for z in info['levels']:
            f.write('%i ' %(z))
        f.write('\n')

        for i, rec in info['fields'].items():
            f.write('%s %s %s %i %i %f %i\n' %(rec['var_name'], rec['source'], rec['dtype'], rec['pos'], rec['member'], t2h(rec['time']), rec['level']))

##parse and generate field_info dict
def field_info(state_def_file, ###state_def filename, see config module
               time,         ##datetime obj for the current analysis time
               t_offsets,    ##analysis window defined by (-a, b) hours
               zlevels,      ##vertical levels (indices) in analysis
               nens, ny, nx, ##ensemble size, and 2d field dimensions
               mask,         ##landmask (True for land)
               ):
    info = {}
    info.update({'nens':nens,
                 'ny':ny,
                 'nx':nx,
                 'var_names':[],
                 'var_sources':[],
                 'time':time,
                 'time_window':(t_offsets[0], t_offsets[1]),
                 'levels':[z for z in zlevels],
                 'fields':{}})

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

            info['var_names'].append(var_name)
            info['var_sources'].append(var_source)

            if variables[var_name]['is_vector']:
                vlist = (var_name+'_x', var_name+'_y')
            else:
                vlist = (var_name,)
            for v in vlist:  ##loop over variable list
                for m in range(nens):  ##loop over ensemble members
                    for n in np.arange(int(t_offsets[0]/var_dt), int(t_offsets[1]/var_dt)+1):  ##time slices
                        for level in [z for z in zlevels if var_zmin<=z<=var_zmax]:  ##vertical levels
                            field = {'var_name':v,
                                     'source':var_source,
                                     'dtype':var_dtype,
                                     'pos':pos,
                                     'member':m,
                                     'time':time + n*var_dt*timedelta(hours=1),
                                     'level':level}
                            info['fields'].update({fld_id:field})  ##add this record to fields
                            ##f.seek position
                            if is_masked[var_source]:  ##if there is landmask, only store the ocean area
                                pos += np.sum((~mask).astype(np.int32)) * type_size[var_dtype]
                            else:
                                pos += ny * nx * type_size[var_dtype]
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
        f.write(struct.pack(ny*nx*type_dic['int'], *mask.flatten().astype(np.int32)))

def read_field(filename, info, mask, fid):
    nx = info['nx']
    ny = info['ny']
    rec = info['fields'][fid]
    if is_masked[rec['source']]:
        fsize = np.sum((~mask).astype(np.int32))
    else:
        fsize = ny*nx

    with open(filename, 'rb') as f:
        f.seek(rec['pos'])
        fld_ = np.array(struct.unpack((fsize*type_dic[rec['dtype']]), f.read(fsize*type_size[rec['dtype']])))
        if is_masked[rec['source']]:
            fld = np.full((ny, nx), np.nan)
            fld[~mask] = fld_
        else:
            fld = fld_.reshape((ny, nx))
        return fld

def write_field(filename, info, mask, fid, fld):
    ny, nx = fld.shape
    assert ny == info['ny'] and nx == info['nx'], 'fld shape (ny,nx) incorrect'
    rec = info['fields'][fid]
    if is_masked[rec['source']]:
        fsize = np.sum((~mask).astype(np.int32))
        fld_ = fld[~mask]
    else:
        fsize = ny*nx
        fld_ = fld.flatten()

    with open(filename, 'r+b') as f:
        f.seek(rec['pos'])
        f.write(struct.pack(fsize*type_dic[rec['dtype']], *fld_))

