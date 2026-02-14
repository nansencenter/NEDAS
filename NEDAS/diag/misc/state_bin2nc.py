"""Convert binary state file contents to netcdf files in analysis directory"""
import os
import struct
import numpy as np
from NEDAS.utils.netcdf_lib import nc_write_var
from NEDAS.utils.conversion import t2h, type_dic, type_size

def read_field(binfile: str, info: dict, mask: np.ndarray, mem_id: int, rec_id: int) -> np.ndarray:
    rec = info['fields'][rec_id]

    nv = 2 if rec['is_vector'] else 1
    fld_shape = (2,)+info['shape'] if rec['is_vector'] else info['shape']
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

def run(c):
    adir = c.analysis_dir(c.time)
    filename = os.path.join(adir, 'prior_state.bin')

    ##convert bin file state variables [nfield, ny, nx] * nens
    ##to nc files [nens, nt, nz, ny, nx] * num_var files
    info = c.state_info
    dims = {'member':None, 'time':None, 'level':None, 'y':info['ny'], 'x':info['nx']}
    mask = np.full((info['ny'], info['nx']), False, dtype=bool)

    for v in list(set(rec['name'] for i, rec in info['fields'].items())):
        print('converting '+v)
        outfile = filename.replace('.bin','.'+v+'.nc')

        members = np.arange(c.nens)
        times = np.array(list(set(t2h(rec['time']) for i, rec in info['fields'].items() if rec['name']==v)))
        levels = np.array(list(set(rec['k'] for i, rec in info['fields'].items() if rec['name']==v)))

        for rec_id, rec in [(i,r) for i,r in info['fields'].items() if r['name']==v]:
            for mem_id in range(c.nens):
                ##get the field from bin file
                fld = read_field(filename, info, mask, mem_id, rec_id)
                ##get record number along time,level dimensions
                id_time = int([i for i,t in enumerate(times) if t2h(rec['time'])==t][0])
                id_level = int([i for i,z in enumerate(levels) if rec['k']==z][0])
                recno = {'member':mem_id, 'time':id_time, 'level':id_level}
                if rec['is_vector']:
                    comp = ('_x', '_y')
                    for i in range(2):
                        nc_write_var(outfile, dims, v+comp[i], fld[i,...], recno=recno)
                else:
                    nc_write_var(outfile, dims, v, fld, recno=recno)

        ##output dimensions
        nc_write_var(outfile, {'member':len(members)}, 'member', members, attr={'standard_name':'member', 'long_name':'ensemble member'})
        nc_write_var(outfile, {'time':len(times)}, 'time', times, attr={'standard_name':'time', 'long_name':'time', 'units':'hours since 1900-01-01 00:00:00', 'calendar':'standard'})
        nc_write_var(outfile, {'level':len(levels)}, 'level', levels, attr={'standard_name':'level', 'long_name':'vertical level'})

    ##TODO: output x,y dimensions
