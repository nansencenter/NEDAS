import numpy as np
from assim_tools.basic_io import read_field_info, read_mask, read_field, t2h, nc_write_var

import sys
filename = sys.argv[1]

##convert bin file state variables [nfield, ny, nx] * nens files
##to nc files [nens, nt, nz, ny, nx] * num_var files
info = read_field_info(filename)
mask = read_mask(filename, info)
flds = info['fields']
ny = info['ny']
nx = info['nx']
dims = {'member':None, 'time':None, 'level':None, 'y':ny, 'x':nx}

for v in list(set(rec['var_name'] for i, rec in info['fields'].items())):
    print('converting '+v)
    outfile = filename.replace('.bin','.'+v+'.nc')

    members = np.array(list(set(rec['member']+1 for i, rec in flds.items() if rec['var_name']==v)))
    times = np.array(list(set(t2h(rec['time']) for i, rec in flds.items() if rec['var_name']==v)))
    levels = np.array(list(set(rec['level'] for i, rec in flds.items() if rec['var_name']==v)))

    for i in [i for i, rec in info['fields'].items() if rec['var_name']==v]:
        rec = info['fields'][i]
        ##get the field from bin file
        fld = read_field(filename, info, mask, i)
        ##get record number along time,level dimensions
        id_time = [i for i,t in enumerate(times) if t2h(rec['time'])==t][0]
        id_level = [i for i,z in enumerate(levels) if rec['level']==z][0]
        recno = {'member':rec['member'], 'time':id_time, 'level':id_level}
        if rec['is_vector']:
            comp = ('_x', '_y')
            for i in range(2):
                nc_write_var(outfile, dims, v+comp[i], fld[i,...], recno)
        else:
            nc_write_var(outfile, dims, v, fld, recno)

    ##output dimensions
    nc_write_var(outfile, {'member':len(members)}, 'member', members, attr={'standard_name':'member', 'long_name':'ensemble member'})
    nc_write_var(outfile, {'time':len(times)}, 'time', times, attr={'standard_name':'time', 'long_name':'time', 'units':'hours since 1900-01-01 00:00:00', 'calendar':'standard'})
    nc_write_var(outfile, {'level':len(levels)}, 'level', levels, attr={'standard_name':'level', 'long_name':'vertical level'})
