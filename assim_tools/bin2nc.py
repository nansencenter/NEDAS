import numpy as np
from assim_tools.basic_io import read_field_info, read_mask, read_field, nc_write_var

##convert bin file state variables [nfield, ny, nx] * nens files
##to nc files [nens, nt, nz, ny, nx] * num_var files
def bin2nc(filename):
    info = read_field_info(filename)
    mask = read_mask(filename, info)
    flds = info['fields']
    nens = info['nens']
    ny = info['ny']
    nx = info['nx']
    for v in list(set(rec['var_name'] for i, rec in info['fields'].items())):
        print(v)
        times = list(set(rec['time'] for i, rec in flds.items() if rec['var_name']==v))
        levels = list(set(rec['level'] for i, rec in flds.items() if rec['var_name']==v))
        dat = np.zeros((nens, len(times), len(levels), ny, nx))
        for m in range(nens):
            for n in range(len(times)):
                for z in range(len(levels)):
                    fid = list(set(i for i, rec in flds.items() if rec['var_name']==v and rec['member']==m and rec['time']==times[n] and rec['level']==levels[z]))[0]
                    dat[m, n, z, :, :] = read_field(filename, info, mask, fid)
        dims = {'member':nens, 'time':len(times), 'level':len(levels), 'y':ny, 'x':nx}
        outfile = filename.replace('.bin','.'+v+'.nc')
        nc_write_var(outfile, dims, v, dat)


if __name__=='__main__':
    import sys
    filename = sys.argv[1]
    bin2nc(filename)
