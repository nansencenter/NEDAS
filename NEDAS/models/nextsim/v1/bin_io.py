import numpy as np
import struct
from NEDAS.utils.conversion import type_convert, type_dic, type_size

def get_info(filename):
    """basic read/write of data from nextsim bin/dat file"""
    info = {}
    with open(filename.replace('.bin','.dat'), 'r') as f:
        lines = f.readlines()

        pos = 4
        for recno, lin in enumerate(lines):
            ss = lin.split()
            if len(ss) < 2:
                continue
            ###get info about the variable record
            v_name = ss[0]
            v_type = ss[1]
            v_len = int(ss[2])
            typeconv = type_convert[v_type]
            v_min_val = typeconv(ss[3])
            v_max_val = typeconv(ss[4])
            v_rec = {}
            v_rec.update({'type':v_type})
            v_rec.update({'len':v_len})
            v_rec.update({'min_val':v_min_val})
            v_rec.update({'max_val':v_max_val})
            v_rec.update({'recno':recno})
            v_rec.update({'pos':pos})

            ##add variable record to the info dict
            info.update({v_name: v_rec})

            ##position of starting point for f.seek
            pos += (4 + v_len *type_size[v_type])
    return info

def read_data(filename, v_name):
    """Read data frim a binary file"""
    info = get_info(filename)
    if v_name not in info:
        raise ValueError('variable %s not found in %s' %(v_name, filename))

    with open(filename, 'rb') as f:
        f.seek(info[v_name]['pos'])
        v_len = info[v_name]['len']
        v_type = info[v_name]['type']
        v_data = np.array(struct.unpack((v_len*type_dic[v_type]),
                                        f.read(v_len*type_size[v_type])))
    return v_data

def write_data(filename, v_name, v_data):
    """
    write some var to existing nextsim restart file
    only for outputing analysis after DA
    to generate new file, see nextsim model documentation
    """
    info = get_info(filename)
    if v_name not in info:
        raise ValueError('variable %s not in %s' %(v_name, filename))
    ##update info with new v_data
    assert v_data.dtype == type_convert[info[v_name]['type']], "input data type mismatch with datfile record"
    assert v_data.size == info[v_name]['len'], f"input data size {v_data.size} mismatch with datfile record {info[v_name]['len']}"
    info[v_name]['min_val'] = np.nanmin(v_data)
    info[v_name]['max_val'] = np.nanmax(v_data)

    ##write the dat file with updated info
    with open(filename.replace('.bin','.dat'), 'wt') as f:
        for v in info:
            ss = '%s %s %i' %(v, info[v]['type'], info[v]['len'])
            if info[v]['type'] == 'int':
                ss += ' %i %i' %(info[v]['min_val'], info[v]['max_val'])
            else:
                ss += ' %g %g' %(info[v]['min_val'], info[v]['max_val'])
            f.write(ss+'\n')

    ##write v_data to the bin file
    with open(filename, 'r+b') as f:
        if len(v_data) != info[v_name]['len']:
            raise ValueError('v_data length does not match length on record')
        f.seek(info[v_name]['pos'])
        f.write(v_data.tobytes())
