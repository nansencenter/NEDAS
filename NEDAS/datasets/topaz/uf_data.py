import struct

def read_uf_data(file):
    data = []
    fmt_str = 'dd5sdddiiidddd?iidii'  ##according to mod_measurement type def
    ##value, variance, id_str, lon, lat, depth, ipiv, jpiv, ns, a1, a2, a3, a4, stat, i_orig_grid, j_orig_grid, h, date, orig_id
    recl = struct.calcsize(fmt_str)
    with open(file, 'rb') as f:
        while True:
            d = f.read(recl)
            if not d:
                break
            data.append(struct.unpack(fmt_str, d))
    return data

