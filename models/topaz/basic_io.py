import numpy as np
from .abfile import ABFileRestart
from .grid import ny, nx

def destagger(dat, v_name):
    ##destagger u,v from C-grid
    dat_ = dat.copy()
    if v_name == 'u':
        ##use linear interp in interior
        dat_[:, :-1] = 0.5*(dat[:, :-1] + dat[:, 1:])
        ##use 2nd-order polynomial extrapolation along borders
        dat_[:, -1] = 3*dat[:, -2] - 3*dat[:, -3] + dat[:, -4]
    elif v_name == 'v':
        ##use linear interp in interior
        dat_[:-1, :] = 0.5*(dat[:-1, :] + dat[1:, :])
        ##use 2nd-order polynomial extrapolation along borders
        dat_[-1, :] = 3*dat[-2, :] - 3*dat[-3, :] + dat[-4, :]
    return dat_

def read_data(filename, v_name, level, tlevel=1):
    f = ABFileRestart(filename, 'r', idm=nx, jdm=ny)
    tmp = f.read_field(v_name, level, tlevel)
    v_data = tmp.data
    v_data[np.where(tmp.mask)] = np.nan
    v_data = destagger(v_data, v_name)
    return v_data

def write_data(filename, v_name, v_data):
    pass

