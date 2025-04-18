import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
from pynextsim import NextsimBin
import bamg
import datetime
import sys
import os

##date range
t1 = datetime.datetime(2007, 2, 1, 6, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 76
x = grid.x_ref
y = grid.y_ref
nx, ny = x.shape

outdir=sys.argv[1]  ##nextsim output dir
v=int(sys.argv[2]) ##var id

##output variable
vname = ('Concentration', 'Thickness', 'M_VT', 'Damage', 'deform')[v]
voname = ('sic', 'sit', 'siu', 'damage', 'deform')[v]

##read bin files
out = np.zeros((2, nx, ny))

for n in range(nt):
    t = t1 + n*dt

    binfile = outdir+'/field_'+t.strftime('%Y%m%dT%H%M%SZ')+'.bin'
    nb = NextsimBin(binfile)
    tmp = nb.get_gridded_vars([vname], x, y)
    if vname == 'M_VT':
        out[0, :, :] = tmp['M_VT_1']
        out[1, :, :] = tmp['M_VT_2']
    if vname == 'deform':
        binfile_past = outdir+'/field_'+(t1+(n-1)*dt).strftime('%Y%m%dT%H%M%SZ')+'.bin'
        nb0 = NextsimBin(binfile_past)
        e1, e2, e3, area, pr, tri, x_e, y_e, u_e, v_e = nb.get_deformation_2files(nb0)
        var_in = [np.array(e2*24*60*60, dtype='double')] ##convert to 1/day
        tmp = bamg.interpMeshToPoints(tri.flatten()+1,
                                      x_e.astype(np.double),
                                      y_e.astype(np.double),
                                      var_in,
                                      x.flatten(), y.flatten(), True, np.nan)
        sic = nb.get_gridded_vars(['Concentration'], x, y)['Concentration']
        tmp = np.reshape(tmp, x.shape)
        tmp[np.where(np.isnan(sic))] = np.nan
        out[0, :, :] = tmp
    if vname != 'M_VT' and vname != 'deform':
        out[0, :, :] = tmp[vname]

    ##output nc file
    out_path = outdir+'/output/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_dat = np.zeros((1, ny, nx))
    out_dat[0, :, :] = out[0, :, :].T
    nc.write(out_path+t.strftime('%Y%m%dT%H%M%SZ')+'.nc', {'t':0, 'y':ny, 'x':nx}, voname, out_dat)
    if vname == 'M_VT':
        out_dat[0, :, :] = out[1, :, :].T
        nc.write(out_path+t.strftime('%Y%m%dT%H%M%SZ')+'.nc', {'t':0, 'y':ny, 'x':nx}, 'siv', out_dat)

