import numpy as np
from pynextsim import NextsimBin
import bamg
import datetime
import sys

##date range
t1 = datetime.datetime(2021, 1, 1, 0, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 41
nens = 40  #ens size
x, y = np.load('output/grid.npy')
ny, nx = x.shape

outdir=sys.argv[1]  ##nextsim output dir
v=int(sys.argv[2]) ##var id

##output variable
vname = ('Concentration', 'Thickness', 'M_VT', 'Damage', 'deform')[v]
voname = ('sic', 'sit', 'siu', 'damage', 'deform')[v]

##read bin files
out = np.zeros((2, ny, nx))

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
    else:
        out[0, :, :] = tmp[vname]

    ##output npy file
    np.save(outdir+'/'+voname+'_'+t.strftime('%Y%m%dT%H%M%SZ')+'.npy', out[0, :, :])
    if vname == 'M_VT':
        np.save(outdir+'/siv'+'_'+t.strftime('%Y%m%dT%H%M%SZ')+'.npy', out[1, :, :])

