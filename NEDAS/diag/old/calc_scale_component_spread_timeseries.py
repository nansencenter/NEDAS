import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
import datetime
import sys

#date range
t1 = datetime.datetime(2007, 2, 1, 6, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 56
nens = 10

outdir=sys.argv[1]
vname=sys.argv[2]  ##variable type (sic, sit, siu, siv, deform, damage)
s=int(sys.argv[3])  ##scale

for n in range(nt):
    t = t1 + n*dt
    tstr = t.strftime('%Y%m%dT%H%M%SZ')
    var = np.zeros((nens, 1500, 1666))
    for m in range(nens):
        var[m, :, :] = np.load(outdir+'/{:03d}'.format(m+1)+'/output/'+vname+'_scale{}'.format(s+1)+'_'+tstr+'.npy')[0, :, :]
    var_sprd = np.nanstd(var, axis=0)
    var_err = np.sqrt(np.nanmean(var_sprd**2))
    np.save(outdir+'/'+vname+'_sprd_scale{}'.format(s+1)+'_'+tstr+'.npy', var_err)

