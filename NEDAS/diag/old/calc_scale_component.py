import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
import datetime
import sys

##Define scale bands
krange = (3.7, 18.3, 110.)

#date range
t1 = datetime.datetime(2007, 2, 1, 6, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 56

outdir=sys.argv[1]
vname=sys.argv[2]  ##variable type (sic, sit, siu, siv, deform, damage)
s=int(sys.argv[3])  ##scale

for n in range(nt):
    t = t1 + n*dt
    tstr = t.strftime('%Y%m%dT%H%M%SZ')
    var = nc.read(outdir+'/output/'+tstr+'.nc', vname)
    mask = np.isnan(var)
    var[np.where(mask)] = 0.
    var_s = grid.get_scale(var, krange, s)
    var_s[np.where(mask)] = np.nan
    np.save(outdir+'/output/'+vname+'_scale{}'.format(s+1)+'_'+tstr+'.npy', var_s)

