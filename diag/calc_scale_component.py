import numpy as np
from scale_util import *
import datetime
import sys

##Define scale bands
krange = (5, 20, 50, 100)

#date range
t1 = datetime.datetime(2021, 1, 1, 0, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 41

vname=sys.argv[1]  ##variable type (sic, sit, siu, siv, deform, damage)
m=int(sys.argv[2])  ##member id
s=int(sys.argv[3])  ##scale

for n in range(nt):
    t = t1 + n*dt
    tstr = t.strftime('%Y%m%dT%H%M%SZ')
    outdir = 'output/ensemble_run/{:03d}'.format(m+1)
    var = np.load(outdir+'/'+vname+'_'+tstr+'.npy')
    mask = np.isnan(var)
    var[np.where(mask)] = 0.
    var_s = get_scale(var, krange, s)
    var_s[np.where(mask)] = np.nan
    np.save(outdir+'/'+vname+'_scale{}'.format(s+1)+'_'+tstr+'.npy', var_s)

