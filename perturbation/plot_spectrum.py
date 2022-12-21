import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from diag.multiscale import pwrspec2d

xdim = 900
ydim = 900
dx = 5
dt = 6

nens = 20
outdir = '/cluster/work/users/yingyue/perturbation/'

f = Dataset(outdir+'/synforc_{:04d}.nc'.format(0))
pert = np.array(f['uwind'])
wn, pwr = pwrspec2d(pert[0, 0, :, :])


plt.figure(figsize=(10,5))
ax = plt.subplot(111)
ax.loglog(wn, pwr)
ax.set_title('')
ax.set_xlabel('')

plt.savefig(outdir+'fig_spectrum.pdf')

