import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from diag.metric import sample_correlation
from diag.multiscale import pwrspec2d

##read synfoc_*nc and plot the diagnosed correlation length scales (time/space)

vname = 'uwind'
xdim = 900
ydim = 900
dx = 5
dt = 6

n_field = 50
nens = 20

outdir = '/cluster/work/users/yingyue/perturbation/'

pert = np.zeros((nens, n_field, ydim, xdim))
for n in range(n_field):
    f = Dataset(outdir+'/synforc_{:04d}.nc'.format(n))
    pert[:, n, :, :] = np.array(f[vname])[0, :, :, :]

##length in time/space to sample
L = int(0.25*xdim)
hcorr = np.zeros(L)
for l in range(L):
    samp1 = pert[:, 0, :, int(0.25*xdim):int(0.75*xdim)].flatten()
    samp2 = np.roll(pert[:, 0, :, int(0.25*xdim):int(0.75*xdim)], -l, axis=2).flatten()
    hcorr[l] = sample_correlation(samp1, samp2)

T = int(0.5*n_field)
tcorr = np.zeros(T)
for t in range(T):
    samp1 = pert[:, 0:T, 0, :].flatten()
    samp2 = np.roll(pert, -t, axis=1)[:, 0:T, 0, :].flatten()
    tcorr[t] = sample_correlation(samp1, samp2)

plt.figure(figsize=(10,5))
ax = plt.subplot(1,2,1)
ax.plot(np.arange(0, dx*L, dx), hcorr, 'k', linewidth=2)
ax.set_title('spatial correlation')
ax.set_xlabel('lag (km)')

ax = plt.subplot(1,2,2)
ax.plot(np.arange(0, dt*T, dt), tcorr, 'k', linewidth=2)
ax.set_title('temporal correlation')
ax.set_xlabel('lag (h)')

plt.savefig(outdir+'fig_correlation.pdf')

