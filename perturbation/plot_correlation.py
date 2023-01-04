import numpy as np
import matplotlib.pyplot as plt
import grid.io.netcdf as nc
from diag import sample_correlation
import config.constants as cc
import sys

##read synfoc_*nc and plot the diagnosed correlation length scales (time/space)

outdir = sys.argv[1]
vname = sys.argv[2]
n_sample = int(sys.argv[3])
nt = int(sys.argv[4])
dx = float(sys.argv[5])  ##km
dt = float(sys.argv[6])  ##hour
cp = cc.CYCLE_PERIOD/60.

_, ny, nx = nc.read(outdir+'/{:03d}/{:04d}/perturb.nc'.format(1, int(dt/cp)), vname).shape
pert = np.zeros((n_sample, nt, nx))
for n in range(n_sample):
    for m in range(1, nt):
        pert[n, m-1, :] = nc.read(outdir+'/{:03d}/{:04d}/perturb.nc'.format(n+1, int(m*dt/cp)), vname)[0, int(0.5*ny), :]

##length in time/space to sample
L = int(0.25*nx)
hcorr = np.zeros(L)
for l in range(L):
    samp1 = pert[:, :, int(0.25*nx):int(0.75*nx)].flatten()
    samp2 = np.roll(pert, -l, axis=2)[:, :, int(0.25*nx):int(0.75*nx)].flatten()
    hcorr[l] = sample_correlation(samp1, samp2)

T = int(0.5*nt)
tcorr = np.zeros(T)
for t in range(T):
    samp1 = pert[:, 0:T, :].flatten()
    samp2 = np.roll(pert, -t, axis=1)[:, 0:T, :].flatten()
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

plt.savefig(outdir+'/fig_correlation.pdf')

