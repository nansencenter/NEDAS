import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

##read synfoc_*nc and plot the diagnosed correlation length scales (time/space)

def sample_correlation(x1, x2):
  x1_mean = np.mean(x1)
  x2_mean = np.mean(x2)
  x1p = x1 - x1_mean
  x2p = x2 - x2_mean
  cov = np.sum(x1p * x2p)
  x1_norm = np.sum(x1p ** 2)
  x2_norm = np.sum(x2p ** 2)
  corr = cov/np.sqrt(x1_norm * x2_norm)
  return corr


vname = 'slp'
xdim = 900
ydim = 800
dx = 5
dt = 6

n_field = 40
nens = 20

outdir = 'output'

pert = np.zeros((nens, n_field, ydim, xdim))
for m in range(nens):
    for n in range(n_field):
        f = Dataset(outdir+'/{:03d}'.format(m+1)+'/synforc_{:04d}.nc'.format(n))
        pert[m, n, :, :] = np.array(f[vname])

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
    samp2 = pert[:, t:T+t, 0, :].flatten()
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

plt.savefig('out.pdf')

