import numpy as np
import matplotlib.pyplot as plt
import grid.io.netcdf as nc
from diag import sample_correlation
import config.constants as cc
import grid.multiscale as ms
import sys

##read perturb_*.nc and diagnose correlation length scales (time/space)

outdir = sys.argv[1]
vname = sys.argv[2]
n_sample = int(sys.argv[3])
nt = int(sys.argv[4])
dx = float(sys.argv[5])  ##km
dt = float(sys.argv[6])  ##hour

cp = cc.CYCLE_PERIOD/60.

_, ny, nx = nc.read(outdir+'/{:03d}/perturb_{:04d}.nc'.format(1, int(dt/cp)), vname).shape
domain_size = np.maximum(nx, ny) * dx
krange = domain_size/np.array([1600., 800, 400., 200., 100, 50.])

plt.figure(figsize=(15,15))
ns = len(krange)

for s in range(ns):
    print(s)

    ###read pert data scale comp
    pert = np.zeros((n_sample, nt, nx))
    for n in range(n_sample):
        for m in range(1, nt):
            tmp = nc.read(outdir+'/{:03d}/perturb_{:04d}.nc'.format(n+1, int(m*dt/cp)), vname)[0, :, :]
            pert[n, m-1, :] = ms.get_scale(tmp, krange, s)[int(0.5*ny), :]

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

    ##time series of pert
    ax = plt.subplot(ns,3,s*3+1)
    err = np.sqrt(np.mean(pert[:, :, :]**2, axis=2))
    r = np.arange(dt, dt*(nt-1), dt)
    for n in range(n_sample):
        ax.plot(r, err[n, 1:nt-1], 'c')
    ax.plot(r, np.mean(err[:, 1:nt-1], axis=0), 'b')
    ax.set_xticks(np.arange(24, 180, 24))
    ax.set_ylim([0, 4.])
    ax.set_xlabel('time (h)')

    ###spatial correlation of pert
    ax = plt.subplot(ns,3,s*3+2)
    r = np.arange(0, dx*L, dx)
    ax.plot(r, hcorr, 'k', linewidth=2)
    ax.plot(r, r*0+0.3, color=[.7, .7, .7])
    ax.set_xlim([0, 600])
    ax.set_ylim([-0.2, 1.0])
    ax.set_xlabel('lag (km)')
    ax.grid()

    ##time correlation of pert
    ax = plt.subplot(ns,3,s*3+3)
    r = np.arange(0, dt*T, dt)
    ax.plot(r, tcorr, 'k', linewidth=2)
    ax.plot(r, r*0+0.3, color=[.7, .7, .7])
    ax.set_xticks(np.arange(24, 100, 24))
    ax.set_xlim([0, 90])
    ax.set_ylim([-0.2, 1.0])
    ax.set_xlabel('lag (h)')
    ax.grid()

plt.savefig(outdir+'/fig_pert_param.pdf')

