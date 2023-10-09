import numpy as np
import matplotlib.pyplot as plt
import config.constants as cc
from datetime import datetime, timedelta

outdir = cc.SCRATCH+'/nextsim_ens_runs/wind10m_err1.0/working/'

cmap = [plt.cm.rainbow(m) for m in np.linspace(0, 1, 91)]
t0 = datetime(2007, 1, 1, 6, 0, 0)
dt = timedelta(hours=6)

ke_spec1 = np.load(outdir+'/001/output/ke_spec.npy')
nk, nt = ke_spec1.shape
nens = 40
nt = 56

ke_spec = np.zeros((nens, nk, nt))
ke_pert_spec = np.zeros((nens, nk, nt))
for i in range(nens):
    ke_spec[i, :, :] = np.load(outdir+'/{:03d}/output/ke_spec.npy'.format(i+1))[:, 0:nt]
    ke_pert_spec[i, :, :] = np.load(outdir+'/{:03d}/output/ke_pert_spec.npy'.format(i+1))[:, 0:nt]

for n in np.arange(nt):
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)

    wn = np.arange(1, nk)
    pwr_sprd = np.sum(ke_pert_spec[:, 1:nk, n], axis=0)/(nens-1)
    ax.loglog(wn, pwr_sprd, color='c', label='ens spread')
    pwr_mean = np.mean(ke_spec[:, 1:nk, n], axis=0)
    ax.loglog(wn, pwr_mean, color='b', label='reference')

    ##some ref lines
    # wn = np.arange(5., 20.)
    # ax.loglog(wn, 6e2*wn**(-3), color=[.7, .7, .7])
    # wn = np.arange(30., 500.)
    # ax.loglog(wn, 1e1*wn**(-5./3), color=[.7, .7, .7])

    ax.grid()
    wl = np.array([3600, 1800, 700, 320, 100, 48, 20])
    ax.set_xticks(5500./wl)
    ax.set_xticklabels(wl)
    ax.set_ylim([1e-5, 1e1])
    t = t0 + n*dt
    tstr = t.strftime('%Y-%m-%d %H:%M')
    ax.set_title('surface wind KE (m2/s2) '+tstr, fontsize=14)
    ax.set_xlabel('wavelength (km)')
    ax.legend(loc='upper right')
    plt.savefig(outdir+'/figs/spectrum/{:03d}.png'.format(n), dpi=200)
    plt.close()

