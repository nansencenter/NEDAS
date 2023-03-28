import numpy as np
import matplotlib.pyplot as plt
import config.constants as cc

pert_path = cc.SCRATCH+'/perturb_param/'
plt.figure(figsize=(6,5))
ax = plt.subplot(111)
cmap = [plt.cm.rainbow(m) for m in np.linspace(0, 1, 44)]

ke_spec = np.load(pert_path+'sample_ECMWF/ke_spec.npy')
n_sample, nk, nt = ke_spec.shape
dt = 4
for t in np.arange(1, nt):
    wn = np.arange(1., 50.)
    pwr = np.mean(ke_spec[50:, :, t], axis=0)
    ax.loglog(wn, pwr[1:50], color=cmap[t*dt][0:3])

sf = 5500./2375.
ke_spec = np.load(pert_path+'sample_AROME/ke_spec.npy')
n_sample, nk, nt = ke_spec.shape
dt = 1
for t in range(1, nt):
    wn = np.arange(10., 200.)*sf
    pwr = np.mean(ke_spec[50:, :, t], axis=0)/sf/sf
    ax.loglog(wn, pwr[10:200], color=cmap[t*dt][0:3])

##reference level
wn = np.arange(1., 50.)
pwr = np.mean(np.load(pert_path+'sample_ECMWF/ke_spec_ref.npy')[:, 1:50, 0], axis=0)
ax.loglog(wn, pwr, 'k', linewidth=2)
wn = np.arange(10., 200.)*sf
pwr = np.mean(np.load(pert_path+'sample_AROME/ke_spec_ref.npy')[:, 10:200, 0], axis=0)/sf/sf
ax.loglog(wn, pwr, 'k', linewidth=2)

##some ref lines
wn = np.arange(5., 20.)
ax.loglog(wn, 6e2*wn**(-3), color=[.7, .7, .7])
wn = np.arange(30., 500.)
ax.loglog(wn, 1e1*wn**(-5./3), color=[.7, .7, .7])

ax.grid()
wl = np.array([2000., 500., 150., 50., 20.])
ax.set_xticks(5500./wl)
ax.set_xticklabels(wl)

plt.savefig(pert_path+'fig_spectrum.pdf')

