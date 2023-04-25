import numpy as np
import matplotlib.pyplot as plt
import config.constants as cc

pert_path = cc.SCRATCH+'/perturbation/'
plt.figure(figsize=(6,5))
ax = plt.subplot(111)
cmap = [plt.cm.rainbow(m) for m in np.linspace(0, 1, 56)]

ke_spec1 = np.load(pert_path+'/001/ke_spec.npy')
nk, nt = ke_spec1.shape
n_sample = 40
nt = 56

ke_spec = np.zeros((n_sample, nk, nt))
for i in range(n_sample):
    ke_spec[i, :, :] = np.load(pert_path+'/{:03d}/ke_spec.npy'.format(n_sample))[:, 0:nt]

dt = 1
for t in np.arange(1, nt):
    wn = np.arange(1., 500.)
    pwr = np.mean(ke_spec[:, :, t], axis=0)
    ax.loglog(wn, pwr[1:500], color=cmap[t*dt][0:3])

##some ref lines
wn = np.arange(5., 20.)
ax.loglog(wn, 6e2*wn**(-3), color=[.7, .7, .7])
wn = np.arange(30., 500.)
ax.loglog(wn, 1e1*wn**(-5./3), color=[.7, .7, .7])

ax.grid()
wl = np.array([3600, 1800, 700, 320, 100, 48, 20])
ax.set_xticks(5500./wl)
ax.set_xticklabels(wl)
ax.set_ylim([1e-5, 1e1])

plt.savefig(pert_path+'fig_spectrum.pdf')

