import numpy as np
import matplotlib.pyplot as plt

pert_path = '/cluster/work/users/yingyue/perturb_param/'
plt.figure(figsize=(7,7))
ax = plt.subplot(111)
cmap = [plt.cm.jet(m) for m in np.linspace(0.2, 0.8, 44)]

ke_spec = np.load(pert_path+'sample_ECMWF/ke_spec.npy')
n_sample, nk, nt = ke_spec.shape
dt = 4
for t in np.arange(1, nt):
    wn = np.arange(1., 20.)
    pwr = np.mean(ke_spec[:, :, t], axis=0)/2.
    ax.loglog(wn, pwr[1:20], color=cmap[t*dt][0:3])

sf = 5500./2375.
ke_spec = np.load(pert_path+'sample_AROME/ke_spec.npy')
n_sample, nk, nt = ke_spec.shape
dt = 1
for t in range(1, nt):
    wn = np.arange(10., 200.)*sf
    pwr = np.mean(ke_spec[:, :, t], axis=0)/sf/sf/2.
    ax.loglog(wn, pwr[10:200], color=cmap[t*dt][0:3])

##reference level
wn = np.arange(1., 20.)
pwr = np.mean(np.load(pert_path+'sample_ECMWF/ke_spec_ref.npy')[0, 1:20, 1:], axis=1)
ax.loglog(wn, pwr, 'k')
wn = np.arange(10., 200.)*sf
pwr = np.mean(np.load(pert_path+'sample_AROME/ke_spec_ref.npy')[0, 10:200, 1:], axis=1)/sf/sf
ax.loglog(wn, pwr, 'k')

##some ref lines
wn = np.arange(5., 20.)
ax.loglog(wn, 3e2*wn**(-3), 'r')
wn = np.arange(30., 400.)
ax.loglog(wn, 5*wn**(-5./3), 'r')

ax.set_title('EKE')
wl = np.array([1e4, 1e3, 1e2, 1e1])
ax.set_xticks(5500./wl)
ax.set_xticklabels(wl)
ax.set_xlabel('wavelength (km)')

plt.savefig(pert_path+'fig_spectrum.pdf')

