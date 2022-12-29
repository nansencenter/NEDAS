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
    wn = np.arange(1., 30.)
    pwr = np.mean(ke_spec[:, :, t], axis=0)
    ax.loglog(wn, pwr[1:30], color=cmap[t*dt][0:3])

# ke_spec = np.load(pert_path+'sample_AROME/ke_spec.npy')
# n_sample, nk, nt = ke_spec.shape
# dt = 1
# for t in range(1, nt+1):
#     wn = np.arange(20., 200.)*3./2.5
#     pwr = np.mean(ke_spec[:, :, t], axis=0)/((3./2.5)**2)
#     ax.loglog(wn, pwr[20:200], color=cmap[t*dt][0:3])

##some ref lines
wn = np.arange(5., 30.)
ax.loglog(wn, 1e3*wn**(-3), 'r')
wn = np.arange(20., 200.)
ax.loglog(wn, 1e1*wn**(-5./3), 'r')

ax.set_title('EKE')
ax.set_xlabel('wavenumber')

plt.savefig(pert_path+'fig_spectrum.pdf')

