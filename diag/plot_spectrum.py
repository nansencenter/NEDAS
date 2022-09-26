import numpy as np
import matplotlib.pyplot as plt
import datetime
from scale_util import *
import sys
import os

t1 = datetime.datetime(2021, 1, 1, 0, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 41
nens = 10

x, y = np.load('output/grid.npy')

if not os.path.exists('output/figs/spectrum'):
    os.makedirs('output/figs/spectrum')

n = int(sys.argv[1])
t = t1+n*dt
tstr = t.strftime('%Y%m%dT%H%M%SZ')

deform = np.load('output/ensemble_run/001/deform_'+tstr+'.npy')
u = np.load('output/ensemble_run/001/siu_'+tstr+'.npy')
v = np.load('output/ensemble_run/001/siv_'+tstr+'.npy')
mask = np.isnan(deform)
deform[np.where(mask)] = 0.
wn, pwr = pwrspec2d(deform)

##averaged spectrum of members state
pwr_deform_mem = np.zeros(pwr.shape)
pwr_velocity_mem = np.zeros(pwr.shape)
for m in range(nens):
    deform = np.load( 'output/ensemble_run/{:03d}'.format(m+1)+'/deform_'+tstr+'.npy')
    u = np.load('output/ensemble_run/{:03d}'.format(m+1)+'/siu_'+tstr+'.npy')
    v = np.load('output/ensemble_run/{:03d}'.format(m+1)+'/siv_'+tstr+'.npy')
    deform[np.where(mask)] = 0.
    u[np.where(mask)] = 0.
    v[np.where(mask)] = 0.
    wn, pwr = pwrspec2d(deform)
    pwr_deform_mem += pwr
    wn, pwr_u = pwrspec2d(u)
    wn, pwr_v = pwrspec2d(v)
    pwr_velocity_mem += 0.5*(pwr_u + pwr_v)
pwr_deform_mem = pwr_deform_mem / nens
pwr_velocity_mem = pwr_velocity_mem / nens

##ens mean
deform_mean = np.zeros(deform.shape)
u_mean = np.zeros(deform.shape)
v_mean = np.zeros(deform.shape)
for m in range(nens):
    deform_mean += np.load( 'output/ensemble_run/{:03d}'.format(m+1)+'/deform_'+tstr+'.npy')
    u_mean += np.load('output/ensemble_run/{:03d}'.format(m+1)+'/siu_'+tstr+'.npy')
    v_mean += np.load('output/ensemble_run/{:03d}'.format(m+1)+'/siv_'+tstr+'.npy')
deform_mean = deform_mean/nens
u_mean = u_mean/nens
v_mean = v_mean/nens

##spectrum of ens spread
pwr_deform_sprd = np.zeros(pwr.shape)
pwr_velocity_sprd = np.zeros(pwr.shape)
for m in range(nens):
    diff_deform = np.load( 'output/ensemble_run/{:03d}'.format(m+1)+'/deform_'+tstr+'.npy') - deform_mean
    diff_u = np.load('output/ensemble_run/{:03d}'.format(m+1)+'/siu_'+tstr+'.npy') - u_mean
    diff_v = np.load('output/ensemble_run/{:03d}'.format(m+1)+'/siv_'+tstr+'.npy') - v_mean
    diff_deform[np.where(mask)] = 0.
    diff_u[np.where(mask)] = 0.
    diff_v[np.where(mask)] = 0.
    wn, pwr = pwrspec2d(diff_deform)
    pwr_deform_sprd += pwr
    wn, pwr_u = pwrspec2d(diff_u)
    wn, pwr_v = pwrspec2d(diff_v)
    pwr_velocity_sprd += 0.5*(pwr_u + pwr_v)
pwr_deform_sprd = pwr_deform_sprd / (nens-1) * 2.
pwr_velocity_sprd = pwr_velocity_sprd / (nens-1) * 2.

##make plot
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.loglog(wn, pwr_deform_mem, 'r', linewidth=2, label='deform')
ax.loglog(wn, pwr_deform_sprd, 'y', label='deform spread')
ax.loglog(wn, pwr_velocity_mem, 'b', linewidth=2, label='velocity')
ax.loglog(wn, pwr_velocity_sprd, 'c', label='velocity spread')
##some reference lines
wn1 = np.arange(50.,1000.,10.)
ax.loglog(wn1, 0.3*wn1**-3, 'k', linewidth=0.5)
ax.text(100, 5e-7, r'$\kappa^{-3}$', fontsize=12)
wn1 = np.arange(50.,1000.,10.)
ax.loglog(wn1, 0.004*wn1**-1, 'k', linewidth=0.5)
ax.text(100, 6e-5, r'$\kappa^{-1}$', fontsize=12)
##
ax.set_xlim(0.4, 500)
ax.set_ylim(1e-8, 3e-2)
ax.legend(loc='upper right', fontsize=12)
ax.set_title(t.strftime('%Y-%m-%d %H:%M'), fontsize=20)
plt.savefig('output/figs/spectrum/{:03d}.png'.format(n), dpi=200)
plt.close()
