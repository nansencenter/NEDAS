import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scale_util import *
import datetime
from cmocean.cm import ice
import os
import sys

##Define scale bands
krange = (5, 20, 50, 100)

#date range
t1 = datetime.datetime(2021, 1, 1, 0, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 41
nens = 10
x, y = np.load('output/grid.npy')
ny, nx = x.shape
plot_crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=60)

v=int(sys.argv[1])  ##variable type
m=int(sys.argv[2])  ##member id
s=int(sys.argv[3])  ##scale

vname = ('sic', 'sit', 'velocity', 'damage', 'deform')[v]
vmin = (     0,     0,          0,      0.8,        0)[v]
vmax = (     1,     3,        0.3,        1,      0.3)[v]
vcolor = ( ice, 'viridis', 'Blues', 'inferno', 'plasma_r')[v]
dv = (vmax-vmin)/40

if not os.path.exists('output/figs/'+vname+'_scale{}'.format(s+1)+'/{:03d}'.format(m+1)):
    os.makedirs('output/figs/'+vname+'_scale{}'.format(s+1)+'/{:03d}'.format(m+1))

for n in range(nt):
    t = t1 + n*dt
    tstr = t.strftime('%Y%m%dT%H%M%SZ')
    outdir = 'output/ensemble_run/{:03d}'.format(m+1)
    if vname=='velocity':
        var_u = np.load(outdir+'/siu_'+tstr+'.npy')
        var_v = np.load(outdir+'/siv_'+tstr+'.npy')
        mask = np.isnan(var_u)
        var_u[np.where(mask)] = 0.
        var_v[np.where(mask)] = 0.
        var_u_s = get_scale(var_u, krange, s)
        var_v_s = get_scale(var_v, krange, s)
        var_s = np.sqrt(var_u_s**2 + var_v_s**2)
        var_s[np.where(mask)] = np.nan
    else:
        var = np.load(outdir+'/'+vname+'_'+tstr+'.npy')
        mask = np.isnan(var)
        var[np.where(mask)] = 0.
        var_s = get_scale(var, krange, s)
        var_s[np.where(mask)] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': plot_crs})
    var_s[np.where(var_s>vmax)]=vmax
    var_s[np.where(var_s<vmin)]=vmin

    c = ax.contourf(x, y, var_s, np.arange(vmin, vmax+dv, dv), cmap=vcolor)
    plt.colorbar(c, fraction=0.025, pad=0.015)
    if vname=='velocity':
        d = 15
        ax.quiver(x[::d, ::d], y[::d, ::d], var_u_s[::d, ::d], var_v_s[::d, ::d], scale=5)

    ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='black', zorder=10, alpha=0.5)
    ax.set_title(vname+' member{:03d} '.format(m+1)+t.strftime('%Y-%m-%d %H:%M'), fontsize=20)
    ax.set_xlim(-2.2e6, 1.3e6)
    ax.set_ylim(-1.1e6, 2e6)
    plt.savefig('output/figs/'+vname+'_scale{}'.format(s+1)+'/{:03d}'.format(m+1)+'/{:03d}.png'.format(n), dpi=200)
    plt.close()

