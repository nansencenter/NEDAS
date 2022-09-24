import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
from cmocean.cm import ice
import os
import sys

#date range
t1 = datetime.datetime(2021, 1, 1, 0, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 41
nens = 40
x, y = np.load('output/grid.npy')
ny, nx = x.shape
plot_crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=60)

v=int(sys.argv[1])  ##variable type
m=int(sys.argv[2])  ##member id

vname = ('sic', 'sit', 'velocity', 'damage', 'deform')[v]
vmin = (     0,     0,          0,      0.8,        0)[v]
vmax = (     1,     3,        0.3,        1,      0.3)[v]
vcolor = ( ice, 'viridis', 'Blues', 'inferno', 'plasma_r')[v]
dv = (vmax-vmin)/40

if not os.path.exists('output/figs/'+vname+'/{:03d}'.format(m+1)):
    os.makedirs('output/figs/'+vname+'/{:03d}'.format(m+1))

for n in range(nt):
    t = t1 + n*dt
    tstr = t.strftime('%Y%m%dT%H%M%SZ')
    outdir = 'output/ensemble_run/{:03d}'.format(m+1)
    if vname=='velocity':
        var_u = np.load(outdir+'/siu_'+tstr+'.npy')
        var_v = np.load(outdir+'/siv_'+tstr+'.npy')
        var = np.sqrt(var_u**2 + var_v**2)
    else:
        var = np.load(outdir+'/'+vname+'_'+tstr+'.npy')


    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': plot_crs})
    var[np.where(var>vmax)]=vmax
    var[np.where(var<vmin)]=vmin

    c = ax.contourf(x, y, var, np.arange(vmin, vmax+dv, dv), cmap=vcolor)
    plt.colorbar(c, fraction=0.025, pad=0.015)
    if vname=='velocity':
        d = 15
        ax.quiver(x[::d, ::d], y[::d, ::d], var_u[::d, ::d], var_v[::d, ::d], scale=5)

    ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='black', zorder=10, alpha=0.5)
    ax.set_title(vname+' member{:03d} '.format(m+1)+t.strftime('%Y-%m-%d %H:%M'), fontsize=20)
    ax.set_xlim(-2.2e6, 1.3e6)
    ax.set_ylim(-1.1e6, 2e6)
    plt.savefig('output/figs/'+vname+'/{:03d}'.format(m+1)+'/{:03d}.png'.format(n), dpi=200)
    plt.close()

