import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
from cmocean.cm import ice
import os
import sys

#date range
t1 = datetime.datetime(2007, 1, 1, 6, 0, 0)
dt = datetime.timedelta(hours=6)
nt = 84
nens = 40
x = grid.x_ref
y = grid.y_ref
ny, nx = x.shape
plot_crs = grid.crs

outdir = sys.argv[1]
v=int(sys.argv[2])  ##variable type
m=int(sys.argv[3])  ##member id

vname = ('sic', 'sit', 'velocity', 'damage', 'deform')[v]
vmin = (     0,     0,          0,      0.8,        0)[v]
vmax = (     1,     3,        0.3,        1,      0.3)[v]
vcolor = ( ice, 'viridis', 'Blues', 'inferno', 'plasma_r')[v]
dv = (vmax-vmin)/40

mstr = '{:03d}'.format(m)

if not os.path.exists(outdir+'/figs/'+vname+'/'+mstr):
    os.makedirs(outdir+'/figs/'+vname+'/'+mstr)

for n in range(nt):
    t = t1 + n*dt
    tstr = t.strftime('%Y%m%dT%H%M%SZ')
    if vname=='velocity':
        var_u = nc.read(outdir+'/output/{:03d}'.format(m)+'/'+tstr+'.nc', 'siu')[0, :, :].T
        var_v = nc.read(outdir+'/output/{:03d}'.format(m)+'/'+tstr+'.nc', 'siv')[0, :, :].T
        var = np.sqrt(var_u**2 + var_v**2)
    else:
        var = nc.read(outdir+'/output/{:03d}'.format(m)+'/'+tstr+'.nc', vname)[0, :, :].T

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': plot_crs})
    var[np.where(var>vmax)]=vmax
    var[np.where(var<vmin)]=vmin

    c = ax.contourf(x, y, var, np.arange(vmin, vmax+dv, dv), cmap=vcolor)
    plt.colorbar(c, fraction=0.025, pad=0.015)
    if vname=='velocity':
        d = 50
        ax.quiver(x[::d, ::d], y[::d, ::d], var_u[::d, ::d], var_v[::d, ::d], scale=5)

    ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='black', zorder=10, alpha=0.5)
    ax.set_title(vname+' member'+mstr+' '+t.strftime('%Y-%m-%d %H:%M'), fontsize=20)
    ax.set_xlim(-2.2e6, 1.3e6)
    ax.set_ylim(-1.1e6, 2e6)
    plt.savefig(outdir+'/figs/'+vname+'/'+mstr+'/{:03d}.png'.format(n), dpi=200)
    plt.close()

