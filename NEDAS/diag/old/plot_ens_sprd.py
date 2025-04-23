import numpy as np
import config.constants as cc
import grid
import grid.io.netcdf as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
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
# s=int(sys.argv[2])  ##scale, -1 if full scale

vname = ('sic', 'sit', 'velocity', 'damage', 'deform')[v]
vmin = (     0,     0,          0,      0.8,        0)[v]
vmax = (     1,     3,        0.3,        1,      0.3)[v]
dv = (vmax-vmin)/40

sstr=''
# if s==-1:
#     sstr = ''
# else:
#     sstr = '_scale{}'.format(s+1)

if not os.path.exists(outdir+'/figs/'+vname+sstr+'/sprd'):
    os.makedirs(outdir+'/figs/'+vname+sstr+'/sprd')

for n in range(nt):
    t = t1 + n*dt
    tstr = t.strftime('%Y%m%dT%H%M%SZ')

    var_sprd = np.zeros((ny, nx))
    if vname=='velocity':
        u_mean = nc.read(outdir+'/output/mean/'+tstr+'.nc', 'siu')[0, :, :].T
        v_mean = nc.read(outdir+'/output/mean/'+tstr+'.nc', 'siv')[0, :, :].T
        var_mean = u_mean
        for m in range(nens):
            var_sprd += (nc.read(outdir+'/output/{:03d}'.format(m+1)+'/'+tstr+'.nc', 'siu')[0, :, :].T - u_mean)**2
            var_sprd += (nc.read(outdir+'/output/{:03d}'.format(m+1)+'/'+tstr+'.nc', 'siv')[0, :, :].T - v_mean)**2
        var_sprd = np.sqrt(var_sprd/(nens-1))

    else:
        var_mean = nc.read(outdir+'/output/mean/'+tstr+'.nc', vname)[0, :, :].T
        for m in range(nens):
            var_sprd += (nc.read(outdir+'/output/{:03d}'.format(m+1)+'/'+tstr+'.nc', vname)[0, :, :].T - var_mean)**2
        var_sprd = np.sqrt(var_sprd/(nens-1))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': plot_crs})
    var_sprd[np.where(var_sprd>vmax)] = vmax
    var_sprd[np.where(np.isnan(var_mean))] = np.nan
    c = ax.contourf(x, y, var_sprd, np.arange(0, vmax+dv, dv), cmap='Reds')
    plt.colorbar(c, fraction=0.025, pad=0.015)
    ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='black', zorder=10, alpha=0.5)
    ax.set_title(vname+' ensemble spread '+t.strftime('%Y-%m-%d %H:%M'), fontsize=20)
    ax.set_xlim(-2.2e6, 1.3e6)
    ax.set_ylim(-1.1e6, 2e6)
    plt.savefig(outdir+'/figs/'+vname+'/sprd/{:03d}.png'.format(n), dpi=200)
    plt.close()
