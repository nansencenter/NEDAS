import numpy as np
from utils.space_op import gradx, grady
from .random_field import random_field_gaussian

###generate a random perturbation for wind u,v and pressure
def random_pres_wind_perturb(grid, dt,                  ##grid obj for the 2D domain; dt: time interval (hours)
                             prev_pres, prev_u, prev_v, ##pres, u, v, perturbation from previous t
                             ampl_pres, ampl_wind,      ##pres amplitude (Pa); wind amplitude (m/s)
                             hscale,                    ##horizontal decorrelation length scale (meters)
                             tscale,                    ##time decorrelation scale (hours)
                             pres_wind_relate=True,     ##if true: calc wind from pres according to pres-wind relation
                             wlat=15.,                  ##latitude bound for pure geostroph wind
                             ):

    ##some constants
    plon, plat = grid.proj(grid.x, grid.y, inverse=True)
    r2d = 180/np.pi
    rhoa = 1.2
    fcor = 2*np.sin(plat/r2d)*2*np.pi/86400  ##coriolis

    rt = tscale / dt  ##time correlation length
    wt = np.exp(-1)**(1/rt)  ##weight on prev pert

    pres = random_field_gaussian(grid.nx, grid.ny, ampl_pres, hscale/grid.dx)

    if pres_wind_relate:  ##calc u,v from pres according to pres-wind relation

        ##pres gradients
        dpresx = gradx(pres, grid.dx)
        dpresy = grady(pres, grid.dx)

        ##geostrophic wind near poles
        vcor =  dpresx / fcor / rhoa
        ucor = -dpresy / fcor / rhoa

        ##gradient wind near equator
        ueq = -dpresx / fcor / rhoa
        veq = -dpresy / fcor / rhoa

        wcor = np.sin(np.minimum(wlat, plat) / wlat * np.pi * 0.5)
        u = wcor*ucor + (1-wcor)*ueq
        v = wcor*vcor + (1-wcor)*veq

    else:  ##perturb u, v independently from pres
        ##wind power spectrum given by hscale and ampl_wind
        u = random_field_gaussian(grid.nx, grid.ny, ampl_wind, hscale/grid.dx)
        v = random_field_gaussian(grid.nx, grid.ny, ampl_wind, hscale/grid.dx)

    ampl_wind_norm = np.hypot(np.std(u), np.std(v))
    u *= ampl_wind / ampl_wind_norm
    v *= ampl_wind / ampl_wind_norm

    ##apply temporal correlation
    if prev_pres is not None and prev_u is not None and prev_v is not None:
        pres = wt * prev_pres + np.sqrt(1 - wt**2) * pres
        u = wt * prev_u + np.sqrt(1 - wt**2) * u
        v = wt * prev_v + np.sqrt(1 - wt**2) * v

    return pres, u, v

