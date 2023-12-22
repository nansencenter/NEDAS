import numpy as np
from perturb import random_field_powerlaw
from fft_lib import fft2, ifft2, get_wn

##initial condition for model: a rankine vortex embedded in a random wind flow
def initialize(grid,
               Vmax,        ##maximum wind speed, m/s
               Rmw,         ##radius of maximum wind, m
               Vbg,         ##background flow wind speed, m/s
               Vslope,
               loc_sprd=0,  ##the spread in center location, m
               ):

    ##the vortex is randomly placed in the domain
    center_x = 0.5*(grid.xmin+grid.xmax) + np.random.normal(0, loc_sprd)
    center_y = 0.5*(grid.ymin+grid.ymax) + np.random.normal(0, loc_sprd)

    vortex = rankine_vortex(grid, Vmax, Rmw, center_x, center_y)

    ##the background wind field is randomly drawn
    bkg_flow = random_flow(grid, Vbg, Vslope)

    return vortex + bkg_flow


##generate a rankine vortex velocity field
def rankine_vortex(grid,    ##model grid obj
                   Vmax,    ##maximum wind speed
                   Rmw,     ##radius of max wind
                   center_x, center_y,  ##vortex center coords x,y
                  ):

    ##radius from vortex center
    r = np.hypot(grid.x - center_x, grid.y - center_y)
    r[np.where(r==0)] = 1e-10  ##avoid divide by 0

    ##wind speed profile with radius
    wspd = np.zeros(r.shape)
    ind = np.where(r <= Rmw)
    wspd[ind] = Vmax * r[ind] / Rmw
    ind = np.where(r > Rmw)
    wspd[ind] = Vmax * (Rmw / r[ind])**1.5
    wspd[np.where(r==0)] = 0

    u = -wspd * (grid.y - center_y) / r
    v = wspd * (grid.x - center_x) / r

    return np.array([u, v])


##random background flow wind field is given by
def random_flow(grid,
                amp,         ##wind speed amplitude
                power_law,   ##wind field power spectrum slope: 0=white noise; -1=red noise
                ):
    ny, nx = grid.x.shape
    fld = np.zeros((2, ny, nx))
    dx = grid.dx

    ##generate random streamfunction for the wind
    ##note: streamfunc powerlaw = wind powerlaw - 2
    psi = random_field_powerlaw(nx, ny, 1, power_law-2)

    ##convert to wind
    u = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0*dx)
    v = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0*dx)

    ##normalize and scale to the required wind amp
    u = amp * (u - np.mean(u)) / np.std(u)
    v = amp * (v - np.mean(v)) / np.std(v)

    return np.array([u, v])


def advance_time(fld, dx, t_intv, dt, gen, diss):
    ##input wind components, convert to spectral space
    uh = fft2(fld[0, :, :])
    vh = fft2(fld[1, :, :])

    ##convert to zeta
    ki, kj = get_scaled_wn(uh, dx)
    zetah = 1j * (ki*vh - kj*uh)
    k2 = ki**2 + kj**2
    k2[0, 0] = 1.
    #k2 = np.where(k2!=0, k2, np.ones_like(k2)) #avoid singularity in inversion

    ##run time loop:
    ##t_intv is run period in hours
    ##dt is model time step in seconds
    for n in range(int(t_intv*3600/dt)):
        ##use rk4 numeric scheme to integrate forward in time:
        rhs1 = forcing(uh, vh, zetah, dx, gen, diss)
        zetah1 = zetah + 0.5*dt*rhs1
        rhs2 = forcing(uh, vh, zetah1, dx, gen, diss)
        zetah2 = zetah + 0.5*dt*rhs2
        rhs3 = forcing(uh, vh, zetah2, dx, gen, diss)
        zetah3 = zetah + dt*rhs3
        rhs4 = forcing(uh, vh, zetah3, dx, gen, diss)
        zetah = zetah + dt*(rhs1/6.0 + rhs2/3.0 + rhs3/3.0 + rhs4/6.0)

        ##inverse zeta to get u, v
        psih = -zetah / k2
        uh = -1j * kj * psih
        vh = 1j * ki * psih

    u = ifft2(uh)
    v = ifft2(vh)

    return np.array([u, v])


##scaled wavenumber k for pseudospectral method
def get_scaled_wn(x, dx):
    n = x.shape[0]
    wni, wnj = get_wn(x)
    ki = (2.*np.pi) * wni / (n*dx)
    kj = (2.*np.pi) * wnj / (n*dx)
    return ki, kj


def forcing(u, v, zeta, dx, gen, diss):
    ki, kj = get_scaled_wn(zeta, dx)
    ug = ifft2(u)
    vg = ifft2(v)
    ##advection term:
    f = -fft2(ug*ifft2(1j*ki*zeta) + vg*ifft2(1j*kj*zeta))
    ##generation term:
    vmax = np.max(np.sqrt(ug**2+vg**2))
    if vmax > 75:  ##cut off generation if vortex intensity exceeds limit
        gen = 0
    n = zeta.shape[0]
    k2d = np.sqrt(ki**2 + kj**2)*(n*dx)/(2.*np.pi)
    kc = 8
    dk = 3
    gen_response = np.exp(-0.5*(k2d-kc)**2/dk**2)
    f += gen*gen_response*zeta
    ##dissipation term:
    f -= diss*(ki**2+kj**2)*zeta
    return f


