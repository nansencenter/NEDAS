import numpy as np
from NEDAS.utils.random_perturb import random_field_powerlaw
from NEDAS.utils.fft_lib import fft2, ifft2, get_wn

def initial_condition(grid, Vmax, Rmw, Vbg, Vslope, loc_sprd=0):
    """
    Initialize the 2d vortex model with a Rankine vortex embedded in a random wind flow.

    Args:
        grid (Grid): The model domain, doubly periodic, described by a Grid obj
        Vmax (float): Maximum wind speed (vortex intensity), m/s
        Rmw (float): Radius of maximum wind (vortex size), m
        Vbg (float): Background flow average wind speed, m/s
        Vslope (int): Background flow kinetic energy spectrum power law (typically -2)
        loc_sprd (float, optional): The ensemble spread in vortex center position, m

    Returns:
        np.ndarray: The vector velocity field, shape (2, ny, nx).
    """
    ##the vortex is randomly placed in the domain
    center_x = 0.5*(grid.xmin+grid.xmax) + np.random.normal(0, loc_sprd)
    center_y = 0.5*(grid.ymin+grid.ymax) + np.random.normal(0, loc_sprd)

    vortex = rankine_vortex(grid, Vmax, Rmw, center_x, center_y)

    ##the background wind field is randomly drawn
    bkg_flow = random_flow(grid, Vbg, Vslope)

    return vortex + bkg_flow

def rankine_vortex(grid, Vmax, Rmw, center_x, center_y):
    """
    Generate a Rankine vortex velocity field.

    Args:
        grid (Grid): The model domain, doubly periodic
        Vmax (float): Maximum wind speed (vortex intensity), m/s
        Rmw (float): Radius of maximum wind (vortex size), m
        center_x (float): Vortex center X-coordinate.
        center_y (float): Vortex center Y-coordinate.

    Returns:
        np.ndarray: The vector velocity field with shape (2, ny, nx)
    """
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

def random_flow(grid, amp, power_law):
    """
    Generate a random velocity field as the background flow

    Args:
        grid (Grid): The model domain, doubly periodic, described by a Grid obj
        amp (float): wind speed amplitude, m/s
        power_law (int): wind kinetic energy spectrum power law (typically -2)

    Returns:
        np.ndarray: The vector velocity field with shape (2, ny, nx)
    """
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
    """
    Advance forward in time to integrate the model (forecasting)

    Args:
        fld (np.ndarray): The prognostic velocity field with shape (2,ny,nx)
        dx (float): Model grid spacing, meter
        t_intv (float): Integration time period, hour
        dt (float): Model time step, second
        gen (float): Vorticity generation rate
        diss (float): Dissipation rate

    Returns:
        np.ndarray: The forecast velocity field
    """
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

def get_scaled_wn(x, dx):
    """scaled wavenumber k for pseudospectral method"""
    n = x.shape[0]
    wni, wnj = get_wn(x)
    ki = (2.*np.pi) * wni / (n*dx)
    kj = (2.*np.pi) * wnj / (n*dx)
    return ki, kj

def forcing(u, v, zeta, dx, gen, diss):
    """forcing terms on RHS of prognostic equations"""
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

