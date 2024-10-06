import numpy as np
from scipy.optimize import fsolve
from scipy.ndimage import distance_transform_edt, gaussian_filter
from utils.spatial_operation import gradx, grady, warp
from utils.fft_lib import fft2, ifft2, get_wn

def random_perturb(grid, fld, prev_perturb=None, **kwargs):
    """
    Add random perturbation to the given 2D field
    Input:
        - grid: Grid object for the 2d field [ny, nx]
        - fld: np.array, [...,ny,nx]
        - options: dict of config variables:
            type: str: 'gaussian', 'powerlaw', or 'displace'
            amplitude: float, (or list of floats, in multiscale approach)
            hcorr: float
            tcorr: float
            powerlaw: float
        - prev_perturb; None, or np.array from previous perturbation data
    """
    perturb_type = kwargs['type']
    if perturb_type == 'gaussian':
        amp = kwargs['amp']
        hcorr = kwargs['hcorr']
        tcorr = kwargs['tcorr']
        dx = grid.dx
        dt = kwargs['dt']
        if isinstance(amp, list):   ##allow a list of perturb to be generated (multiscale)
            nscale = len(amp)
        else:
            nscale = 1
            amp, hcorr, tcorr = [amp], [hcorr], [tcorr]
        perturb = np.zeros((nscale,)+fld.shape)
        ##loop over scale s
        for s in range(nscale):
            ##draw a random field for each component [ny,nx] in fld
            for ind in np.ndindex(fld.shape[:-2]):
                perturb[(s,)+ind] = random_field_gaussian(grid.nx, grid.ny, amp[s], hcorr[s]/dx)

            ##create perturbations that are correlated in time
            autocorr = np.exp(-1.0)
            alpha = autocorr**(1.0 / (tcorr[s]/dt))
            if prev_perturb is not None:
                perturb[s] = np.sqrt(1-alpha**2) * perturb[s] + alpha * prev_perturb[s]

        fld += np.sum(perturb, axis=0)

    ##TODO: gaussian with exp error

    elif perturb_type == 'powerlaw':
        perturb = random_field_powerlaw(grid.nx, grid.ny, amp, powerlaw)
        fld += perturb

    elif perturb_type == 'displace':
        mask = np.full(grid.x.shape, False)
        du, dv = random_displacement(grid, mask, amp, hcorr)
        perturb = np.array([du, dv])
        fld = warp(fld, perturb[0,...], perturb[1,...])

    else:
        raise NotImplementedError('unknown perturbation type: '+perturb_type)

    return fld, perturb


def random_field_gaussian(nx, ny, amp, hcorr):
    """
    Random field with a Gaussian spectrum

    Inputs
    - nx, ny: int
      Grid size in x and y (number of grid points)
    - amp: float
      Amplitude (standard deviation) of the random field
    - hcorr: float
      Horizontal decorrelation length (number of grid points)

    Output:
    - fld: np.array, [ny,nx]
      The random 2D field
    """

    fld = np.zeros((ny, nx))
    nup = int(max(nx, ny))
    kx, ky = get_wn(fld)
    k2d = np.hypot(kx, ky)

    def gaussian(k, sig):
        return np.exp(- k**2 / sig**2)

    def func2d(sig):
        sum1 = np.sum(gaussian(k2d, sig)**2 * np.cos(2*np.pi * kx/nup * hcorr))
        sum2 = np.sum(gaussian(k2d, sig)**2)
        return sum1/sum2 - np.exp(-1)

    ##solve for sig given hcorr so that func2d=0
    ##first deal with some edge cases
    if hcorr <= 2:  ##shorter than resolved scale, sig_out should be largest
        sig_out = nup
    elif hcorr >= nup/2:  ##longer than domain scale, sig_out should be smallest
        sig_out = 1.
    else:
        sig_out = np.abs(fsolve(func2d, 1)[0])
    # print(sig_out, func2d(sig_out))

    ##draw random phase from a white noise field
    ph = fft2(np.random.normal(0, 1, fld.shape))

    ##scaling factor to get the right amplitude
    norm2 = np.sum(gaussian(k2d, sig_out)**2) / (nx*ny)
    sf = amp / np.sqrt(norm2)

    return np.real(ifft2(sf * gaussian(k2d, sig_out) * ph))


def random_field_powerlaw(nx, ny, amp, pwrlaw):

    fld = np.zeros((ny, nx))
    kx, ky = get_wn(fld)
    k2d = np.hypot(kx, ky)

    k2d[np.where(k2d==0.0)] = 1e-10  ##avoid singularity, set wn-0 to small value

    ##draw random phase from a white noise field
    ph = fft2(np.random.normal(0, 1, fld.shape))

    ##assemble random field given amplitude from power spectrum, and random phase ph
    norm = 2 * np.pi * k2d
    amp_ = amp * np.sqrt(k2d**(pwrlaw+1) / norm)
    amp_[np.where(k2d==1e-10)] = 0.0 ##zero amp for wn-0

    return np.real(ifft2(amp_ * ph))


def random_displacement(grid, mask, amp, hcorr):
    ##prototype: generate a random wavenumber-1 displacement for the whole domain

    ##set the boundaries to be masked if they are not cyclic
    ##i.e. we don't want displacement across non-cyclic boundary
    if grid.cyclic_dim is not None:
        cyclic_dim = grid.cyclic_dim
    else:
        cyclic_dim = ''
    if 'x' not in cyclic_dim:
        mask[:, 0] = True
        mask[:, -1] = True
    if 'y' not in cyclic_dim:
        mask[0, :] = True
        mask[-1, :] = True

    ##distance to masked area
    dist = distance_transform_edt(1-mask.astype(float))
    dist /= np.max(dist)
    dist = gaussian_filter(dist, sigma=10)
    dist[mask] = 0

    ##the displacement vector field from a random streamfunction
    psi = random_field_gaussian(grid.nx, grid.ny, 1, hcorr)

    du, dv = -grady(psi, 1), gradx(psi, 1)
    norm = np.hypot(du, dv).max()
    du *= amp / norm
    dv *= amp / norm

    du *= dist  ##taper near mask
    dv *= dist

    return du, dv


def random_press_wind_perturb(grid, dt, prev_pres, prev_u, prev_v,
                              ampl_pres, ampl_wind, hscale, tscale,
                              pres_wind_relate=True, wlat=15.):
    ###generate a random perturbation for wind u,v and pressure

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

