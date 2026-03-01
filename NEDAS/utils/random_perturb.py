import numpy as np
from functools import lru_cache
from scipy.optimize import fsolve
from scipy.ndimage import distance_transform_edt, gaussian_filter
from NEDAS.utils.spatial_operation import gradx, grady
from NEDAS.utils.fft_lib import fft2, ifft2, get_wn

def random_field_gaussian(nx, ny, amp, hcorr):
    """
    Random field with a Gaussian spectrum.

    Args:
        nx (int): Number of grid points in x direction.
        ny (int): Number of grid points in y direction.
        amp (float): Amplitude (standard deviation) of the random field.
        hcorr (float): Horizontal decorrelation length (number of grid points).

    Returns:
        np.ndarray: The output random field with shape (ny, nx).
    """
    fld = np.zeros((ny, nx))
    kx, ky = get_wn(fld)
    k2d = np.hypot(kx, ky)
    sig_out = get_sig_in_gaussian(nx, ny, hcorr)

    ##draw random phase from a white noise field
    ph = fft2(np.random.normal(0, 1, fld.shape))

    ##scaling factor to get the right amplitude
    norm2 = np.sum(gaussian(k2d, sig_out)**2) / (nx*ny)
    sf = amp / np.sqrt(norm2)
    return np.real(ifft2(sf * gaussian(k2d, sig_out) * ph))

def gaussian(k, sig):
    """gaussian spectrum"""
    return np.exp(- k**2 / sig**2)

@lru_cache
def get_sig_in_gaussian(nx, ny, hcorr):
    """
    Derive the sig parameter in gaussian spectrum.
    Called by random_field_gaussian, cached the result since in external for loop there will be fixed input
    """
    fld = np.zeros((ny, nx))
    nup = int(max(nx, ny))
    kx, ky = get_wn(fld)
    k2d = np.hypot(kx, ky)

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
    return sig_out

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
    #dist /= dist.max()
    dist = gaussian_filter(dist, sigma=10)
    dist[mask] = 0

    ##the displacement vector field from a random streamfunction
    psi = random_field_gaussian(grid.nx, grid.ny, 1, hcorr)

    du, dv = -grady(psi, 1, grid.cyclic_dim), gradx(psi, 1, grid.cyclic_dim)
    norm = np.hypot(du, dv).max()
    du *= amp / norm
    dv *= amp / norm

    du *= dist  ##taper near mask
    dv *= dist
    return du, dv

def get_velocity_from_press(grid, pres, scale_wind=False, press_amp=None, press_hcorr=None, wind_amp=None):
    """derive wind velocity from pressure field """
    rhoa = 1.2
    r2d = 180/np.pi
    wlat = 15.
    plon, plat = grid.proj(grid.x, grid.y, inverse=True)
    fcor = 2 * np.sin(40./r2d)* 2*np.pi / 86400  ##coriolis at 40N

    ##grid spacing
    dx = grid.dx / grid.mfx
    dy = grid.dy / grid.mfy
    ##mid-domain dx (to be consistent with ReanalysisTP5/Perturb_forcing)
    dx_ = grid.dx / grid.mfx[grid.ny//2, grid.nx//2]

    wprsfac = 1.
    if scale_wind:
        ds = (press_hcorr / dx_) * np.hypot(dx, dy) * 0.96   ##horizontal scale rh * dx, 0.96 is a tuning factor
        wind_scale = press_amp / ds / fcor  ##expected wind scale from pressure field
        wprsfac = wind_amp / wind_scale     ##scaling factor to match wind perturbation with given amp

    ##pres gradients
    dpresx = gradx(pres, dx, grid.cyclic_dim) * wprsfac
    dpresy = grady(pres, dy, grid.cyclic_dim) * wprsfac

    ##geostrophic wind near poles
    vcor =  dpresx / fcor / rhoa * np.sign(plat)
    ucor = -dpresy / fcor / rhoa * np.sign(plat)

    ##gradient wind near equator
    ueq = -dpresx / fcor / rhoa
    veq = -dpresy / fcor / rhoa

    wcor = np.sin(np.minimum(wlat, plat) / wlat * np.pi * 0.5)
    u = wcor*ucor + (1-wcor)*ueq
    v = wcor*vcor + (1-wcor)*veq

    return np.array([u, v])
