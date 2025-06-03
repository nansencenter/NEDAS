import os
import numpy as np
from functools import lru_cache
from scipy.optimize import fsolve
from scipy.ndimage import distance_transform_edt, gaussian_filter
from NEDAS.utils.conversion import ensure_list
from NEDAS.utils.spatial_operation import gradx, grady, warp
from NEDAS.utils.fft_lib import fft2, ifft2, get_wn

def random_perturb(grid, fields, prev_perturb, dt=1, n=0, seed=None, **kwargs):
    """
    Add random perturbation to the given 2D field

    Args:
        grid: Grid object describing the 2d domain
        fields: list of np.array shape[...,ny,nx]
        prev_perturb; list of np.array from previous perturbation data, None if unavailable
        dt: float, interval (hours) between time steps
        n: int, current time step index
        variable: str, or list of str
        type: str: 'gaussian', 'powerlaw', or 'displace'
        amplitude: float, (or list of floats, in multiscale approach)
        hcorr: float, or list of float, horizontal corr length (meters)
        tcorr: float, or list of float, time corr length (hours)
        **kwargs: other arguments
    """
    if seed is None:
        ##try to randomize using system entropy
        seed = int.from_bytes(os.urandom(4), 'little')
    else:
        assert isinstance(seed, int)
    np.random.seed(seed)

    perturb = {}
    perturb_type, other_opts, params = parse_perturb_opts(**kwargs)

    for vname,rec in params.items():
        fld = fields[vname]
        assert grid.x.shape == fld.shape[-2:], f"input fields[{vname}] dimension mismatch with grid"

        ns = rec['nscale']
        if perturb_type == 'displace':
            perturb[vname] = np.zeros((ns,2)+fld.shape[-2:])
        else:
            perturb[vname] = np.zeros((ns,)+fld.shape)

        if prev_perturb[vname] is not None and n==0:
            perturb[vname] = prev_perturb[vname]
            continue

        ##loop over scale s and generate perturbation
        for s in range(ns):
            ##draw a random field for each 2d field component in fields
            for ind in np.ndindex(fld.shape[:-2]):
                if perturb_type == 'gaussian':
                    perturb[vname][(s,)+ind] = random_field_gaussian(grid.nx, grid.ny, rec['amp'][s], rec['hcorr'][s]/grid.dx)

                elif perturb_type == 'powerlaw':
                    perturb[vname][(s,)+ind] = random_field_powerlaw(grid.nx, grid.ny, rec['amp'][s], rec['powerlaw'][s])

                elif perturb_type == 'displace':
                    mask = np.full(grid.x.shape, False)
                    du, dv = random_displacement(grid, mask, rec['amp'][s], rec['hcorr'][s]/grid.dx)
                    perturb[vname][s] = np.array([du, dv])

                else:
                    raise NotImplementedError('unknown perturbation type: '+perturb_type)

            ##create perturbations that are correlated in time
            autocorr = 0.75
            ncorr = rec['tcorr'][s] / dt  ##time steps at decorrelation
            alpha = autocorr**(1.0 / ncorr)
            if prev_perturb[vname] is not None:
                perturb[vname][s] = np.sqrt(1-alpha**2) * perturb[vname][s] + alpha * prev_perturb[vname][s]

    ###legacy prsflg==1,2 options in force_perturb program, reproduced here
    if 'press_wind_relate' in other_opts:
        for vname in ['atmos_surf_velocity', 'atmos_surf_press']:
            assert vname in params.keys(), f'{vname} not in variable list, cannot run press_wind_relate option'

        for s in range(ns):
            perturb['atmos_surf_velocity'][s] = get_velocity_from_press(grid, perturb['atmos_surf_press'][s], ('scale_wind' in other_opts), params['atmos_surf_press']['amp'][s], params['atmos_surf_press']['hcorr'][s], params['atmos_surf_velocity']['amp'][s])

    ##now add perturbations to each field
    for vname,rec in params.items():
        for s in range(rec['nscale']):
            if perturb_type == 'displace':
                fields[vname] = warp(grid, fields[vname], perturb[vname][s,0,...], perturb[vname][s,1,...])

            else:
                if 'exp' in other_opts:
                    ##add lognormal perturbations
                    fields[vname] *= np.exp(perturb[vname][s,...] - 0.5*rec['amp'][s]**4)

                else:
                    ##just add the gaussian perturbations
                    fields[vname] += perturb[vname][s,...]

        ##respect value bounds after perturbing
        if 'bounds' in kwargs:
            vmin, vmax = kwargs['bounds']
            fields[vname] = np.minimum(np.maximum(fields[vname], vmin), vmax)

    return fields, perturb

def parse_perturb_opts(**kwargs):
    ##perturb['type'] string format:
    #main option (gaussian/powerlaw/displace) followed by , then additional options separated by ,
    opts = kwargs['type'].split(',')
    perturb_type = opts[0]
    other_opts = []
    for opt in opts[1:]:
        other_opts.append(opt)

    key_list = []
    for key in ['amp', 'hcorr', 'tcorr', 'powerlaw']:
        if key in kwargs:
            key_list.append(key)

    ##a list of variables can be specified if running a multivariate perturbation scheme
    ##form a variable list for further processing
    variable_list = ensure_list(kwargs['variable'])
    nv = len(variable_list)
    for key in key_list:
        kwargs[key] = ensure_list(kwargs[key])

    ##get perturbation parameters for each variable from kwargs
    params = {}
    for v in range(nv):
        vname = variable_list[v]
        params[vname] = {}
        ##in multiscale approach, a list of parameters can be specified for a variable;
        ##one separate perturbation will be generated for each, then they will be added together
        if isinstance(kwargs[key_list[0]][v], list):
            nscale = len(kwargs[key_list[0]][v])
        else:
            nscale = 1
            for key in key_list:  ##make a list even if only one value for the key
                kwargs[key][v] = [kwargs[key][v]]
        params[vname]['nscale'] = nscale
        ##check if all keys are lists with same len
        for key in key_list[1:]:
            assert len(kwargs[key][v]) == nscale, f"perturb option: {key} has different number of entries from {key_list[0]}, check config"
        ##assign the parameters
        for key in key_list:
            params[vname][key] = kwargs[key][v]

    return perturb_type, other_opts, params

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
    ##hcorr = rh * dx according to mid-domain dx (to be consistent with ReanalysisTP5/Perturb_forcing)
    dx_ = grid.dx / grid.mfx[grid.ny//2, grid.nx//2]

    wprsfac = 1.
    if scale_wind:
        ds = 0.54 * press_hcorr / dx_ * np.hypot(dx, dy)  ##horizontal scale
        wind_scale = press_amp / ds / fcor / rhoa  ##expected wind scale from pressure field
        wprsfac = wind_amp / wind_scale  ##scaling factor to match wind perturbation with given amp

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
    dist /= np.max(dist)
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

