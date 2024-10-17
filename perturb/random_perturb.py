import numpy as np
import typing
from scipy.optimize import fsolve, root_scalar # type: ignore
from scipy.ndimage import distance_transform_edt, gaussian_filter # type: ignore
from utils.spatial_operation import gradx, grady, warp
from utils.fft_lib import fft2, ifft2, get_wn, fftwn
from grid.grid import Grid

##top-level function
def random_perturb(field, grid, perturb_type, amp, hcorr, tcorr=1.0, powerlaw=-3, prev_perturb=None):
    """
    Add random perturbation to the given 2D field
    Input:
        - field: np.array
        - grid: Grid object for the 2d field
        - perturb_type: 'gaussian', 'powerlaw', 'empirical', or 'displace'
        - amp: float
        - hcorr: float
        - tcorr: float (optional)
        - perturb_prev; dict(str, np.array), previous perturbation data
    """
    ##generate perturbation according to prescribed parameters
    if perturb_type == 'gaussian':
        perturb = random_field_gaussian(grid.nx, grid.ny, amp, hcorr)

    elif perturb_type == 'powerlaw':
        perturb = random_field_powerlaw(grid.nx, grid.ny, amp, powerlaw)

    elif perturb_type == 'empirical':
        ##TODO: allow user to specify empirical spectrum for the perturbation
        raise NotImplementedError

    elif perturb_type == 'displace':
        mask = np.full(grid.x.shape, False)
        du, dv = random_displacement(grid, mask, amp, hcorr)
        perturb = np.array([du, dv])

    else:
        raise TypeError('unknown perturbation type: '+perturb_type)

    ##create perturbations that are correlated in time
    autocorr = np.exp(-1.0)
    alpha = autocorr**(1.0/tcorr)
    if prev_perturb is not None:
        perturb = np.sqrt(1-alpha**2) * perturb + alpha * prev_perturb

    ##apply the perturbation
    if perturb_type == 'displace':
        field = warp(field, perturb[0,...], perturb[1,...])
    else:
        field += perturb

    return field, perturb

def gen_perturb(grid: Grid,
                   perturb_type: typing.Literal['gaussian', 'powerlaw', 'displace', 'gaussian_evensen'],
                   amp: float,
                   hcorr : int,
                   powerlaw:float=-3.) -> np.ndarray:
    """
    get random perturbation fields

    Parameters
    ----------
    grid : Grid
        grid object for the 2D domain
    perturb_type : typing.Literal['gaussian', 'powerlaw', 'displace', 'gaussian_evensen']
        type of perturbation method
    amp : float
        amplitude of the perturbation
    hcorr : float
        horizontal decorrelation length scale in grid points
    powerlaw : float, optional
        power law exponent, by default -3.

    Returns
    -------
    np.ndarray
        random perturbation field
    """
    ##generate perturbation according to prescribed parameters
    if perturb_type == 'gaussian':
        perturb = random_field_gaussian(grid.nx, grid.ny, amp, hcorr)

    elif perturb_type == 'powerlaw':
        perturb = random_field_powerlaw(grid.nx, grid.ny, amp, powerlaw)

    elif perturb_type == 'empirical':
        ##TODO: allow user to specify empirical spectrum for the perturbation
        raise NotImplementedError

    elif perturb_type == 'displace':
        mask = np.full(grid.x.shape, False)
        du, dv = random_displacement(grid, mask, amp, hcorr)
        perturb = np.array([du, dv])
    elif perturb_type == 'gaussian_evensen':
        perturb = random_field_gaussian_evensen(grid.nx, grid.ny, amp, hcorr)
    else:
        raise TypeError('unknown perturbation type: '+perturb_type)

    return perturb


def apply_AR1_perturb(perturb: np.ndarray, tcorr:float=1.0, prev_perturb:typing.Union[None, np.ndarray]=None) -> np.ndarray:
    """apply AR1 perturbation to the field

    Parameters
    ----------
    perturb : np.ndarray
        perturbation field
    tcorr : float, optional
        time correlation length, by default 1.0
    prev_perturb : typing.Union[None, np.ndarray], optional
        previous perturbation field, by default None
    """
    ##create perturbations that are correlated in time
    autocorr = np.exp(-1.0)
    alpha = autocorr**(1.0/tcorr)
    if prev_perturb is not None:
        perturb = np.sqrt(1-alpha**2) * perturb + alpha * prev_perturb
    return perturb


def apply_perturb(grid: Grid, field:np.ndarray, perturb:np.ndarray, perturb_type:typing.Literal['gaussian', 'powerlaw', 'displace', 'gaussian_evensen']) -> np.ndarray:
    """apply perturbation to the field

    Parameters
    ----------
    grid : Grid
        grid object for the 2D domain
    field : np.ndarray
        field to be perturbed
    perturb : np.ndarray
        perturbation field
    perturb_type : typing.Literal['gaussian', 'powerlaw', 'displace', 'gaussian_evensen']
        type of perturbation method

    Returns
    -------
    np.ndarray
        perturbed field
    """
    ##apply the perturbation
    # todo: adding logorithmic perturbation
    if perturb_type == 'displace':
        field = warp(grid, field, perturb[0], perturb[1])
    else:
        field += perturb
    return field


def random_field_gaussian_evensen(nx:int, ny:int, amplitude:float, hcorr:int) -> np.ndarray:
    """generate a 2D random field with Gaussian correlation length scale with given amplitude.

    Parameters
    ----------
    nx : int
        number of grid points in x direction
    ny : int
        number of grid points in y direction
    amplitude : float
        amplitude of the random field
    hcorr : int
        horizontal correlation length scale in grid points

    Returns
    -------
    np.ndarray
        2D random field with Gaussian correlation length scale
    """
    assert hcorr < nx, f'hcorr {hcorr} must be smaller than nx {nx}'
    assert hcorr > 0, f'hcorr {hcorr} must be larger than 0'
    # generate a 2D field with given spatial resolution
    fld: np.ndarray = np.zeros((ny, nx))
    # generate the wave frequency
    kappa:float = 2*np.pi/(nx)
    lamda:float = 2*np.pi/(ny)
    # squared wave frequency
    kappa2:float = kappa*kappa
    lambda2:float = lamda*lamda
    # get the wave numbers
    n1:np.ndarray = fftwn(nx)
    n2:np.ndarray = fftwn(ny)
    l: np.ndarray
    p: np.ndarray
    l, p = np.meshgrid(n1, n2)

    def func2d(sigma:float) -> float:
      """solve the equation for sigma given horizontal correlations.

      This function is solved to obtain the sigma for unit variance of the random field.
      """
      sum1:float = np.sum(np.exp(-2.0*(kappa2*l*l+lambda2*p*p)/sigma/sigma)*np.cos(kappa*l*hcorr))
      sum2:float = np.sum(np.exp(-2.0*(kappa2*l*l+lambda2*p*p)/sigma/sigma))
      return sum1/sum2 - np.exp(-1)

    # solve for sigma given hcorr so that func2d=0
    sigma:float = root_scalar(func2d, bracket=(1e-6, 2*np.pi), method='bisect', xtol=1e-10).root

    # compute the scaling factor c for getting the unit variance
    sum2:float = np.sum(np.exp(-2.*(kappa2*l*l+lambda2*p*p)/sigma/sigma))
    c:float = 1./np.sqrt(sum2)

    # getting the random phase shift in Fourier space
    val:np.ndarray = np.random.uniform(0, 1, fld.shape)
    val = np.exp(2*np.pi*val*1j)

    # getting the Fourier space representation of the field
    gauss:np.ndarray = np.exp(-(kappa2*l*l+lambda2*p*p)/sigma/sigma)
    ph:np.ndarray = c* gauss * val
    # the nx*ny is the normalisation factor because fftw normalise the inverse fft by default
    return np.real(ifft2(ph * nx * ny))


##draw a 2D random field given its power spectrum
##two types of power spectra available: gaussian, or fixed power law (slope)
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


def random_pres_wind_perturb(grid, dt, prev_pres, prev_u, prev_v,
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


def pres_adjusted_wind_perturb(grid: Grid, ampl_pres:float, ampl_wind:float,
                               scorr:float, pres:np.ndarray,
                               with_wind_speed_limit:bool=True, wlat:float=15.) -> tuple[np.ndarray, np.ndarray]:
    """generate a random perturbation for wind u,v by pressure perturbation.

    Parameters
    ----------
    grid : Grid
        grid object for the 2D domain
    ampl_pres : float
        amplitude of pressure perturbations (Pa)
    ampl_wind : float
        amplitude of wind perturbations (m/s)
    scorr : float
        horizontal decorrelation length scale by number of grid points
    pres : np.ndarray
        pressure field perturbations
    with_wind_speed_limit : bool, optional
        if true: limit pressure perturbation by wind speed to account for horizontal scale of perturbation, by default True.
        This option will force `pres_wind_relate` to be True.
    wlat : float, optional
        latitude bound for pure geostroph wind, by default 15.

    Returns
    -------
    tuple
        pressure perturbation, u wind perturbation, v wind perturbation
    """

    # get latitude from grid
    plat: np.ndarray
    _, plat = grid.proj(grid.x, grid.y, inverse=True)
    # convert radians to degrees
    r2d:float = 180/np.pi
    # air density
    rhoa:float = 1.2
    # reference latitude where we use the coriolis parameter
    rlat:float = 40.
    # coriolis parameter where we use 40 degrees as the latitude
    # the sign function is used to distinguish the coriolis parameter for different hemisphere
    fcor:np.ndarray = np.sign(plat)*(2*np.sin(rlat/r2d)*2*np.pi/86400)

    # minimum spatial resolution
    dx_min: float = grid.dx.min()
    # ratio between the wind speed amplitude and a typical wind speed from pressure gradient
    wprsfac:float =1.
    if with_wind_speed_limit:
        # typical pressure gradient
        wprsfac=np.sqrt(ampl_pres)/(scorr*dx_min)
        wprsfac=wprsfac/fcor.max()
        wprsfac=np.sqrt(ampl_wind)/wprsfac

    # calc u,v from pres according to pres-wind relation
    # pres gradients
    dpresx:np.ndarray = gradx(pres, grid.dx)*wprsfac
    dpresy:np.ndarray = grady(pres, grid.dx)*wprsfac

    # geostrophic wind near poles
    vcor: np.ndarray =  dpresx / fcor / rhoa
    ucor: np.ndarray = -dpresy / fcor / rhoa

    # wind near equator is proportional to pressure gradient
    # coriolis parameter is just used to limit the equator wind perturbations
    ueq: np.ndarray = -dpresx / np.abs(fcor) / rhoa
    veq: np.ndarray = -dpresy / np.abs(fcor) / rhoa

    # weight for geostrophic wind
    # the weight is 1 when the latitude is outside of the band of wlat
    # the weight is 0 when the latitude is within the band of wlat
    wcor:np.ndarray = np.sin(np.minimum(wlat, np.abs(plat)) / wlat * np.pi * 0.5)
    u = wcor*ucor + (1-wcor)*ueq
    v = wcor*vcor + (1-wcor)*veq

    return u, v