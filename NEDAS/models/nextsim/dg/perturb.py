import numpy as np
import typing
from scipy.optimize import fsolve, root_scalar # type: ignore
from scipy.ndimage import distance_transform_edt, gaussian_filter # type: ignore
from NEDAS.utils.spatial_operation import gradx, grady, warp
from NEDAS.utils.fft_lib import fft2, ifft2, get_wn, fftwn
from NEDAS.grid import Grid

def gen_perturb(grid: Grid,
                perturb_type: typing.Literal['gaussian'],
                amp: float,
                hcorr: int) -> np.ndarray:
    """
    get random perturbation fields

    Parameters
    ----------
    grid : Grid
        grid object for the 2D domain
    perturb_type : typing.Literal['gaussian']
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
        perturb = perturb - perturb.mean()
    else:
        raise TypeError('unknown perturbation type: '+perturb_type)

    return perturb


def apply_AR1_perturb(perturb: np.ndarray,
                      tcorr:float=1.0,
                      prev_perturb:typing.Union[None, np.ndarray]=None) -> np.ndarray:
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


def apply_perturb(grid: Grid,
                  field:np.ndarray,
                  perturb:np.ndarray,
                  perturb_type:typing.Literal['gaussian']) -> np.ndarray:
    """apply perturbation to the field

    Parameters
    ----------
    grid : Grid
        grid object for the 2D domain
    field : np.ndarray
        field to be perturbed
    perturb : np.ndarray
        perturbation field
    perturb_type : typing.Literal['gaussian']
        type of perturbation method

    Returns
    -------
    np.ndarray
        perturbed field
    """
    ##apply the perturbation
    # todo: adding logorithmic perturbation
    field += perturb
    return field


def random_field_gaussian(nx:int, ny:int, amplitude:float, hcorr:int) -> np.ndarray:
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
    ph:np.ndarray = amplitude* c* gauss * val
    # the nx*ny is the normalisation factor because fftw normalise the inverse fft by default
    return np.real(ifft2(ph * nx * ny))


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

    # ratio between the wind speed amplitude and a typical wind speed from pressure gradient
    wprsfac:float =1.
    if with_wind_speed_limit:
        # typical pressure gradient
        wprsfac=np.sqrt(ampl_pres)/scorr/fcor
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
