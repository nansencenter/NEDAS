import numpy as np
from scipy.optimize import fsolve
from fft_lib import fft2, ifft2, get_wn

##draw a 2D random field given its power spectrum
##two types of power spectra available: gaussian, or fixed power law (slope)
def random_field_gaussian(nx, ny, amp, hcorr):
    ##Gaussian spectrum with parameter sigma determined by
    ##the decorrelation distance hcorr (number of grid points)

    fld = np.zeros((ny, nx))
    kx, ky = get_wn(fld)
    k2d = np.hypot(kx, ky)

    def gaussian(k, sig):
        return np.exp(-2.0 * k / sig**2)

    def func2d(sig):
        sum1 = np.sum(gaussian(k2d, sig)**2 * np.cos(2*np.pi * kx/nx * hcorr))
        sum2 = np.sum(gaussian(k2d, sig)**2)
        return sum1/sum2 - np.exp(-1)

    ##solve for sig given hcorr so that func2d=0
    sig_out = np.abs(fsolve(func2d, 1)[0])
    print(sig_out, func2d(sig_out))

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
    amp = np.sqrt(k2d**(pwrlaw+1) / norm)
    amp[np.where(k2d==1e-10)] = 0.0 ##zero amp for wn-0

    return np.real(ifft2(amp * ph))


###some more complicated random perturbation with physical constraints
def random_displace():
    ##TODO
    pass


###generate a random perturbation for wind u,v and pressure
def random_pres_wind_perturb(grid, dt,                  ##grid obj for the 2D domain; dt: time interval (hours)
                             prev_pres, prev_u, prev_v, ##pres, u, v, perturbation from previous t
                             ampl_pres, ampl_wind,      ##pres amplitude (Pa); wind amplitude (m/s) for each scale
                             scales_sig, scales_wgt,
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

    ##power spectrum given by sigma and weight for each scale
    ns = len(scales_wgt)
    pwr_spec = lambda k: np.sum([scales_wgt[s] * np.exp(-k**2/scales_sig[s]**2) for s in range(ns)], axis=0)
    ###TODO: normalize for each gauss

    ##TODO: find sigma from hradius (zeroin func2D in pseudo2d.f)

    pres = random_field(grid, pwr_spec)
    # pres = ampl_pres * (pres - np.mean(pres))/np.std(pres)

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
        u = random_field(grid, pwr_spec)
        v = random_field(grid, pwr_spec)

    u = ampl_wind * (u - np.mean(u))/np.std(u)
    v = ampl_wind * (v - np.mean(v))/np.std(v)

    ##apply temporal correlation
    if prev_pres is not None and prev_u is not None and prev_v is not None:
        pres = wt * prev_pres + np.sqrt(1 - wt**2) * pres
        u = wt * prev_u + np.sqrt(1 - wt**2) * u
        v = wt * prev_v + np.sqrt(1 - wt**2) * v

    return pres, u, v


def gradx(fld, dx):
    gradx_fld = np.zeros(fld.shape)
    gradx_fld[..., 1:] = (fld[..., 1:] - fld[..., :-1]) / dx
    return gradx_fld


def grady(fld, dy):
    grady_fld = np.zeros(fld.shape)
    grady_fld[..., 1:, :] = (fld[..., 1:, :] - fld[..., :-1, :]) / dy
    return grady_fld



