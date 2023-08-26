import numpy as np

###utility function for multiscale grid

###fft implementation using FFTW
import pyfftw

def fft2(f):
    ##prepare fftw plan
    a = pyfftw.empty_aligned(f.shape, dtype='float32')
    b = pyfftw.empty_aligned(f.shape[:-1] + (f.shape[-1]//2+1,), dtype='complex64')
    fft_obj = pyfftw.FFTW(a, b, axes=(-2, -1))
    ##perform the fft2, output fh with same shape as f (keep dimension info)
    fh_ = fft_obj(f)
    fh = np.zeros(f.shape, dtype='complex64')
    fh[..., 0:f.shape[1]//2+1] = fh_
    ##TODO: fill the zero half with conj, so that pwrspec2d will sum correctly
    return fh


def ifft2(fh):
    ##prepare fftw plan
    b = pyfftw.empty_aligned(fh.shape[:-1] + (fh.shape[-1]//2+1,), dtype='complex64')
    a = pyfftw.empty_aligned(fh.shape, dtype='float32')
    fft_obj = pyfftw.FFTW(b, a, axes=(-2, -1), direction='FFTW_BACKWARD')
    ##perform the ifft2
    f = fft_obj(fh[..., 0:fh.shape[1]//2+1])
    return f


def fftwn(n):
    ##wavenumber sequence for fft results in 1 dimension
    nup = int(np.ceil((n+1)/2))
    if n%2 == 0:
        wn = np.concatenate((np.arange(0, nup), np.arange(2-nup, 0)))
    else:
        wn = np.concatenate((np.arange(0, nup), np.arange(1-nup, 0)))
    return wn


def get_wn(fld):
    ##generate meshgrid wavenumber for input field
    ## the last two dimensions are horizontal (y, x)
    ny, nx = fld.shape[-2:]
    wnx = np.zeros(fld.shape)
    wny = np.zeros(fld.shape)
    for i in fftwn(fld.shape[-1]):
        wnx[..., i] = i
    for j in fftwn(fld.shape[-2]):
        wny[..., j, :] = j
    return wnx, wny

##scale decomposition
def lowpass_resp(Kh, k1, k2):
    r = np.zeros(Kh.shape)
    r[np.where(Kh<k1)] = 1.0
    r[np.where(Kh>k2)] = 0.0
    ind = np.where(np.logical_and(Kh>=k1, Kh<=k2))
    r[ind] = np.cos((Kh[ind] - k1)*(0.5*np.pi/(k2 - k1)))**2
    return r


def get_scale_resp(Kh, kr, s):
    ns = len(kr)
    resp = np.zeros(Kh.shape)
    if ns > 1:
        if s == 0:
            resp = lowpass_resp(Kh, kr[s], kr[s+1])
        if s == ns-1:
            resp = 1 - lowpass_resp(Kh, kr[s-1], kr[s])
        if s > 0 and s < ns-1:
            resp = lowpass_resp(Kh, kr[s], kr[s+1]) - lowpass_resp(Kh, kr[s-1], kr[s])
    return resp


def get_scale_comp(fld, krange, s):
    ##compute scale component of fld on the grid
    ##given kr: list of center wavenumber k defining the scale bands
    ##       s: the index for scale components

    # xk = grid2spec(x)
    # xkout = xk.copy()
    # ns = len(kr)
    # if ns > 1:
    #     kx, ky = get_wn(x)
    #     Kh = np.sqrt(kx**2 + ky**2)
    #     xkout = xk * get_scale_resp(Kh, kr, s)
    # return spec2grid(xkout)
    pass


def convolve_fft(grid, fld, kernel):
    assert grid.regular, 'Convolution using FFT approach only works for regular grids'
    fld_spec = fft2(fld)
    knl_spec = fft2(kernel)
    mask = np.isnan(fld)
    fld[mask] = 0.  ##temporarily put zeros in NaN area

    fld_conv_spec = fld_spec * knl_spec
    fld_conv = ifft2(fld_conv_spec)

    fld_conv[mask] = np.nan

    return fld_conv


def convolve(grid, fld, kernel):
    fld_shp = fld.shape
    assert fld_shp == grid.x.shape, 'fld shape mismatch with grid'

    x = grid.x.flatten()
    y = grid.y.flatten()
    fld = fld.flatten()
    mask = np.isnan(fld)
    fld_conv = fld.copy()

    for i in np.where(mask):
        fld_conv[i]

    return fld_conv
