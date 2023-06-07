###utility function for multiscale grid
import numpy as np
from numba import njit

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

def meshgrid(xi, yi):
    nx = len(xi)
    ny = len(yi)
    x = np.full((ny, nx), np.nan)
    y = np.full((ny, nx), np.nan)
    for i in range(nx):
        y[:, i] = yi
    for i in range(ny):
        x[i, :] = xi
    return x, y


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

def get_scale(x, kr, s):
    xk = grid2spec(x)
    xkout = xk.copy()
    ns = len(kr)
    if ns > 1:
        kx, ky = get_wn(x)
        Kh = np.sqrt(kx**2 + ky**2)
        xkout = xk * get_scale_resp(Kh, kr, s)
    return spec2grid(xkout)

