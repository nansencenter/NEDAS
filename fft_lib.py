import numpy as np

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

    nup = f.shape[-1]//2+1

    ##top half of the spectrum
    fh[..., 0:nup] = fh_

    ##the bottom half is conj of top half of the spectrum
    if f.shape[-1]%2 == 0:
        fh[..., 0, nup:] = np.conj(fh_[..., 0, nup-2:0:-1])
        fh[..., 1:, nup:] = np.conj(fh_[..., :0:-1, nup-2:0:-1])
    else:
        fh[..., 0, nup:] = np.conj(fh_[..., 0, nup-1:0:-1])
        fh[..., 1:, nup:] = np.conj(fh_[..., :0:-1, nup-1:0:-1])

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


##or, if fftw is not available, use np.fft instead
##from np.fft import fft2, ifft2


def get_wn(fld):
    ##generate meshgrid wavenumber for input field
    ## the last two dimensions are horizontal (y, x)
    ny, nx = fld.shape[-2:]

    wnx = np.zeros(fld.shape)
    wny = np.zeros(fld.shape)

    nup = int(max(nx, ny))

    for i in fftwn(nx):
        wnx[..., i] = i * nup / nx
    for j in fftwn(ny):
        wny[..., j, :] = j * nup / ny

    return wnx, wny

