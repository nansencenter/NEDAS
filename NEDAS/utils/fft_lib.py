import numpy as np

try:
    ###fft implementation using FFTW
    import pyfftw

    def fft2(f):
        """
        2D FFT implemented by pyFFTW
        """
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
        """
        Inverse 2D FFT implemented by pyFFTW
        """
        ##prepare fftw plan
        b = pyfftw.empty_aligned(fh.shape[:-1] + (fh.shape[-1]//2+1,), dtype='complex64')
        a = pyfftw.empty_aligned(fh.shape, dtype='float32')
        fft_obj = pyfftw.FFTW(b, a, axes=(-2, -1), direction='FFTW_BACKWARD')
        ##perform the ifft2
        f = fft_obj(fh[..., 0:fh.shape[1]//2+1])
        return f

except ImportError:
    print("Warning: pyFFTW not found in your environment, will use numpy.fft instead.", flush=True)

    def fft2(x):
        return np.fft.fft2(x)

    def ifft2(x):
        return np.real(np.fft.ifft2(x))

def fftwn(n):
    """
    Wavenumber sequence for FFT output in 1 dimension

    Input:
    - n: int
      The size of the dimension

    Output:
    - wn: np.array
      The sequence of wavenumber (0,1,2,...-2,-1) for this dimension
    """
    nup = int(np.ceil((n+1)/2))
    if n%2 == 0:
        wn = np.concatenate((np.arange(0, nup), np.arange(2-nup, 0)))
    else:
        wn = np.concatenate((np.arange(0, nup), np.arange(1-nup, 0)))
    return wn

def get_wn(fld):
    """
    Generate meshgrid wavenumber for the input field

    Input:
    - fld: np.array
      n-dimensional field, the last two dimensions are the horizontal directions (y, x)

    Return:
    - wnx, wny: np.array, same dimensions as fld
      The wavenumber in x, y directions, according to the dimension (nx or ny, whichever is larger)
    """
    ny, nx = fld.shape[-2:]

    wnx = np.zeros(fld.shape)
    wny = np.zeros(fld.shape)

    nup = int(max(nx, ny))

    for i in fftwn(nx):
        wnx[..., i] = i * nup / nx
    for j in fftwn(ny):
        wny[..., j, :] = j * nup / ny

    return wnx, wny

