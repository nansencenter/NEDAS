import numpy as np

try:
    ###fft implementation using FFTW
    import pyfftw

    def fft2(f: np.ndarray) -> np.ndarray:
        """
        2D FFT implemented by pyFFTW. If pyFFTW is not available, will switch to np.fft.fft2

        Args:
            f (np.ndarray):
                Input field, of shape (..., ny, nx), of float32 type,
                the last two dimensions will be transformed by the 2D FFT.

        Returns:
            np.ndarray: Output field in spectral space, of complex64 type.
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

    def ifft2(fh: np.ndarray) -> np.ndarray:
        """
        Inverse 2D FFT implemented by pyFFTW. If pyFFTW is not available, will switch to np.fft.ifft2

        Args:
            fh (np.ndarray):
                Input field, of shape (..., nky, nkx), of complex64 type,
                the last two dimensions will be inverse transformed by 2D FFT.

        Returns:
            np.ndarray: Output field in physical space, of float32 type.
        """
        ##prepare fftw plan
        b = pyfftw.empty_aligned(fh.shape[:-1] + (fh.shape[-1]//2+1,), dtype='complex64')
        a = pyfftw.empty_aligned(fh.shape, dtype='float32')
        fft_obj = pyfftw.FFTW(b, a, axes=(-2, -1), direction='FFTW_BACKWARD')
        ##perform the ifft2
        ##TODO: check index
        ##TODO: normalize?
        f = fft_obj(fh[..., 0:fh.shape[1]//2+1])
        return f

except ImportError:
    #print("Warning: pyFFTW not found in your environment, will use numpy.fft instead.", flush=True)

    def fft2(x):
        return np.fft.fft2(x)

    def ifft2(x):
        return np.real(np.fft.ifft2(x))

def fftwn(n):
    """
    Wavenumber sequence corresponding to the FFT output in 1 dimension.

    Args:
        n (int): The size of the dimension

    Returns:
        np.ndarray: The sequence of wavenumber (0,1,2,...-2,-1) for this dimension
    """
    nup = int(np.ceil((n+1)/2))
    if n%2 == 0:
        wn = np.concatenate((np.arange(0, nup), np.arange(2-nup, 0)))
    else:
        wn = np.concatenate((np.arange(0, nup), np.arange(1-nup, 0)))
    return wn

def get_wn(fld):
    """
    Generates a meshgrid of wavenumbers corresponding to the input field dimensions.

    Args:
        fld (np.ndarray):
            Input field, the last two dimensions are the horizontal directions (ny, nx)

    Returns:
        np.ndarray:
            Wavenumbers corresponding to the input fields dimensions,
            relative to the domain size (nx or ny, whichever is larger).
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

