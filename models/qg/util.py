from fft_lib import fft2, ifft2, get_wn


##scaled wavenumber k for pseudospectral method
def get_scaled_wn(fld, dx):
    n = fld.shape[0]
    wni, wnj = get_wn(fld)
    ki = (2.*np.pi) * wni / (n*dx)
    kj = (2.*np.pi) * wnj / (n*dx)
    return ki, kj


def spec2grid():
    pass


def grid2spec():
    pass


