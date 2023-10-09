import numpy as np

from fft_lib import fft2, ifft2, fftwn, get_wn

##scale decomposition
def lowpass_resp(k2d, k1, k2):
    r = np.zeros(k2d.shape)
    r[np.where(k2d<k1)] = 1
    r[np.where(k2d>k2)] = 0

    ind = np.where(np.logical_and(k2d>=k1, k2d<=k2))

    ##linear
    # r[ind] = (np.log(k2) - np.log(k2d[ind]))/(np.log(k2) - np.log(k1))
    r[ind] = (k2 - k2d[ind])/(k2 - k1)
    ##cos-square
    # r[ind] = np.cos((k2d[ind] - k1)*(0.5*np.pi/(k2 - k1)))**2

    return r


def scale_response(k2d, scales, s):
    ns = len(scales)
    resp = np.full(k2d.shape, 1.0)  ##default all ones
    if ns > 1:
        if s == 0:
            resp = lowpass_resp(k2d, scales[s], scales[s+1])
        if s == ns-1:
            resp = 1 - lowpass_resp(k2d, scales[s-1], scales[s])
        if s > 0 and s < ns-1:
            resp = lowpass_resp(k2d, scales[s], scales[s+1]) - lowpass_resp(k2d, scales[s-1], scales[s])
    return resp


def get_scale_comp(fld, scales, s):
    ##compute scale component of fld on the grid
    ##given scales: list of center wavenumber k defining the scale bands
    ##       s: the index for scale components

    # xk = grid2spec(x)
    # xkout = xk.copy()
    # ns = len(scales)
    # if ns > 1:
    #     kx, ky = get_wn(x)
    #     k2d = np.sqrt(kx**2 + ky**2)
    #     xkout = xk * get_scale_resp(k2d, scales, s)
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


