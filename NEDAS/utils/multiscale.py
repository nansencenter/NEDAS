import numpy as np
from NEDAS.grid import Grid
from NEDAS.utils.fft_lib import fft2, ifft2, get_wn

"""functions to perform scale decomposition, through bandpass filtering
on spatial data (fields), can also use gcm_filter to achieve this.
"""
def lowpass_response(k2d, k1, k2):
    """
    Low-pass spectral response function
    Input
    - k2d: array, 2d wavenumber
    - k1, k2: float, wavenumbers defining the transition zone (from 1 to 0 in response)
    Return
    - r: array same dimension as k2d, the response function
    """
    r = np.zeros(k2d.shape)
    r[np.where(k2d<k1)] = 1.0
    r[np.where(k2d>k2)] = 0.0
    ind = np.where(np.logical_and(k2d>=k1, k2d<=k2))
    ##cos-square transition
    r[ind] = np.cos((k2d[ind] - k1)*(0.5*np.pi/(k2 - k1)))**2
    return r

def get_scale_component_spec_bandpass(grid, fld, character_length, s):
    assert grid.regular, "get_scale_component_spec_bandpass only works for regular grid"

    ##convert length to wavenumber
    L = max(grid.Lx, grid.Ly)
    character_k = L / np.array(character_length)

    nscale = len(character_k)
    if nscale == 1:
        return fld ##nothing to be done, return the original field

    kx, ky = get_wn(fld)
    k2d = np.hypot(kx, ky)
    fld_spec = fft2(fld)
    if s == 0:
        r = lowpass_response(k2d, character_k[s], character_k[s+1])
    if s == nscale-1:
        r = 1 - lowpass_response(k2d, character_k[s-1], character_k[s])
    if s > 0 and s < nscale-1:
        r = lowpass_response(k2d, character_k[s], character_k[s+1]) - lowpass_response(k2d, character_k[s-1], character_k[s])

    return ifft2(fld_spec * r)

def convolve(grid, fld, rgrid, response):
    L = max(grid.Lx, grid.Ly)
    kx, ky = get_wn(rgrid.x)
    ny, nx = rgrid.x.shape
    fld1 = fld.copy()
    ##go through grid points in the field and perform convolution
    for i in np.ndindex(grid.x.shape):
        ##shift to the grid point i position
        r = response * np.exp(-2 * np.pi * complex(0,1) * (kx*grid.x[i] + ky*grid.y[i]) / L)
        ##convert to physical space
        kernel = ifft2(r) * nx * ny / grid.x.size
        w = rgrid.convert(kernel)
        ##perform convolution
        fld1[i] = np.nansum(fld * w)
    return fld1

def get_scale_component_convolve(grid, fld, character_length, s):
    ##convert length to wavenumber
    L = max(grid.Lx, grid.Ly)
    character_k = L / np.array(character_length)

    nscale = len(character_k)
    if nscale == 1:
        ##nothing to be done, return the original field
        return fld

    ##make a regular grid for the kernel function
    rgrid = Grid.regular_grid(grid.proj, grid.xmin, grid.xmax, grid.ymin, grid.ymax, grid.dx)
    rgrid.set_destination_grid(grid)
    kx, ky = get_wn(rgrid.x)
    ny, nx = rgrid.x.shape
    k2d = np.hypot(kx, ky)

    if s == 0:
        r = lowpass_response(k2d, character_k[s], character_k[s+1])
        return convolve(grid, fld, rgrid, r)
    if s == nscale-1:
        r = lowpass_response(k2d, character_k[s-1], character_k[s])
        return fld - convolve(grid, fld, rgrid, r)
    if s > 0 and s < nscale-1:
        r1 = lowpass_response(k2d, character_k[s-1], character_k[s])
        r2 = lowpass_response(k2d, character_k[s], character_k[s+1])
        return convolve(grid, fld, rgrid, r2) - convolve(grid, fld, rgrid, r1)

def get_scale_component(grid, fld, character_length, s):
    """
    Get scale component using a bandpass filter in spectral space
    Input:
    - grid: Grid object
    - fld: array, [..., ny, nx], the input field
    - character_length: list of characteristic length for each scale
    - s: int, scale index
    Return:
    - fld: array, the scale component s of input fld
    """
    flds = fld.copy()
    if grid.regular:
        for i in np.ndindex(fld.shape[:-2]):
            flds[i] = get_scale_component_spec_bandpass(grid, fld[i], character_length, s)
    else:
        for i in np.ndindex(fld.shape[:-1]):
            flds[i] = get_scale_component_convolve(grid, fld[i], character_length, s)
    return flds

def get_error_scale_factor(grid, character_length, s):
    err_scale_fac = np.ones(grid.x.shape)
    L = max(grid.Lx, grid.Ly)
    character_k = L / np.array(character_length)
    nscale = len(character_k)
    if nscale == 1:
        return err_scale_fac

    rgrid = Grid.regular_grid(grid.proj, grid.xmin, grid.xmax, grid.ymin, grid.ymax, grid.dx)
    rgrid.set_destination_grid(grid)
    kx, ky = get_wn(rgrid.x)
    ny, nx = rgrid.x.shape
    k2d = np.hypot(kx, ky)
    if s == 0:
        response = lowpass_response(k2d, character_k[s], character_k[s+1])
    if s == nscale-1:
        response = 1 - lowpass_response(k2d, character_k[s-1], character_k[s])
    if s > 0 and s < nscale-1:
        r1 = lowpass_response(k2d, character_k[s-1], character_k[s])
        r2 = lowpass_response(k2d, character_k[s], character_k[s+1])
        response = r2 - r1

    for i in np.ndindex(grid.x.shape):
        ##shift to the grid point i position
        r = response * np.exp(-2 * np.pi * complex(0,1) * (kx*grid.x[i] + ky*grid.y[i]) / L)
        ##convert to physical space
        kernel = ifft2(r) * nx * ny / grid.x.size
        w = rgrid.convert(kernel)
        err_scale_fac[i] = w[i]
    return err_scale_fac

