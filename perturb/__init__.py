import numpy as np
from grid.multiscale import fft2, ifft2, get_wn, meshgrid

###generate a random field
###  spec: the power spectrum of the field P(k2d)
###  hcorr: horizontal correlation length scale (m)
###  tcorr: time correlation length scale

def spectrum_gaussian(hcorr):
    return spec

def spectrum_slope(power_law):
    return lambda k: k**((power_law-1)/2)

def random_field(grid, pwr_spec):
    kx, ky = get_wn(grid.x)
    k2d = np.sqrt(kx**2 + ky**2)
    k2d[np.where(k2d==0.0)] = 1e-10
    noise = fft2(np.random.normal(0, 1, grid.x.shape))
    amplitude = pk(k2d)
    amplitude[np.where(k2d==1e-10)] = 0.0
    noise1 = np.real(ifft2(noise * amplitude))
    return (noise1 - np.mean(noise1))/np.std(noise1)

