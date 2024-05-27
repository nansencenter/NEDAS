import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from utils.space_op import gradx, grady

##prototype: generate a random wavenumber-1 displacement for the whole domain
def random_displacement(grid, mask, amp, hcorr):

    ##set the boundaries to be masked if they are not cyclic
    ##i.e. we don't want displacement across non-cyclic boundary
    if grid.cyclic_dim is not None:
        cyclic_dim = grid.cyclic_dim
    else:
        cyclic_dim = ''
    if 'x' not in cyclic_dim:
        mask[:, 0] = True
        mask[:, -1] = True
    if 'y' not in cyclic_dim:
        mask[0, :] = True
        mask[-1, :] = True

    ##distance to masked area
    dist = distance_transform_edt(1-mask.astype(float))
    dist /= np.max(dist)
    dist = gaussian_filter(dist, sigma=10)
    dist[mask] = 0

    ##the displacement vector field from a random streamfunction
    psi = random_field_gaussian(grid.nx, grid.ny, 1, hcorr)

    du, dv = -grady(psi, 1), gradx(psi, 1)
    norm = np.hypot(du, dv).max()
    du *= amp / norm
    dv *= amp / norm

    du *= dist  ##taper near mask
    dv *= dist

    return du, dv

