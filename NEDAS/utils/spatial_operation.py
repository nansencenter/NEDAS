import numpy as np
from NEDAS.utils.njit import njit

@njit
def gradx(fld, dx, cyclic_dim=None):
    """Gradient of input field in x direction

    Args:
        fld (np.ndarray): input field, last two dimensions (ny, nx)
        dx (int): grid spacing in x, fld.shape
        cyclic_dim (str, optional): string 'x', 'y', 'xy', indicating the dimension(s) that are cyclic.

    Returns:
        np.ndarray: gradx of fld with same shape
    """
    fld_gradx = np.zeros(fld.shape)
    if cyclic_dim is not None and 'x' in cyclic_dim:
        ##centered difference of the neighboring grid points
        fld_gradx[..., :-1] = fld[..., 1:]  ##right neighbor
        fld_gradx[..., -1]  = fld[..., 0]
        fld_gradx[..., 1:] -= fld[..., :-1] ##left neighbor
        fld_gradx[..., 0]  -= fld[..., -1]
        fld_gradx /= 2.*dx
    else:
        ##centered difference for the middle part
        fld_gradx[..., 1:-1] = (fld[..., 2:] - fld[..., :-2]) / 2.0
        ##one-sided difference for the left,right edge points
        fld_gradx[..., 0] = (fld[..., 1] - fld[..., 0])
        fld_gradx[..., -1] = (fld[..., -1] - fld[..., -2])
        fld_gradx /= dx
    return fld_gradx

@njit
def grady(fld, dy, cyclic_dim=None):
    """gradient of input fld in y direction, similar to gradx"""
    fld_grady = np.zeros(fld.shape)
    if cyclic_dim is not None and 'y' in cyclic_dim:
        ##centered difference of the neighboring grid points
        fld_grady[..., :-1, :] = fld[..., 1:, :]
        fld_grady[..., -1, :]  = fld[..., 0, :]
        fld_grady[..., 1:, :] -= fld[..., :-1, :]
        fld_grady[..., 0, :]  -= fld[..., -1, :]
        fld_grady /= 2.*dy
    else:
        ##centered difference for the middle part
        fld_grady[..., 1:-1, :] = (fld[..., 2:, :] - fld[..., :-2, :]) / 2.0
        ##one-sided difference for the left,right edge points
        fld_grady[..., 0, :] = (fld[..., 1, :] - fld[..., 0, :])
        fld_grady[..., -1, :] = (fld[..., -1, :] - fld[..., -2, :])
        fld_grady /= dy
    return fld_grady

@njit
def gradx2(fld, dx, cyclic_dim=None):
    return gradx(gradx(fld, dx, cyclic_dim), dx, cyclic_dim)

@njit
def grady2(fld, dy, cyclic_dim=None):
    return grady(grady(fld, dy, cyclic_dim), dy, cyclic_dim)

@njit
def gradxy(fld, dx, dy, cyclic_dim=None):
    return grady(gradx(fld, dx, cyclic_dim), dy, cyclic_dim)

@njit
def laplacian(fld, dx, dy, cyclic_dim=None):
    return gradx2(fld, dx, cyclic_dim) + grady2(fld, dy, cyclic_dim)


def coarsen(grid, fld, nlevel):
    """
    Coarsen the image by downsampling the grid points by factors of 1/2,

    Args:
        grid (Grid): the original grid
        fld (np.ndarray): field with last two dimensions ny,nx
        nlevel (int): number of resolution levels to go down

    Returns:
        Grid: new grid with lower resolution
        np.ndarray: new field defined on the new grid
    """
    assert grid.x.shape == fld.shape[-2:], "coarsen: input fld size mismatch with grid"
    if nlevel == 0:
        return grid, fld
    grid1 = grid.change_resolution_level(nlevel)
    grid.set_destination_grid(grid1)
    fld1 = np.zeros(fld.shape[:-2]+(grid1.ny, grid1.nx))
    for ind in np.ndindex(fld.shape[:-2]):
        fld1[ind] = grid.convert(fld[ind], is_vector=False, method='linear', coarse_grain=True)
    return grid1, fld1


def refine(grid, mask, fld, nlevel):
    """
    Refine the image by upsampling the grid points by factors of 2,

    Args:
        grid (Grid): the original grid
        fld (np.ndarray): field with last two dimensions ny,nx
        nlevel (int): number of resolution levels to go up

    Returns:
        Grid: new grid with higher resolution
        np.ndarray: new field on new grid
    """
    assert grid.x.shape == fld.shape[-2:], "refine: input fld size mismatch with grid"
    if nlevel == 0:
        return grid, fld
    grid1 = grid.change_resolution_level(-nlevel)
    grid1.mask = mask  ##don't know how to sharpen mask, need to provide high-res mask from input args
    grid.set_destination_grid(grid1)
    fld1 = np.zeros(fld.shape[:-2]+(grid1.ny, grid1.nx))
    for ind in np.ndindex(fld.shape[:-2]):
        fld_nearest = grid.convert(fld[ind], is_vector=False, method='nearest')
        fld_linear = grid.convert(fld[ind], is_vector=False, method='linear')
        void = np.isnan(fld_linear)
        fld_linear[void] = fld_nearest[void]
        fld1[ind] = fld_linear
    return grid1, fld1

##some original coarsen/sharpen functions working for 2**n grid points
def coarsen_mask(mask, lev1, lev2):  ##only subsample no smoothing, avoid mask growing
    if lev1 < lev2:
        for k in range(lev1, lev2):
            ni, nj = mask.shape[-2:]
            mask1 = mask[..., 0:ni:2, 0:nj:2]
            mask = mask1
    return mask

def coarsen_field(fld, lev1, lev2):
    if lev1 < lev2:
        for k in range(lev1, lev2):
            ni, nj = fld.shape[-2:]
            fld1 = 0.25*(fld[..., 0:ni:2, 0:nj:2] + fld[..., 1:ni:2, 0:nj:2] + fld[..., 0:ni:2, 1:nj:2] + fld[..., 1:ni:2, 1:nj:2])
            fld = fld1
    return fld

def sharpen_field(fld, lev1, lev2):
    if lev1 > lev2:
        for k in range(lev1, lev2, -1):
            ni, nj = fld.shape[-2:]
            fld1 = np.zeros(fld.shape[:-2]+(ni*2, nj))
            fld1[..., 0:ni*2:2, :] = fld
            fld1[..., 1:ni*2:2, :] = 0.5*(np.roll(fld, -1, axis=-2) + fld)
            fld2 = np.zeros(fld.shape[:-2]+(ni*2, nj*2))
            fld2[..., :, 0:nj*2:2] = fld1
            fld2[..., :, 1:nj*2:2] = 0.5*(np.roll(fld1, -1, axis=-1) + fld1)
            fld = fld2
    return fld

def warp(grid, fld, u, v):
    """
    Warp the image with input vector field

    Args:
        grid (Grid): the grid on which the image is defined
        fld (np.ndarray): input image
        u (np.ndarray): displacement vector x component, in :code:`grid.x` units
        v (np.ndarray): displacement vector y component, in :code:`grid.y` units

    Returns:
        np.ndarray: the warped image
    """
    assert grid.x.shape == fld.shape[-2:], "warp: input fld size mismatch with grid"
    assert grid.x.shape == u.shape, "warp: input u size mismatch with grid"
    assert grid.x.shape == v.shape, "warp: input v size mismatch with grid"
    fld1 = fld.copy()
    for ind in np.ndindex(fld.shape[:-2]):
        fld_nearest = grid.interp(fld[ind], grid.x-u, grid.y-v, method='nearest')
        fld_linear = grid.interp(fld[ind], grid.x-u, grid.y-v, method='linear')
        void = np.isnan(fld_linear)
        fld_linear[void] = fld_nearest[void]
        fld1[ind] = fld_linear
    return fld1

