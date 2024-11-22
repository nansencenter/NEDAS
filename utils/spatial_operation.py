import numpy as np
from .njit import njit

@njit
def gradx(fld, dx, cyclic_dim=None):
    """gradient of input field in x direction
    input:
        -fld: array, input field, last two dimensions (ny, nx)
        -dx: grid spacing in x, fld.shape
        -cyclic_dim: None (default) not cyclic boundary, or string 'x', 'y', 'xy', etc.
    return:
        -gradx of fld with same shape
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
        fld_gradx[..., -1] = (fld[..., -2] - fld[..., -1])
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
        fld_grady[..., -1, :] = (fld[..., -2, :] - fld[..., -1, :])
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
    coarsen the image by downsampling the grid points by factors of 1/2,
    input:
    - grid: Grid object of the original grid
    - fld: array with last two dimensions ny,nx
    - nlevel: int, number of resolution levels to go down
    return:
    - grid1: new grid with lower resolution
    - fld1: new field on new grid
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


def refine(grid, fld, nlevel):
    """
    refine the image by upsampling the grid points by factors of 2,
    input:
    - grid: Grid object of the original grid
    - fld: array with last two dimensions ny,nx
    - nlevel: int, number of resolution levels to go up
    return:
    - grid1: new grid with higher resolution
    - fld1: new field on new grid
    """
    assert grid.x.shape == fld.shape[-2:], "refine: input fld size mismatch with grid"
    if nlevel == 0:
        return grid, fld
    grid1 = grid.change_resolution_level(-nlevel)
    grid.set_destination_grid(grid1)
    fld1 = np.zeros(fld.shape[:-2]+(grid1.ny, grid1.nx))
    for ind in np.ndindex(fld.shape[:-2]):
        fld1[ind] = grid.convert(fld[ind], is_vector=False, method='linear')
    return grid1, fld1


def warp(grid, fld, u, v):
    """
    warp the image with input vector field
    input:
    - grid: Grid object
    - fld: input image
    - u, v: vector field (in grid coordinate units)
    """
    assert grid.x.shape == fld.shape[-2:], "warp: input fld size mismatch with grid"
    assert grid.x.shape == u.shape, "warp: input u size mismatch with grid"
    assert grid.x.shape == v.shape, "warp: input v size mismatch with grid"
    fld1 = fld.copy()
    for ind in np.ndindex(fld.shape[:-2]):
        fld1[ind] = grid.interp(fld[ind], grid.x-u, grid.y-v, method='linear')
    return fld1

