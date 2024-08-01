import numpy as np

def gradx(fld, dx, cyclic_dim=None):
    """gradient of input field in x direction
    input:
        -fld: array, input field, last two dimensions (ny, nx)
        -dx: grid spacing in x
        -cyclic_dim: None (default) not cyclic boundary, or string 'x', 'y', 'xy', etc.
    return:
        -gradx of fld with same shape
    """
    fld_gradx = np.zeros(fld.shape)
    if cyclic_dim is not None and 'x' in cyclic_dim:
        fld_gradx = (np.roll(fld, -1, axis=-1) - np.roll(fld, 1, axis=-1)) / (2.0*dx)
    else:
        fld_gradx[..., 1:-1] = (fld[..., 2:] - fld[..., :-2]) / (2.0*dx)
        fld_gradx[..., 0] = (fld[..., 1] - fld[..., 0]) / dx
        fld_gradx[..., -1] = (fld[..., -2] - fld[..., -1]) / dx
    return fld_gradx


def grady(fld, dy, cyclic_dim=None):
    """gradient of input fld in y direction, similar to gradx"""
    fld_grady = np.zeros(fld.shape)
    if cyclic_dim is not None and 'y' in cyclic_dim:
        fld_grady = (np.roll(fld, -1, axis=-2) - np.roll(fld, 1, axis=-2)) / (2.0*dy)
    else:
        fld_grady[..., 1:-1, :] = (fld[..., 2:, :] - fld[..., :-2, :]) / (2.0*dy)
        fld_grady[..., 0, :] = (fld[..., 1, :] - fld[..., 0, :]) / dy
        fld_grady[..., -1, :] = (fld[..., -2, :] - fld[..., -1, :]) / dy
    return fld_grady


def deriv_x(grid, fld):
    assert grid.x.shape == fld.shape[-2:], "deriv_x: input fld size mismatch with grid"
    return gradx(fld, grid.dx, grid.cyclic_dim)


def deriv_y(grid, fld):
    assert grid.x.shape == fld.shape[-2:], "deriv_y: input fld size mismatch with grid"
    return grady(fld, grid.dy, grid.cyclic_dim)


def deriv_xx(grid, fld):
    return deriv_x(grid, deriv_x(grid, fld))


def deriv_yy(grid, fld):
    return deriv_y(grid, deriv_y(grid, fld))


def deriv_xy(grid, fld):
    return deriv_y(grid, deriv_x(grid, fld))


def laplacian(grid, fld):
    return deriv_xx(grid, fld) + deriv_yy(grid, fld)


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
    grid1 = grid.change_resolution_level(nlevel)
    grid.set_destination_grid(grid1)
    fld1 = np.zeros(fld.shape[:-2]+(grid1.ny, grid1.nx))
    for ind in np.ndindex(fld.shape[:-2]):
        fld1[ind] = grid.convert(fld[ind], is_vector=False, method='linear', coarse_grain=True)
    return grid1, fld1


def sharpen(grid, fld, nlevel):
    """
    sharpen the image by upsampling the grid points by factors of 2,
    input:
    - grid: Grid object of the original grid
    - fld: array with last two dimensions ny,nx
    - nlevel: int, number of resolution levels to go up
    return:
    - grid1: new grid with higher resolution
    - fld1: new field on new grid
    """
    assert grid.x.shape == fld.shape[-2:], "sharpen: input fld size mismatch with grid"
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
        fld1[ind, ...] = grid.interp(fld[ind, ...], grid.x+u, grid.y+v, method='linear')
    return fld1

