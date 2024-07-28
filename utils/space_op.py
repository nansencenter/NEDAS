import numpy as np

def gradx(fld, dx):
    gradx_fld = np.zeros(fld.shape)
    gradx_fld[..., 1:] = (fld[..., 1:] - fld[..., :-1]) / dx
    return gradx_fld


def grady(fld, dy):
    grady_fld = np.zeros(fld.shape)
    grady_fld[..., 1:, :] = (fld[..., 1:, :] - fld[..., :-1, :]) / dy
    return grady_fld


def warp(fld, u, v):
    fld_warp = fld.copy()
    ny, nx = fld.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    fld_warp = interp2d(fld, ii+u, jj+v)
    return fld_warp

def interp2d(x, io, jo):
    nj, ni = x.shape
    io1 = np.floor(io).astype(int) % ni
    jo1 = np.floor(jo).astype(int) % nj
    io2 = np.floor(io+1).astype(int) % ni
    jo2 = np.floor(jo+1).astype(int) % nj
    di = io - np.floor(io)
    dj = jo - np.floor(jo)
    xo = (1-di)*(1-dj)*x[jo1, io1] + di*(1-dj)*x[jo1, io2] + (1-di)*dj*x[jo2, io1] + di*dj*x[jo2, io2]
    return xo

