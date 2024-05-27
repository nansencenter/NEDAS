import numpy as np

def gradx(fld, dx):
    gradx_fld = np.zeros(fld.shape)
    gradx_fld[..., 1:] = (fld[..., 1:] - fld[..., :-1]) / dx
    return gradx_fld


def grady(fld, dy):
    grady_fld = np.zeros(fld.shape)
    grady_fld[..., 1:, :] = (fld[..., 1:, :] - fld[..., :-1, :]) / dy
    return grady_fld

