import numpy as np
import os
from utils.parallel import by_rank
from utils.njit import njit
from utils.conversion import t2s, s2t
from utils.progress import print_with_cache, progress_bar
import utils.spatial_operation as sop

def optical_flow(grid, mask, fld1, fld2, nlevel=5, niter_max=100, smoothness_weight=1, **kwargs):
    """
    Compute optical flow from fld1 to fld2
    Input:
    - grid: Grid object for the field
    - mask: bool array, mask of the fixed points without displacement
    - fld1, fld2: array, (ny, nx), the input 2D fields
    - nlevel: levels of resolution used in pyramid method
    - niter_max: max number of iterations used in minimization
    - smoothness_weight: strength of the smoothness constraint
    Return:
    - u, v: array, (ny, nx), the optical flow so that fld1 warped with u,v is close to fld2
    """
    ny, nx = fld1.shape
    assert fld1.shape == fld2.shape, 'input fields size mismatch!'
    u, v = np.zeros((ny, nx)), np.zeros((ny, nx))

    for lev in range(nlevel, -1, -1): ##multigrid approach
        fld1w = sop.warp(grid, fld1, u, v)
        gridc, fld1c = sop.coarsen(grid, fld1w, lev)
        _,     fld2c = sop.coarsen(grid, fld2,  lev)
        maskc = np.full(fld1c.shape, False, dtype=bool)
        du_, dv_ = get_HS80_optical_flow(fld1c, fld2c, maskc, gridc.dx, gridc.dy, gridc.cyclic_dim, niter_max, smoothness_weight)
        _, du = sop.refine(gridc, du_, lev)
        _, dv = sop.refine(gridc, dv_, lev)
        u += du
        v += dv
    return np.array([u, v])

@njit(cache=True)
def get_HS80_optical_flow(fld1, fld2, mask, dx, dy, cyclic_dim, niter_max, w):
    """ Get optical flow u,v using Horn & Schunck 1980 algorithm. """
    ny, nx = fld1.shape
    Ix = 0.5*(sop.gradx(fld1, dx, cyclic_dim) + sop.gradx(fld2, dx, cyclic_dim))
    Iy = 0.5*(sop.grady(fld1, dy, cyclic_dim) + sop.grady(fld2, dy, cyclic_dim))
    It = fld2 - fld1
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    u1 = np.ones((ny, nx))
    v1 = np.ones((ny, nx))
    niter = 0
    diff = 1e7
    while diff > 1e-3 and niter < niter_max:
        ##enforce masked points to have no flow
        for i in np.ndindex(mask.shape):
            if mask[i]:
                u[i] = 0.
                v[i] = 0.
        ##compute new flow
        ubar = sop.laplacian(u, dx, dy, cyclic_dim) + u
        vbar = sop.laplacian(v, dx, dy, cyclic_dim) + v
        u1 = ubar - Ix*(Ix*ubar + Iy*vbar + It) / (w + Ix**2 + Iy**2)
        v1 = vbar - Iy*(Ix*ubar + Iy*vbar + It) / (w + Ix**2 + Iy**2)
        ##compare to previous iteration and update
        diff = np.max(np.hypot(u1-u, v1-v))
        u = u1
        v = v1
        niter += 1
    return u, v

def alignment(c, fields_prior, fields_post, **kwargs):
    """
    Alignment technique, find displacement increment from prior to posterior
    for one variable, then use this increment to adjust the model grid to
    precondition all the analysis variables for next assimilation step
    See more details in Ying 2019

    Inputs:
    - c: config object
    - fields_prior, fields_post: prior and posterior fields dict
    """
    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    print("alignment based on analysis increment of '"+c.alignment['variable']+"'\n")
    option = kwargs.get('option', 'optical_flow')

    ##compute the displacement increments
    for rec_id in c.rec_list[c.pid_rec]:
        rec = c.state_info['fields'][rec_id]
        if rec['name'] != c.alignment['variable']:
            continue
        for mem_id in c.mem_list[c.pid_mem]:
            if option == 'optical_flow':
                fld1 = fields_prior[mem_id, rec_id]
                fld2 = fields_post[mem_id, rec_id]
                displace = optical_flow(c.grid, c.mask, fld1, fld2, **kwargs)
                np.save(os.path.join(c.analysis_dir, f"displace.m{mem_id}.k{rec['k']}.npy"), displace)
            else:
                raise NotImplementedError(f"alignment: option '{option}' is not implemented!")
    c.comm.Barrier()

    ##apply the displacement increments
    # loop over mem_id
    # loop over model
    # for rec_id in c.rec_list[c.pid_rec]:
    #     rec = c.state_info['fields'][rec_id]
    #     model = c.model_config[rec['model_src']]

    #     for mem_id in c.mem_list[c.pid_mem]:
    #         ##directory storing model restart files
    #         path = forecast_dir(c, rec['time'], rec['model_src'])
    #         if hasattr(model, 'update_grid'):
    #             ##the model class offer a method to update grid (a lagrangian approach)
    #             ##so displace increment will be applied directly to the grid elements
    #             model.grid.x += u
    #             model.grid.y += v
    #             model.update_grid(path=path, member=mem_id, **rec)

    #         else:
    #             ##change all the model variables to reflect the displace increment
    #             sop.warp(grid, fld, u, v)

