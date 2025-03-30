import os
import numpy as np
#from utils.njit import njit
#import utils.spatial_operation as sop
from .base import Updator

class AlignmentUpdator(Updator):
    """Updator class with alignment technique"""

    def compute_increment(self, c, state):
        """
        Alignment technique: compute optical flows from the pair of prior/posterior state variable field
        """
        c.print_1p("Compute alignment based on analysis increment of '"+c.alignment['variable']+"'...\n")

        for rec_id in state.rec_list[c.pid_rec]:
            rec = state.info['fields'][rec_id]
            model = c.model_config[rec['model_src']]
            path = c.forecast_dir(rec['time'], rec['model_src'])
            if rec['name'] != c.alignment['variable']:
                continue

            for mem_id in state.mem_list[c.pid_mem]:
                ##compute displacement on analysis grid
                fld_prior = state.fields_prior[mem_id, rec_id]
                fld_post = state.fields_post[mem_id, rec_id]
                displace = optical_flow(c.grid, fld_prior, fld_post, **c.alignment)
                np.save(os.path.join(state.analysis_dir, f"displace.m{mem_id}.k{rec['k']}.npy"), displace)

                ##the model class offer a method to update grid (a lagrangian approach)
                ##so displace increment will be applied directly to the grid elements
                if hasattr(model, 'displace'):
                    ##convert the displacement from analysis grid to model grid
                    model.read_grid(path=path, member=mem_id, **rec)
                    c.grid.set_destination_grid(model.grid)
                    displace_m = c.grid.convert(displace, is_vector=True, method='linear')
                    ##apply the displacement
                    model.displace(displace_m[0,...], displace_m[1,...], path=path, member=mem_id, **rec)

        c.comm.Barrier()
        
    def update_restartfile(self, c, state, mem_id, rec_id):
        """
        Alignment technique, use the displace increment to adjust the model grid to
        precondition all the analysis variables for next assimilation step
        See more details in Ying 2019
        """
        rec = state.info['fields'][rec_id]
        model = c.model_config[rec['model_src']]
        path = c.forecast_dir(rec['time'], rec['model_src'])
        fld_prior = state.fields_prior[mem_id, rec_id]
        fld_post = state.fields_post[mem_id, rec_id]

        ##get model state variable prior
        model.read_grid(path=path, member=mem_id, **rec)
        c.grid.set_destination_grid(model.grid)
        var_prior = model.read_var(path=path, member=mem_id, **rec)

        if rec['is_vector']:
            fld_shape = var_prior.shape[1:]
        else:
            fld_shape = var_prior.shape

        ##read the corresponding displacement and convert to model grid
        displace = np.load(os.path.join(state.analysis_dir, f"displace.m{mem_id}.k{rec['k']}.npy"))

        ##warp the prior field with displacement
        fld_prior_warp = fld_prior.copy()
        u = displace[0,...]/c.grid.dx
        v = displace[1,...]/c.grid.dx
        for ind in np.ndindex(fld_prior.shape[:-2]):
            fld_prior_warp[ind] = warp(fld_prior[ind], -u, -v)

        ##residual increments not explained by the displacement
        res_incr = fld_post - fld_prior_warp

        if hasattr(model, 'displace'):
            ##apply the residual increment
            res_incr_m = c.grid.convert(res_incr, is_vector=rec['is_vector'], method='linear')
            if fld_shape == model.grid.x.shape:
                var_post = var_prior + res_incr_m
            elif fld_shape == model.grid.x_elem.shape:
                var_post = var_prior + np.mean(res_incr_m[...,model.grid.tri.triangles], axis=-1)
            else:
                raise RuntimeError(f"mismatch in field prior {var_prior.shape} with residual increment {res_incr_m.shape}")

        else:
            new_var = c.grid.convert(fld_post, is_vector=rec['is_vector'], method='linear')
            if fld_shape == model.grid.x.shape:
                var_post = new_var
            elif fld_shape == model.grid.x_elem.shape:
                var_post = np.mean(new_var[...,model.grid.tri.triangles], axis=-1)
            else:
                raise RuntimeError(f"mismatch in field prior {var_prior.shape} with posterior {new_var.shape}")
        
        #write the posterior variable to restart file
        ind = np.where(np.isnan(var_post))
        var_post[ind] = var_prior[ind]
        #if np.isnan(var_post).any():
        #    raise ValueError('nan detected in var_post')
        model.write_var(var_post, path=path, member=mem_id, comm=c.comm, **rec)


# def optical_flow(grid, fld1, fld2, nlevel=5, niter_max=100, smoothness_weight=1, **kwargs):
#     """
#     Compute optical flow from fld1 to fld2
#     Input:
#     - grid: Grid object for the field
#     - fld1, fld2: array, (ny, nx), the input 2D fields
#     - nlevel: levels of resolution used in pyramid method
#     - niter_max: max number of iterations used in minimization
#     - smoothness_weight: strength of the smoothness constraint
#     Return:
#     - u, v: array, (ny, nx), the optical flow so that fld1 warped with u,v is close to fld2
#     """
#     ny, nx = fld1.shape
#     assert fld1.shape == fld2.shape, 'input fields size mismatch!'
#     u, v = np.zeros((ny, nx)), np.zeros((ny, nx))

#     ##normalize fld
#     fld_max = np.nanmax(fld1)
#     fld_min = np.nanmin(fld1)
#     if (fld_max > fld_min):
#         fld1 = (fld1 - fld_min) / (fld_max - fld_min)
#         fld2 = (fld2 - fld_min) / (fld_max - fld_min)

#     ##remove nan, use 0 for now
#     fld1[grid.mask] = 0
#     fld2[grid.mask] = 0

#     for lev in range(nlevel, -1, -1): ##multigrid approach
#         fld1w = sop.warp(grid, fld1, u, v)
#         fld1w[np.where(np.isnan(fld1w))] = 0

#         gridc, fld1c = sop.coarsen(grid, fld1w, lev)
#         _,     fld2c = sop.coarsen(grid, fld2,  lev)
        
#         ##TODO: mask coarsening is growing, remove mask for now
#         #maskc = gridc.mask
#         maskc = np.full(fld1c.shape, False)

#         du_, dv_ = get_HS80_optical_flow(fld1c, fld2c, maskc, 1, 1, gridc.cyclic_dim, niter_max, smoothness_weight)

#         _, du = sop.refine(gridc, grid.mask, du_*(2**lev), lev)
#         _, dv = sop.refine(gridc, grid.mask, dv_*(2**lev), lev)
#         du[grid.mask] = 0
#         dv[grid.mask] = 0

#         u += du * grid.dx
#         v += dv * grid.dy

#     return np.array([u, v])

# @njit(cache=True)
# def get_HS80_optical_flow(fld1, fld2, mask, dx, dy, cyclic_dim, niter_max, w):
#     """ Get optical flow u,v using Horn & Schunck 1980 algorithm. """
#     ny, nx = fld1.shape

#     Ix = 0.5*(sop.gradx(fld1, dx, cyclic_dim) + sop.gradx(fld2, dx, cyclic_dim))
#     Iy = 0.5*(sop.grady(fld1, dy, cyclic_dim) + sop.grady(fld2, dy, cyclic_dim))
#     It = fld2 - fld1

#     u = np.zeros((ny, nx))
#     v = np.zeros((ny, nx))
#     u1 = np.ones((ny, nx))
#     v1 = np.ones((ny, nx))
#     niter = 0
#     diff = 1e7
#     while diff > 1e-3 and niter < niter_max:
#         ##enforce masked points to have no flow
#         for i in np.ndindex(mask.shape):
#             if mask[i]:
#                 u[i] = 0.
#                 v[i] = 0.

#         ##compute new flow
#         ubar = sop.laplacian(u, dx, dy, cyclic_dim) + u
#         vbar = sop.laplacian(v, dx, dy, cyclic_dim) + v
#         u1 = ubar - Ix*(Ix*ubar + Iy*vbar + It) / (w + Ix**2 + Iy**2)
#         v1 = vbar - Iy*(Ix*ubar + Iy*vbar + It) / (w + Ix**2 + Iy**2)

#         ##compare to previous iteration and update
#         diff = np.max(np.hypot(u1-u, v1-v))
#         u = u1
#         v = v1
#         niter += 1

#     return u, v

def optical_flow(grid, fld1, fld2, nlevel=5, niter_max=100, smoothness_weight=1, **kwargs):
    ni = int(2**np.ceil(np.log(np.max(fld1.shape))/np.log(2)))
    x1 = np.full((ni,ni), np.nan)
    x2 = np.full((ni,ni), np.nan)
    x1[0:grid.ny, 0:grid.nx] = fld1.copy()
    x2[0:grid.ny, 0:grid.nx] = fld2.copy()
    mask = np.logical_or(np.isnan(x1), (np.abs(x2-x1)<0.00001))
    x1[mask] = 0
    x2[mask] = 0
    w = smoothness_weight
    ni, nj = x1.shape
    ##normalize field so that w can be fixed
    xmax = np.max(x1[:, :]); xmin = np.min(x1[:, :])
    if (xmax>xmin):
        x1[:, :] = (x1[:, :] - xmin) / (xmax -xmin)
        x2[:, :] = (x2[:, :] - xmin) / (xmax -xmin)
    u = np.zeros((ni, nj))
    v = np.zeros((ni, nj))
    ###pyramid levels
    for lev in range(nlevel, -1, -1):
        x1w = warp(x1, -u, -v)
        x1c = coarsen(x1w, 1, lev)
        x2c = coarsen(x2, 1, lev)
        maskc = coarsen_mask(mask, 1, lev)
        xdx = 0.5*(deriv_x(x1c) + deriv_x(x2c))
        xdy = 0.5*(deriv_y(x1c) + deriv_y(x2c))
        xdt = x2c - x1c
        ###compute incremental flow using iterative solver
        du = np.zeros(xdx.shape)
        dv = np.zeros(xdx.shape)
        du1 = np.zeros(xdx.shape)
        dv1 = np.zeros(xdx.shape)
        niter = 0
        diff = 1e7
        while diff > 1e-3 and niter < niter_max:
            du[0,:] = 0; du[-1,:] = 0; du[:,0] = 0; du[:,-1] = 0  ##boundary conditions
            dv[0,:] = 0; dv[-1,:] = 0; dv[:,0] = 0; dv[:,-1] = 0
            du[maskc] = 0
            dv[maskc] = 0
            ubar = laplacian(du) + du
            vbar = laplacian(dv) + dv
            du1 = ubar - xdx*(xdx*ubar + xdy*vbar + xdt)/(w + xdx**2 + xdy**2)
            dv1 = vbar - xdy*(xdx*ubar + xdy*vbar + xdt)/(w + xdx**2 + xdy**2)
            diff = np.max(np.abs(du1-du) + np.abs(dv1-dv))
            du = du1
            dv = dv1
            niter += 1
        #print(niter, diff)
        u += sharpen(du*2**(lev-1), lev, 1)
        v += sharpen(dv*2**(lev-1), lev, 1)

    disp_u = u[0:grid.ny, 0:grid.nx] * grid.dx
    disp_v = v[0:grid.ny, 0:grid.nx] * grid.dy
    return np.array([disp_u, disp_v])

def warp(x, u, v):
    xw = x.copy()
    ni, nj = x.shape
    ii, jj = np.mgrid[0:ni, 0:nj]
    xw = interp2d(x, ii+v, jj+u)
    return xw

def coarsen_mask(x, lev1, lev2):  ##only subsample no smoothing, avoid mask growing
    if lev1 < lev2:
        for _ in range(lev1, lev2):
            ni, nj = x.shape
            x1 = x[0:ni:2, 0:nj:2]
            x = x1
    return x

def coarsen(x, lev1, lev2):
    if lev1 < lev2:
        for _ in range(lev1, lev2):          
            ni, nj = x.shape
            x1 = 0.25*(x[0:ni:2, 0:nj:2] + x[1:ni:2, 0:nj:2] + x[0:ni:2, 1:nj:2] + x[1:ni:2, 1:nj:2])
            x = x1
    return x

def sharpen(x, lev1, lev2):
    if lev1 > lev2:
        for _ in range(lev1, lev2, -1):
            ni, nj = x.shape
            x1 = np.zeros((ni*2, nj))
            x1[0:ni*2:2, :] = x
            x1[1:ni*2:2, :] = 0.5*(np.roll(x, -1, axis=0) + x)
            x2 = np.zeros((ni*2, nj*2))
            x2[:, 0:nj*2:2] = x1
            x2[:, 1:nj*2:2] = 0.5*(np.roll(x1, -1, axis=1) + x1)
            x = x2
    return x

def interp2d(x, io, jo):
    ni, nj = x.shape
    io1 = np.floor(io).astype(int) % ni
    jo1 = np.floor(jo).astype(int) % nj
    io2 = np.floor(io+1).astype(int) % ni
    jo2 = np.floor(jo+1).astype(int) % nj
    di = io - np.floor(io)
    dj = jo - np.floor(jo)
    xo = (1-di)*(1-dj)*x[io1, jo1] + di*(1-dj)*x[io2, jo1] + (1-di)*dj*x[io1, jo2] + di*dj*x[io2, jo2]
    return xo

def deriv_y(f):
    return 0.5*(np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))

def deriv_x(f):
    return 0.5*(np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))

def laplacian(f):
    out = (np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) + np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0))/6
    out += (np.roll(np.roll(f, -1, axis=0), -1, axis=1) + np.roll(np.roll(f, -1, axis=0), 1, axis=1) + np.roll(np.roll(f, 1, axis=0), -1, axis=1) + np.roll(np.roll(f, 1, axis=0), 1, axis=1))/12
    out -= f
    return out
