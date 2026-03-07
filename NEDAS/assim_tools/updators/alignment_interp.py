import os
import numpy as np
from NEDAS.core import Context, Updator

class AlignmentUpdator(Updator):
    """Updator class with alignment technique"""
    displace = {}

    def compute_increment(self, c: Context):
        """
        Alignment technique: compute optical flows from the pair of prior/posterior state variable field
        """
        if c.config.alignment is None:
            c.config.alignment = {}
        c.print_1p("Compute alignment based on analysis increment of '"+c.config.alignment['variable']+"'...\n")

        for rec_id in c.state.rec_list[c.pid_rec]:
            rec = c.state.info.fields[rec_id].asdict()
            model = c.models[rec['model_src']]
            if rec['name'] != c.config.alignment['variable']:
                continue

            for mem_id in c.mem_list[c.pid_mem]:
                ##compute displacement on analysis grid
                fld_prior = c.state.fields_prior[mem_id, rec_id]
                fld_post = c.state.fields_post[mem_id, rec_id]
                displace = optical_flow(c.grid, fld_prior, fld_post, **c.config.alignment)
                self.displace[mem_id, rec['k']] = displace
                #np.save(os.path.join(state.analysis_dir, f"displace.m{mem_id}.k{rec['k']}.npy"), displace)

        c.comm.Barrier()

    def update_files(self, c, mem_id, rec_id):
        """
        Alignment technique, use the displace increment to adjust the model grid to
        precondition all the analysis variables for next assimilation step
        See more details in Ying 2019
        """
        rec = c.state.info.fields[rec_id].asdict()
        model = c.models[rec['model_src']]

        fld_prior = c.state.fields_prior[mem_id, rec_id]
        fld_post = c.state.fields_post[mem_id, rec_id]

        ##get model state variable prior
        var_prior = c.io.call_io_method(c, 'prior', model.read_var, member=mem_id, **rec)
        c.grid.set_destination_grid(model.grid)

        if rec['is_vector']:
            fld_shape = var_prior.shape[1:]
        else:
            fld_shape = var_prior.shape

        ##read the corresponding displacement and convert to model grid
        displace = self.displace[mem_id, rec['k']]
        # displace = np.load(os.path.join(state.analysis_dir, f"displace.m{mem_id}.k{rec['k']}.npy"))

        ##warp the prior field with displacement
        fld_prior_warp = fld_prior.copy()
        u = displace[0,...]/c.grid.dx
        v = displace[1,...]/c.grid.dx
        for ind in np.ndindex(fld_prior.shape[:-2]):
            fld_prior_warp[ind] = warp(fld_prior[ind], -u, -v)

        # ##residual increments not explained by the displacement
        res_incr = fld_post - fld_prior_warp

        ##convert the displacement from analysis grid to model grid
        displace_m = c.grid.convert(displace, is_vector=True, method='linear')
        u, v = displace_m[0,...], displace_m[1,...]
        taper_boundary = getattr(model, 'taper_boundary')
        u = taper_boundary(u)
        v = taper_boundary(v)

        ##apply the residual increment
        res_incr_m = c.grid.convert(res_incr, is_vector=rec['is_vector'], method='linear')
        if fld_shape == model.grid.x.shape:
            if rec['is_vector']:
                var_prior_warp_x = model.grid.interp(var_prior[0,...], model.grid.x-u, model.grid.y-v)
                var_prior_warp_y = model.grid.interp(var_prior[1,...], model.grid.x-u, model.grid.y-v)
                var_prior_warp = np.array([var_prior_warp_x, var_prior_warp_y])
            else:
                var_prior_warp = model.grid.interp(var_prior[...], model.grid.x-u, model.grid.y-v)
            var_post = var_prior_warp + res_incr_m
        elif fld_shape == model.grid.x_elem.shape:
            u_elem = np.mean(u[...,model.grid.tri.triangles], axis=-1)
            v_elem = np.mean(v[...,model.grid.tri.triangles], axis=-1)
            var_prior_warp = model.grid.interp(var_prior, model.grid.x_elem-u_elem, model.grid.y_elem-v_elem)
            var_post = var_prior_warp + np.mean(res_incr_m[...,model.grid.tri.triangles], axis=-1)
        else:
            raise RuntimeError(f"mismatch in field prior {var_prior.shape} with residual increment {res_incr_m.shape}")

        #write the posterior variable to restart file
        ind = np.where(np.isnan(var_post))
        var_post[ind] = var_prior[ind]
        #if np.isnan(var_post).any():
        #    raise ValueError('nan detected in var_post')
        c.io.call_io_method(c, 'prior', model.write_var, var_post, member=mem_id, comm=c.comm, **rec)

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
