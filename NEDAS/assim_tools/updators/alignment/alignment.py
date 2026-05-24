import numpy as np
from NEDAS.core import Context, Updator
from .optical_flow import OpticalFlow

class AlignmentUpdator(Updator):
    """Updator class with alignment technique"""
    displace = {}

    def __init__(self, c: Context):
        super().__init__(c)
        self.optical_flow = OpticalFlow(**c.config.alignment)

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
                # compute displacement on analysis grid
                fld_prior = c.state.fields_prior[mem_id, rec_id]
                fld_post = c.state.fields_post[mem_id, rec_id]
                displace = self.optical_flow(c.grid, fld_prior, fld_post, **c.config.alignment)
                self.displace[mem_id, rec['k']] = displace
                #np.save(os.path.join(c.io.analysis_dir, f"displace.m{mem_id}.k{rec['k']}.npy"), displace)

                # the model class offer a method to update grid (a lagrangian approach)
                # so displace increment will be applied directly to the grid elements
                if hasattr(model, 'displace'):
                    # convert the displacement from analysis grid to model grid
                    c.io.call_method(c, 'current', model.read_grid, member=mem_id, **rec)
                    c.grid.set_destination_grid(model.grid)
                    displace_m = c.grid.convert(displace, is_vector=True, method='linear')
                    # apply the displacement
                    c.io.call_method(c, 'current', getattr(model, 'displace'), displace_m[0,...], displace_m[1,...], member=mem_id, **rec)

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

        # get model state variable prior
        var_prior = c.io.call_method(c, 'current', model.read_var, member=mem_id, **rec)
        c.grid.set_destination_grid(model.grid)

        if rec['is_vector']:
            fld_shape = var_prior.shape[1:]
        else:
            fld_shape = var_prior.shape

        # read the corresponding displacement and convert to model grid
        #displace = np.load(os.path.join(state.analysis_dir, f"displace.m{mem_id}.k{rec['k']}.npy"))
        displace = self.displace[mem_id, rec['k']]

        # warp the prior field with displacement
        fld_prior_warp = fld_prior.copy()
        u = displace[0,...]/c.grid.dx
        v = displace[1,...]/c.grid.dx
        for ind in np.ndindex(fld_prior.shape[:-2]):
            fld_prior_warp[ind] = warp(fld_prior[ind], -u, -v)

        # residual increments not explained by the displacement
        res_incr = fld_post - fld_prior_warp

        if hasattr(model, 'displace'):
            # apply the residual increment
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
        c.io.call_method(c, 'current', model.write_var, var_post, member=mem_id, comm=c.comm, **rec)

def warp(x, u, v):
    xw = x.copy()
    ni, nj = x.shape
    ii, jj = np.mgrid[0:ni, 0:nj]
    xw = interp2d(x, ii+v, jj+u)
    return xw

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
