import numpy as np
from NEDAS.core import Context, Updator
from ..optical_flow import OpticalFlow, warp

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
                displace = self.optical_flow(c.grid, fld_prior, fld_post)
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

        # get model state variable prior
        var_prior = c.io.call_method(c, 'current', model.read_var, member=mem_id, **rec)
        c.grid.set_destination_grid(model.grid)

        if rec['is_vector']:
            fld_shape = var_prior.shape[1:]
        else:
            fld_shape = var_prior.shape

        # read the corresponding displacement and convert to model grid
        displace = self.displace[mem_id, rec['k']]
        # displace = np.load(os.path.join(state.analysis_dir, f"displace.m{mem_id}.k{rec['k']}.npy"))

        # warp the prior field with displacement
        fld_prior_warp = fld_prior.copy()
        u = displace[0,...]/c.grid.dx
        v = displace[1,...]/c.grid.dx
        for ind in np.ndindex(fld_prior.shape[:-2]):
            fld_prior_warp[ind] = warp(fld_prior[ind], -u, -v)

        # # residual increments not explained by the displacement
        res_incr = fld_post - fld_prior_warp

        # convert the displacement from analysis grid to model grid
        displace_m = c.grid.convert(displace, is_vector=True, method='linear')
        u, v = displace_m[0,...], displace_m[1,...]
        taper_boundary = getattr(model, 'taper_boundary')
        u = taper_boundary(u)
        v = taper_boundary(v)

        # apply the residual increment
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
        c.io.call_method(c, 'current', model.write_var, var_post, member=mem_id, comm=c.comm, **rec)

