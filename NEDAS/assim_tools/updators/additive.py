import numpy as np
from .base import Updator

class AdditiveUpdator(Updator):

    def compute_increment(self, c, state):
        """
        Additive updator: just compute the difference between prior and posterior as increments
        """
        self.increment = {}
        for mem_id, rec_id in state.fields_prior.keys():
            rec = state.info['fields'][rec_id]
            fld_prior = state.fields_prior[mem_id, rec_id]
            fld_post = state.fields_post[mem_id, rec_id]

            ##misc transform inverse
            for transform_func in c.transform_funcs:
                fld_prior = transform_func.backward_state(c, rec, fld_prior)
                fld_post = transform_func.backward_state(c, rec, fld_post)

            ##collect the increments
            self.increment[mem_id, rec_id] = fld_post - fld_prior

    def update_restartfile(self, c, state, mem_id, rec_id):
        """
        Method to update a single field rec_id in the model restart file.
        This can be overridden by derived classes for specific update methods
        Inputs:
        - c: config object
        - state: state object
        - mem_id: member index
        - rec_id: record index
        """
        rec = state.info['fields'][rec_id]
        model = c.models[rec['model_src']]
        path = c.forecast_dir(rec['time'], rec['model_src'])
        model.read_grid(path=path, member=mem_id, **rec)

        ##convert the posterior variable back to native model grid
        var_prior = model.read_var(path=path, member=mem_id, **rec)

        c.grid.set_destination_grid(model.grid)
        incr = c.grid.convert(self.increment[mem_id, rec_id], is_vector=rec['is_vector'], method='linear')

        if rec['is_vector']:
            fld_shape = var_prior.shape[1:]
        else:
            fld_shape = var_prior.shape

        if fld_shape == model.grid.x.shape:
            var_post = var_prior + incr
        elif fld_shape == model.grid.x_elem.shape:
            incr = np.mean(incr[...,model.grid.tri.triangles], axis=-1)
            var_post = var_prior + incr
        else:
            raise RuntimeError(f"mismatch in field prior {var_prior.shape} with increment {incr.shape}")

        ##TODO: temporary solution for nan values due to interpolation
        ind = np.where(np.isnan(var_post))
        var_post[ind] = var_prior[ind]
        # if np.isnan(var_post).any():
        #     raise ValueError('nan detected in var_post')

        ##write the posterior variable to restart filediff
        model.write_var(var_post, path=path, member=mem_id, comm=c.comm, **rec)
