import numpy as np
from .base import Updator

class AdditiveUpdator(Updator):

    def compute_increment(self, c, fields_prior, fields_post):

        increment = {}
        for mem_id, rec_id in fields_prior.keys():
            increment[mem_id, rec_id] = fields_post[mem_id, rec_id] - fields_prior[mem_id, rec_id]

        return increment

    def update_restartfile(self, c, mem_id, rec_id, rec, path, model, increment):
        """
        Method to update a single field rec_id in the model restart file.
        This can be overridden by derived classes for specific update methods
        Inputs:
        - c: config module
        - mem_id: member index
        - rec_id: record index
        - rec: c.state_info['fields'][rec_id] record
        - path: path to the restart file
        - model: model module
        - fields_prior, fields_post: the field-complete state variables before and after assimilate()
        """
        model.read_grid(path=path, member=mem_id, **rec)

        ##misc. inverse transform
        ##e.g. multiscale approach: just use the analysis increment directly

        ##convert the posterior variable back to native model grid
        var_prior = model.read_var(path=path, member=mem_id, **rec)

        c.grid.set_destination_grid(model.grid)
        incr = c.grid.convert(increment[mem_id, rec_id], is_vector=rec['is_vector'], method='linear')

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
        if np.isnan(var_post).any():
            raise ValueError('nan detected in var_post')

        ##write the posterior variable to restart file
        model.write_var(var_post, path=path, member=mem_id, comm=c.comm, **rec)
