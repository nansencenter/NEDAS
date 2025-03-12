
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
        c.grid.set_destination_grid(model.grid)

        ##misc. inverse transform
        ##e.g. multiscale approach: just use the analysis increment directly

        ##convert the posterior variable back to native model grid
        var_prior = model.read_var(path=path, member=mem_id, **rec)
        var_post = var_prior + c.grid.convert(increment[mem_id, rec_id], is_vector=rec['is_vector'], method='linear')

        ##TODO: temporary solution for nan values due to interpolation
        # ind = np.where(np.isnan(var_post))
        # var_post[ind] = var_prior[ind]
        # if np.isnan(var_post).any():
        #     raise ValueError('nan detected in var_post')

        ##write the posterior variable to restart file
        model.write_var(var_post, path=path, member=mem_id, comm=c.comm, **rec)
