import numpy as np
import struct
import importlib
import os
from utils.parallel import by_rank
from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from utils.progress import print_with_cache, progress_bar
from .state import read_field

def update_restart(c, fields_prior, fields_post):
    """
    Top-level routine to apply the analysis increments to the original model
    restart files (as initial conditions for the next forecast)

    Inputs:
    - c: config module
    - field_prior, field_post
      the field-complete state variables before and after assimilate()
    """

    pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in c.rec_list.items() if len(lst)>0][0]
    c.pid_show = pid_rec_show * c.nproc_mem + pid_mem_show
    print = by_rank(c.comm, c.pid_show)(print_with_cache)

    if c.debug:
        print('update model restart files with analysis increments\n')

    ##process the fields, each processor goes through its own subset of
    ##mem_id,rec_id simultaneously
    nm = len(c.mem_list[c.pid_mem])
    nr = len(c.rec_list[c.pid_rec])

    for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
        rec = c.state_info['fields'][rec_id]

        for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
            if c.debug:
                print(progress_bar(m*nr+r, nm*nr))

            ##directory storing model output
            path = os.path.join(c.work_dir,'cycle',t2s(rec['time']),rec['model_src'])

            ##load the module for handling source model
            model = c.model_config[rec['model_src']]
            model.read_grid(path=path, member=mem_id, **rec)
            c.grid.set_destination_grid(model.grid)

            fld_prior = fields_prior[mem_id, rec_id]
            fld_post = fields_post[mem_id, rec_id]

            ##deal with analysis increment
            ##misc. inverse transform

            ##convert the posterior variable back to native model grid
            var_prior = model.read_var(path=path, member=mem_id, **rec)
            var_post = c.grid.convert(fld_post, is_vector=rec['is_vector'], method='linear')

            ##TODO: temporary solution for nan values due to interpolation
            ind = np.where(np.isnan(var_post))
            var_post[ind] = var_prior[ind]

            ##post-processing by model module
            if hasattr(model, 'postproc'):
                var_post = model.postproc(var_post, **rec)

            if np.isnan(var_post).any():
                raise ValueError('nan detected in var_post')

            ##write the posterior variable to restart file
            model.write_var(var_post, path=path, member=mem_id, **rec)

    if c.debug:
        print(' done.\n')


##alignment technique
def alignment():
    pass


def warp():
    pass


