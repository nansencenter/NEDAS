import numpy as np
import struct
import importlib
import os
from utils.parallel import by_rank
from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from utils.progress import print_with_cache, progress_bar
from .state import read_field

def update_restart(c, fields_prior, fields_post, relax_coef=0):
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

    grid_bank = {}

    ##process the fields, each processor goes through its own subset of
    ##mem_id,rec_id simultaneously
    nm = len(c.mem_list[c.pid_mem])
    nr = len(c.rec_list[c.pid_rec])

    for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
        rec = c.state_info['fields'][rec_id]

        if relax_coef > 0:
            ##read the prior and post mean field with rec_id
            prior_mean_file = os.path.join(c.work_dir,'cycle',t2s(rec['time']),'analysis','prior_mean_state.bin')
            fld_prior_mean = read_field(prior_mean_file, c.state_info, c.mask, 0, rec_id)
            post_mean_file = os.path.join(c.work_dir,'cycle',t2s(rec['time']),'analysis','prior_mean_state.bin')
            fld_post_mean = read_field(post_mean_file, c.state_info, c.mask, 0, rec_id)

        for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
            if c.debug:
                print(progress_bar(m*nr+r, nm*nr))

            ##directory storing model output
            path = os.path.join(c.work_dir,'cycle',t2s(rec['time']),rec['model_src'])

            ##load the module for handling source model
            # src = importlib.import_module('models.'+rec['model_src'])
            src = c.model_config[rec['model_src']]

            ##only need to generate the uniq grid objs, store them in memory bank
            member = mem_id if 'member' in src.uniq_grid_key else None
            var_name = rec['name'] if 'variable' in src.uniq_grid_key else None
            time = rec['time'] if 'time' in src.uniq_grid_key else None
            k = rec['k'] if 'k' in src.uniq_grid_key else None

            grid_key = (member, rec['model_src'], var_name, time, k)
            if grid_key in grid_bank:
                grid = grid_bank[grid_key]

            else:
                grid = src.read_grid(path=path, member=mem_id, **rec)

            ##grid is the model native grid; c.grid is the analysis grid
            c.grid.set_destination_grid(grid)

            fld_prior = fields_prior[mem_id, rec_id]
            fld_post = fields_post[mem_id, rec_id]

            ##inflation by relaxing to prior perturbation
            if relax_coef > 0:
                fld_post = fld_post_mean + relax_coef*(fld_prior-fld_prior_mean) + (1.-relax_coef)*(fld_post-fld_post_mean)

            ##misc. inverse transform

            ##convert the posterior variable back to native model grid
            var_prior = src.read_var(path=path, member=mem_id, **rec)
            var_post = c.grid.convert(fld_post, is_vector=rec['is_vector'], method='linear')
            # if rec['is_vector']:
            # else:
            #     var_post = c.grid.interp(fld_post, grid.x_elem, grid.y_elem)
                # var_post_coarse = c.grid.convert(fld_post, method='linear', coarse_grain=True)
                # var_post_coarse = np.nanmean(var_post_coarse[grid.tri.triangles], axis=1)

            ##TODO: temporary solution for nan values due to interpolation
            ind = np.where(np.isnan(var_post))
            var_post[ind] = var_prior[ind]

            ##post-processing by model module
            if hasattr(src, 'postproc'):
                var_post = src.postproc(var_post, **rec)

            if np.isnan(var_post).any():
                raise ValueError('nan detected in var_post')

            ##write the posterior variable to restart file
            src.write_var(var_post, path=path, member=mem_id, **rec)

    if c.debug:
        print(' done.\n')

    ##clean up
    # del grid_bank, grid, var_post, fld_post


##alignment technique
def alignment():
    pass


def warp():
    pass


