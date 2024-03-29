import numpy as np
import struct
import importlib
import sys
from datetime import datetime, timedelta
from conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from log import message, show_progress

def update_restart(c, state_info, mem_list, rec_list, fields_prior, fields_post):
    """
    Top-level routine to apply the analysis increments to the original model
    restart files (as initial conditions for the next forecast)

    Inputs:
    - c: config module
    - state_info: from parse_state_info()
    - mem_list, rec_list: from build_state_tasks()
    - field_prior, field_post
      the field-complete state variables before and after assimilate()
    """

    pid_mem_show = [p for p,lst in mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in rec_list.items() if len(lst)>0][0]
    c.pid_show = pid_rec_show * c.nproc_mem + pid_mem_show

    message(c.comm, 'update model restart files with analysis increments\n', c.pid_show)

    grid_bank = {}

    ##process the fields, each processor goes through its own subset of
    ##mem_id,rec_id simultaneously
    nm = len(mem_list[c.pid_mem])
    nr = len(rec_list[c.pid_rec])

    for m, mem_id in enumerate(mem_list[c.pid_mem]):
        for r, rec_id in enumerate(rec_list[c.pid_rec]):
            show_progress(c.comm, m*nr+r, nm*nr, c.pid_show)

            rec = state_info['fields'][rec_id]

            ##directory storing model output
            path = c.work_dir+'/cycle/'+t2s(rec['time'])+'/'+rec['source']

            ##load the module for handling source model
            src = importlib.import_module('models.'+rec['source'])

            ##only need to generate the uniq grid objs, store them in memory bank
            member = mem_id if 'member' in src.uniq_grid_key else None
            var_name = rec['name'] if 'variable' in src.uniq_grid_key else None
            time = rec['time'] if 'time' in src.uniq_grid_key else None
            k = rec['k'] if 'k' in src.uniq_grid_key else None

            grid_key = (member, rec['source'], var_name, time, k)
            if grid_key in grid_bank:
                grid = grid_bank[grid_key]

            else:
                grid = src.read_grid(path, member=mem_id, **rec)

            ##grid is the model native grid; c.grid is the analysis grid
            c.grid.set_destination_grid(grid)

            fld_post = fields_post[mem_id, rec_id]

            ##misc. inverse transform

            ##convert the posterior variable back to native model grid
            var_prior = src.read_var(path, grid, member=mem_id, **rec)
            if rec['is_vector']:
                var_post = c.grid.convert(fld_post, is_vector=rec['is_vector'], method='linear')
            else:
                var_post = c.grid.interp(fld_post, grid.x_elem, grid.y_elem)
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
            src.write_var(path, grid, var_post, member=mem_id, **rec)

    message(c.comm, ' done.\n', c.pid_show)

    ##clean up
    # del grid_bank, grid, var_post, fld_post


##alignment technique
def alignment():
    pass


def warp():
    pass


