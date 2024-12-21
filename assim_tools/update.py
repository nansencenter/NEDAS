import numpy as np
import struct
import os
from utils.parallel import by_rank
from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from utils.progress import print_with_cache, progress_bar
from utils.dir_def import forecast_dir
from .state import read_field
from .alignment import alignment

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
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)

    print_1p(f'>>> update model restart files with analysis increments\n')

    if c.run_alignment and c.scale_id < c.nscale:
        alignment(c, fields_prior, fields_post, **c.alignment)

    ##process the fields, each processor goes through its own subset of
    ##mem_id,rec_id simultaneously
    ##but need to keep every rank in sync to coordinate multiprocess file access
    nm_max = np.max([len(lst) for _,lst in c.mem_list.items()])
    nr_max = np.max([len(lst) for _,lst in c.rec_list.items()])
    for r in range(nr_max):
        for m in range(nm_max):
            pid_active = ( m < len(c.mem_list[c.pid_mem]) and r < len(c.rec_list[c.pid_rec]) )
            if pid_active:
                mem_id = c.mem_list[c.pid_mem][m]
                rec_id = c.rec_list[c.pid_rec][r]
                rec = c.state_info['fields'][rec_id]
                path = forecast_dir(c, rec['time'], rec['model_src'])
                model = c.model_config[rec['model_src']]
                model.read_grid(path=path, member=mem_id, **rec)
                file = model.filename(path=path, member=mem_id, **rec)
            else:
                file = None
            all_files = c.comm.allgather(file)
            for file in all_files:
                c.comm.init_file_lock(file)
            
            if pid_active:                
                if c.debug:
                    print(f"PID {c.pid}: update_restart mem{mem_id+1:03d} '{rec['name']}' {rec['time']} k={rec['k']}", flush=True)
                else:
                    print_1p(progress_bar(m*nr_max+r, nm_max*nr_max))

                c.grid.set_destination_grid(model.grid)

                fld_prior = fields_prior[mem_id, rec_id]
                fld_post = fields_post[mem_id, rec_id]

                ##analysis increment
                fld_incr = fld_post - fld_prior

                ##misc. inverse transform
                ##e.g. multiscale approach: just use the analysis increment directly

                ##convert the posterior variable back to native model grid
                var_prior = model.read_var(path=path, member=mem_id, **rec)
                var_post = var_prior + c.grid.convert(fld_incr, is_vector=rec['is_vector'], method='linear')

                ##post-processing by model module
                # if hasattr(model, 'postproc'):
                #     var_post = model.postproc(var_post, **rec)

                ##TODO: temporary solution for nan values due to interpolation
                # ind = np.where(np.isnan(var_post))
                # var_post[ind] = var_prior[ind]
                # if np.isnan(var_post).any():
                #     raise ValueError('nan detected in var_post')

                ##write the posterior variable to restart file
                model.write_var(var_post, path=path, member=mem_id, comm=c.comm, **rec)

    c.comm.Barrier()
    c.comm.cleanup_file_locks()
    print_1p(' done.\n')

