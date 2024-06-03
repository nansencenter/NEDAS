import numpy as np
import os
from utils.parallel import by_rank
from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from utils.progress import print_with_cache, progress_bar
from .state import read_field

def inflate_prior_state(c, fields_prior, inflate_coef=1.0):

    pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in c.rec_list.items() if len(lst)>0][0]
    c.pid_show = pid_rec_show * c.nproc_mem + pid_mem_show
    print = by_rank(c.comm, c.pid_show)(print_with_cache)

    print(f'inflating prior ensemble with inflate_coef={inflate_coef}\n')

    ##process the fields, each processor goes through its own subset of
    ##mem_id,rec_id simultaneously
    nm = len(c.mem_list[c.pid_mem])
    nr = len(c.rec_list[c.pid_rec])

    for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
        rec = c.state_info['fields'][rec_id]

        ##read the prior and post mean field with rec_id
        prior_mean_file = os.path.join(c.work_dir,'cycle',t2s(rec['time']),'analysis','prior_mean_state.bin')
        fld_prior_mean = read_field(prior_mean_file, c.state_info, c.mask, 0, rec_id)

        for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
            if c.debug:
                print(progress_bar(m*nr+r, nm*nr))

            ##inflation by relaxing to prior perturbation
            fld_prior = fields_prior[mem_id, rec_id]
            fld_prior = inflate_coef*(fld_prior-fld_prior_mean) + fld_prior_mean
            fields_prior[mem_id, rec_id] = fld_prior

    if c.debug:
        print(' done.\n')



def adaptive_inflate_coef():
    pass


def adaptive_relax_coef(obs_prior, obs_post, obs, obs_err):
    ####Adaptive covariance relaxation method (Ying and Zhang 2015, QJRMS)
    ###ensures ensemble spread is large enough during cycling
    pass


