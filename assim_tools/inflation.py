import numpy as np
import os
from utils.parallel import by_rank
from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from utils.progress import print_with_cache, progress_bar
from .state import read_field

def inflate_state(c, fields, mean_file):

    pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in c.rec_list.items() if len(lst)>0][0]
    c.pid_show = pid_rec_show * c.nproc_mem + pid_mem_show
    print = by_rank(c.comm, c.pid_show)(print_with_cache)

    print(f'inflating ensemble with inflate_coef={c.inflate_coef}\n')

    ##process the fields, each processor goes through its own subset of
    ##mem_id,rec_id simultaneously
    nm = len(c.mem_list[c.pid_mem])
    nr = len(c.rec_list[c.pid_rec])

    for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
        rec = c.state_info['fields'][rec_id]

        ##read the mean field with rec_id
        fields_mean = read_field(mean_file, c.state_info, c.mask, 0, rec_id)

        for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
            if c.debug:
                print(progress_bar(m*nr+r, nm*nr))

            ##inflate the ensemble perturbations by c.inflate_coef
            fields[mem_id, rec_id] = c.inflate_coef*(fields[mem_id, rec_id] - fields_mean) + fields_mean

    if c.debug:
        print(' done.\n')


def adaptive_prior_inflation(c, obs_seq, obs_prior_seq):
    ##compute inflate coef by obs-space statistics (Desroziers et al. 2005)
    print = by_rank(c.comm, c.pid_show)(print_with_cache)

    ##collect obs-space statistics
    total_nobs = 0
    omb2 = 0.0  ##obs-minus-background differences squared
    varo = 0.0  ##obs err variance
    varb = 0.0  ##obs_prior (background) ensemble variances

    ##go through each obs record
    for r, obs_rec_id in enumerate(c.obs_rec_list[c.pid_rec]):
        obs_rec = c.obs_info['records'][obs_rec_id]
        nobs = obs_rec['nobs']

        ##1. get ensemble mean obs_prior:
        if obs_rec['is_vector']:
            nv = 2
            shape = (nv, nobs)
        else:
            nv = 1
            shape = (nobs,)

        ##sum over all obs_prior_seq locally stored on pid
        sum_obs_prior_pid = np.zeros(shape)
        for mem_id in c.mem_list[c.pid_mem]:
            sum_obs_prior_pid += obs_prior_seq[mem_id, obs_rec_id]
        ##sum over all obs_prior_seq on differnet pids to get the total sum
        sum_obs_prior = c.comm_mem.allreduce(sum_obs_prior_pid)
        mean_obs_prior = sum_obs_prior / c.nens

        ##2. get ensemble spread obs_prior:
        pert2_obs_prior_pid = np.zeros(shape)
        for mem_id in c.mem_list[c.pid_mem]:
            pert2_obs_prior_pid += (obs_prior_seq[mem_id, obs_rec_id] - mean_obs_prior)**2
        pert2_obs_prior = c.comm_mem.allreduce(pert2_obs_prior_pid)
        variance_obs_prior = pert2_obs_prior / (c.nens - 1)

        obs = obs_seq[obs_rec_id]['obs']
        total_nobs += nv * nobs
        omb2 += np.sum((obs - mean_obs_prior)**2)
        varo += np.sum(obs_seq[obs_rec_id]['err_std']**2) * nv
        varb += np.sum(variance_obs_prior)

    print("adaptive prior inflation:\n")
    print(f"varb = {varb/total_nobs}, varo={varo/total_nobs}\n")
    print(f"omb2 = {omb2/total_nobs}\n")
    inflate_coef = np.sqrt(max(0.1, (omb2 - varo) / varb))
    return inflate_coef


def adaptive_post_inflation(c, obs_seq, obs_prior_seq, obs_post_seq):
    ##compute inflate coef by obs-space statistics (Desroziers et al. 2005)
    print = by_rank(c.comm, c.pid_show)(print_with_cache)

    ##collect obs-space statistics
    total_nobs = 0
    omb2 = 0.0  ##obs-minus-background differences squared
    oma2 = 0.0
    amb2 = 0.0  ##analysis-minus-background diff squared
    varo = 0.0  ##obs err variance
    varb = 0.0  ##obs_prior (background) ensemble variances
    vara = 0.0  ##obs_post (analysis) ensemble variances

    ##go through each obs record
    for r, obs_rec_id in enumerate(c.obs_rec_list[c.pid_rec]):
        obs_rec = c.obs_info['records'][obs_rec_id]
        nobs = obs_rec['nobs']

        ##1. get ensemble mean obs_prior:
        if obs_rec['is_vector']:
            nv = 2
            shape = (nv, nobs)
        else:
            nv = 1
            shape = (nobs,)

        ##sum over all obs_prior_seq locally stored on pid
        sum_obs_prior_pid = np.zeros(shape)
        sum_obs_post_pid = np.zeros(shape)
        for mem_id in c.mem_list[c.pid_mem]:
            sum_obs_prior_pid += obs_prior_seq[mem_id, obs_rec_id]
            sum_obs_post_pid += obs_post_seq[mem_id, obs_rec_id]
        ##sum over all obs_prior_seq on differnet pids to get the total sum
        sum_obs_prior = c.comm_mem.allreduce(sum_obs_prior_pid)
        sum_obs_post = c.comm_mem.allreduce(sum_obs_post_pid)
        mean_obs_prior = sum_obs_prior / c.nens
        mean_obs_post = sum_obs_post / c.nens

        ##2. get ensemble spread obs_prior:
        pert2_obs_prior_pid = np.zeros(shape)
        pert2_obs_post_pid = np.zeros(shape)
        for mem_id in c.mem_list[c.pid_mem]:
            pert2_obs_prior_pid += (obs_prior_seq[mem_id, obs_rec_id] - mean_obs_prior)**2
            pert2_obs_post_pid += (obs_post_seq[mem_id, obs_rec_id] - mean_obs_post)**2
        pert2_obs_prior = c.comm_mem.allreduce(pert2_obs_prior_pid)
        pert2_obs_post = c.comm_mem.allreduce(pert2_obs_post_pid)
        variance_obs_prior = pert2_obs_prior / (c.nens - 1)
        variance_obs_post = pert2_obs_post / (c.nens - 1)

        obs = obs_seq[obs_rec_id]['obs']
        total_nobs += nv * nobs
        omb2 += np.sum((obs - mean_obs_prior)**2)
        oma2 += np.sum((obs - mean_obs_post)**2)
        amb2 += np.sum((mean_obs_post - mean_obs_prior)**2)
        varo += np.sum(obs_seq[obs_rec_id]['err_std']**2) * nv
        varb += np.sum(variance_obs_prior)
        vara += np.sum(variance_obs_post)
        print(f"{obs_rec,obs.shape, total_nobs, mean_obs_prior.shape, variance_obs_prior.shape, obs_seq[obs_rec_id]['err_std'].shape}\n")

    print("adaptive posterior inflation:\n")
    print(f"varb = {varb/total_nobs}, vara = {vara/total_nobs}, varo={varo/total_nobs}\n")
    print(f"omb2 = {omb2/total_nobs}, oma2 = {oma2/total_nobs}, amb2={amb2/total_nobs}\n")
    beta = np.sqrt(vara/varb)
    print(f"spread reduction ratio = {np.sqrt(beta)}\n")

    inflate_coef = np.sqrt(max(0.1, (omb2-varo-amb2)/vara))
    # inflate_coef = np.sqrt(max(0.1, (oma2-varo)/vara))
    return inflate_coef


def adaptive_relax_coef(obs_prior, obs_post, obs, obs_err):
    ####Adaptive covariance relaxation method (Ying and Zhang 2015, QJRMS)
    ###ensures ensemble spread is large enough during cycling
    pass


