import numpy as np
import os
import time
from config import Config
from utils.conversion import t2s, s2t, dt1h
from utils.parallel import bcast_by_root
from utils.progress import timer
from assim_tools.state import parse_state_info, distribute_state_tasks, partition_grid, prepare_state, output_state, output_ens_mean
from assim_tools.obs import parse_obs_info, distribute_obs_tasks, prepare_obs, prepare_obs_from_state, assign_obs, distribute_partitions
from assim_tools.transpose import transpose_forward, transpose_backward
from assim_tools.analysis import analysis
from assim_tools.inflation import inflation
from assim_tools.update import update_restart

def assimilate(c):
    ##the algorithm can be iterated over several scale components
    ##for s = 0, ..., nscale
    ##if nscale = 1, then just
    c.s_dir = ''

    analysis_dir = os.path.join(c.work_dir,'cycle', t2s(c.time), 'analysis', c.s_dir)
    if c.pid == 0:
        os.system("mkdir -p "+analysis_dir)
        print("running assimilation in "+analysis_dir, flush=True)

    c.state_info = bcast_by_root(c.comm)(parse_state_info)(c)
    c.mem_list, c.rec_list = bcast_by_root(c.comm)(distribute_state_tasks)(c)
    c.partitions = bcast_by_root(c.comm)(partition_grid)(c)
    fields_prior, z_fields = timer(c)(prepare_state)(c)

    # # if c.pid_rec == 0:
    # #     print('mem', c.pid_mem, mem_list[c.pid_mem])
    # # if c.pid_mem == 0:
    # #     print('rec', c.pid_rec, rec_list[c.pid_rec])

    # if c.debug:
    #     np.save(analysis_dir+'/fields_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), fields_prior)

    timer(c)(output_state)(c, fields_prior, os.path.join(analysis_dir,'prior_state.bin'))
    timer(c)(output_ens_mean)(c, fields_prior, os.path.join(analysis_dir,'prior_mean_state.bin'))
    timer(c)(output_ens_mean)(c, z_fields, os.path.join(analysis_dir,'z_coords.bin'))

    c.obs_info = bcast_by_root(c.comm)(parse_obs_info)(c)
    c.obs_rec_list = bcast_by_root(c.comm)(distribute_obs_tasks)(c)
    c.obs_info, obs_seq = timer(c)(bcast_by_root(c.comm_mem)(prepare_obs))(c)

    if c.pid_mem == 0:
        np.save(analysis_dir+'/obs_seq.{}.npy'.format(c.pid_rec), obs_seq)

    c.obs_inds = bcast_by_root(c.comm_mem)(assign_obs)(c, obs_seq)
    c.par_list = bcast_by_root(c.comm)(distribute_partitions)(c)

    # if c.pid == 0 and c.debug:
    #     np.save(analysis_dir+'/obs_inds.npy', obs_inds)
    #     np.save(analysis_dir+'/partitions.npy', partitions)
    #     np.save(analysis_dir+'/par_list.npy', par_list)

    obs_prior_seq = timer(c)(prepare_obs_from_state)(c, obs_seq, fields_prior, z_fields)

    # if c.debug:
    #     np.save(analysis_dir+'/obs_prior_seq.{}.{}.npy'.format(c.pid_mem, c.pid_rec), obs_prior_seq)

    inflation(c, 'prior', fields_prior, os.path.join(analysis_dir,'prior_mean_state.bin'), obs_seq, obs_prior_seq)

    c.comm.Barrier()
    state_prior, z_state, lobs, lobs_prior = transpose_forward(c, fields_prior, z_fields, obs_seq, obs_prior_seq)

    # if c.debug:
    #     np.save(analysis_dir+'/state_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), state_prior)
    #     np.save(analysis_dir+'/z_state.{}.{}.npy'.format(c.pid_mem, c.pid_rec), z_state)
    #     np.save(analysis_dir+'/lobs.{}.{}.npy'.format(c.pid_mem, c.pid_rec), lobs)
    #     np.save(analysis_dir+'/lobs_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), lobs_prior)

    state_post = timer(c)(analysis)(c, state_prior, z_state, lobs, lobs_prior)

    c.comm.Barrier()
    fields_post = transpose_backward(c, state_post)

    timer(c)(output_ens_mean)(c, fields_post, os.path.join(analysis_dir,'post_mean_state.bin'))

    obs_post_seq = timer(c)(prepare_obs_from_state)(c, obs_seq, fields_post, z_fields)

    inflation(c, 'posterior', fields_prior, os.path.join(analysis_dir,'prior_mean_state.bin'), obs_seq, obs_prior_seq, fields_post, os.path.join(analysis_dir,'post_mean_state.bin'), obs_post_seq)

    timer(c)(output_state)(c, fields_post, os.path.join(analysis_dir,'post_state.bin'))

    timer(c)(update_restart)(c, fields_prior, fields_post)


if __name__ == '__main__':
    from config import Config
    from utils.progress import timer
    c = Config(parse_args=True)

    assert c.nproc==c.comm.Get_size(), f"Error: nproc {c.nproc} not equal to mpi size {c.comm.Get_size()}"

    timer(c)(assimilate)(c)

