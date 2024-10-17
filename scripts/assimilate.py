import numpy as np
import os
import sys
import subprocess
from utils.conversion import t2s, s2t, dt1h
from utils.parallel import bcast_by_root
from utils.progress import timer
from utils.dir_def import analysis_dir
from assim_tools.state import parse_state_info, distribute_state_tasks, partition_grid, prepare_state, output_state, output_ens_mean
from assim_tools.obs import parse_obs_info, distribute_obs_tasks, prepare_obs, prepare_obs_from_state, assign_obs, distribute_partitions
from assim_tools.transpose import transpose_forward, transpose_backward
from assim_tools.analysis import analysis
from assim_tools.inflation import inflation
from assim_tools.update import update_restart

assimilate_script_path = os.path.abspath(__file__)

def assimilate(c):
    assert c.nproc==c.comm.Get_size(), f"Error: nproc {c.nproc} not equal to mpi size {c.comm.Get_size()}"

    c.analysis_dir = analysis_dir(c, c.time)
    if c.pid == 0:
        os.system("mkdir -p "+c.analysis_dir)
        print(f"\nRunning assimilation step in {c.analysis_dir}\n", flush=True)

    c.state_info = bcast_by_root(c.comm)(parse_state_info)(c)
    c.mem_list, c.rec_list = bcast_by_root(c.comm)(distribute_state_tasks)(c)
    c.partitions = bcast_by_root(c.comm)(partition_grid)(c)
    fields_prior, z_fields = timer(c)(prepare_state)(c)

    timer(c)(output_state)(c, fields_prior, os.path.join(c.analysis_dir,'prior_state.bin'))
    timer(c)(output_ens_mean)(c, fields_prior, os.path.join(c.analysis_dir,'prior_mean_state.bin'))
    timer(c)(output_ens_mean)(c, z_fields, os.path.join(c.analysis_dir,'z_coords.bin'))

    c.obs_info = bcast_by_root(c.comm)(parse_obs_info)(c)
    c.obs_rec_list = bcast_by_root(c.comm)(distribute_obs_tasks)(c)
    c.obs_info, obs_seq = timer(c)(bcast_by_root(c.comm_mem)(prepare_obs))(c)

    c.obs_inds = bcast_by_root(c.comm_mem)(assign_obs)(c, obs_seq)
    c.par_list = bcast_by_root(c.comm)(distribute_partitions)(c)

    obs_prior_seq = timer(c)(prepare_obs_from_state)(c, obs_seq, fields_prior, z_fields)

    inflation(c, 'prior', fields_prior, os.path.join(c.analysis_dir,'prior_mean_state.bin'), obs_seq, obs_prior_seq)

    state_prior, z_state, lobs, lobs_prior = timer(c)(transpose_forward)(c, fields_prior, z_fields, obs_seq, obs_prior_seq)

    state_post, lobs_post = timer(c)(analysis)(c, state_prior, z_state, lobs, lobs_prior)

    fields_post, obs_post_seq = timer(c)(transpose_backward)(c, state_post, lobs_post)

    timer(c)(output_ens_mean)(c, fields_post, os.path.join(c.analysis_dir,'post_mean_state.bin'))

    timer(c)(inflation)(c, 'posterior', fields_prior, os.path.join(c.analysis_dir,'prior_mean_state.bin'), obs_seq, obs_prior_seq, fields_post, os.path.join(c.analysis_dir,'post_mean_state.bin'), obs_post_seq)

    timer(c)(output_state)(c, fields_post, os.path.join(c.analysis_dir,'post_state.bin'))

    timer(c)(update_restart)(c, fields_prior, fields_post)

if __name__ == '__main__':
    from config import Config
    from utils.progress import timer
    c = Config(parse_args=True)

    assimilate(c)

