import os
import sys
import importlib.util
from utils.parallel import bcast_by_root
from utils.progress import timer
from utils.shell_utils import run_job, makedir
from utils.dir_def import analysis_dir, cycle_dir
from assim_tools.state import parse_state_info, distribute_state_tasks, partition_grid, prepare_state, output_state, output_ens_mean
from assim_tools.obs import parse_obs_info, distribute_obs_tasks, prepare_obs, prepare_obs_from_state, assign_obs, distribute_partitions
from assim_tools.transpose import transpose_forward, transpose_backward
from assim_tools.inflation import inflation
from assim_tools.update import update_restart

def assimilate(c):
    """
    The core assimilation algorithm
    """
    assert c.nproc==c.comm.Get_size(), f"Error: nproc {c.nproc} not equal to mpi size {c.comm.Get_size()}"

    c.analysis_dir = analysis_dir(c, c.time)
    if c.pid == 0:
        makedir(c.analysis_dir)
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

    c.comm.Barrier()
    state_prior, z_state, lobs, lobs_prior = timer(c)(transpose_forward)(c, fields_prior, z_fields, obs_seq, obs_prior_seq)

    if c.assim_mode == 'batch':
        from assim_tools.batch_assim import batch_assim as assim
    elif c.assim_mode == 'serial':
        from assim_tools.serial_assim import serial_assim as assim
    else:
        raise ValueError(f"Error: assimilation mode {c.assim_mode} not recognized")
    state_post, lobs_post = timer(c)(assim)(c, state_prior, z_state, lobs, lobs_prior)

    c.comm.Barrier()
    fields_post, obs_post_seq = timer(c)(transpose_backward)(c, state_post, lobs_post)

    timer(c)(output_ens_mean)(c, fields_post, os.path.join(c.analysis_dir,'post_mean_state.bin'))

    timer(c)(inflation)(c, 'posterior', fields_prior, os.path.join(c.analysis_dir,'prior_mean_state.bin'), obs_seq, obs_prior_seq, fields_post, os.path.join(c.analysis_dir,'post_mean_state.bin'), obs_post_seq)

    timer(c)(output_state)(c, fields_post, os.path.join(c.analysis_dir,'post_state.bin'))

    timer(c)(update_restart)(c, fields_prior, fields_post)

def run(c):
    """
    Run the assimilate.py script with specified job submit options at runtime
    """
    script_file = os.path.abspath(__file__)
    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    print(f"\033[1;33mRUNNING\033[0m {script_file}")

    ##build run commands for the perturb script
    commands = f"source {c.python_env}; "

    if importlib.util.find_spec("mpi4py") is not None:
        commands += f"JOB_EXECUTE {sys.executable} -m mpi4py {script_file} -c {config_file}"
    else:
        print("Warning: mpi4py is not found, will try to run with nproc=1.", flush=True)
        commands += f"{sys.executable} {script_file} -c {config_file} --nproc=1"

    job_submit_opts = {}
    if c.job_submit:
        job_submit_opts = c.job_submit

    if hasattr(c, 'ppn'):
        job_submit_opts['ppn'] = c.ppn

    run_job(commands, job_name='assimilate', run_dir=cycle_dir(c, c.time), nproc=c.nproc, **job_submit_opts)

if __name__ == '__main__':
    from config import Config
    c = Config(parse_args=True)

    ##multiscale approach: loop over scale components and perform assimilation on each scale
    ##more complex loops can be implemented here
    for c.scale_id in range(c.nscale):
        assimilate(c)

