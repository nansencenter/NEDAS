import numpy as np
import os
import sys
import importlib.util
from utils.conversion import dt1h, ensure_list
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.progress import timer, print_with_cache, progress_bar
from utils.dir_def import forecast_dir, analysis_dir
from utils.shell_utils import run_job

def diag(c):
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    print_1p('\n\033[1;33mRunning diagnostics \033[0m\n')

def run(c):
    script_file = os.path.abspath(__file__)
    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    if importlib.util.find_spec("mpi4py") is not None:
        commands = f"JOB_EXECUTE {sys.executable} -m mpi4py {script_file} -c {config_file}"
    else:
        print("Warning: mpi4py is not found, will try to run with nproc=1.", flush=True)
        commands = f"{sys.executable} {script_file} -c {config_file} --nproc=1"

    run_job(commands, job_name="diag", nproc=c.nproc, **c.job_submit)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    timer(c)(diag)(c)

