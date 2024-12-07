import os
import sys
import importlib.util
from utils.conversion import dt1h, ensure_list
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.progress import timer, print_with_cache, progress_bar
from utils.dir_def import forecast_dir, analysis_dir
from utils.shell_utils import run_job

def diag(c):
    task_list = bcast_by_root(c.comm)(distribute_diag_tasks)(c)

    c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)

    diag_id = 0
    for rec in task_list[c.pid]:
        print(rec)

    c.comm.Barrier()
    print_1p(' done.\n\n')

def distribute_diag_tasks(c):
    task_list_full = []
    for rec in ensure_list(c.diag):
        if rec['ensemble']:
            task_list_full.append(rec)
        else:
            for mem_id in range(c.nens):
                task_list_full.append({**rec, 'member':mem_id})
    task_list = distribute_tasks(c.comm, task_list_full)
    return task_list

def run(c):
    script_file = os.path.abspath(__file__)
    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    if importlib.util.find_spec("mpi4py") is not None:
        commands = f"JOB_EXECUTE {sys.executable} -m mpi4py {script_file} -c {config_file}"
    else:
        print("Warning: mpi4py is not found, will try to run with nproc=1.", flush=True)
        commands = f"{sys.executable} {script_file} -c {config_file} --nproc=1"

    run_job(commands, job_name="diag", run_dir=c.work_dir, nproc=c.nproc, **c.job_submit)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    print_1p = by_rank(c.comm, 0)(print_with_cache)
    print_1p('\n\033[1;33mRunning diagnostics \033[0m\n')
    if not hasattr(c, 'diag') or c.diag is None:
        print_1p('no diagnostic defined in config\n\n')
        exit()

    timer(c)(diag)(c)

