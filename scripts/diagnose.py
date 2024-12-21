import os
import sys
import importlib
import importlib.util
from utils.conversion import dt1h, ensure_list
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.progress import timer, print_with_cache, progress_bar
from utils.dir_def import cycle_dir
from utils.shell_utils import run_job

def diagnose(c):
    assert c.nproc==c.comm.Get_size(), f"Error: nproc {c.nproc} not equal to mpi size {c.comm.Get_size()}"

    task_list = bcast_by_root(c.comm)(distribute_diag_tasks)(c)

    c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)

    ntask = len(task_list[c.pid])
    for task_id, rec in enumerate(task_list[c.pid]):
        if c.debug:
            print(f"PID {c.pid:4} running diagnostics '{rec['method']}'", flush=True)
        else:
            print_1p(progress_bar(task_id, ntask))

        method_name = f"diag.{rec['method']}"
        module = importlib.import_module(method_name)
        module.run(c, **rec)        

    print_1p(' done.\n\n')

def distribute_diag_tasks(c):
    task_list_full = []
    for rec in ensure_list(c.diag):
        if rec['is_ensemble']:
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

    print(f"\033[1;33mRUNNING\033[0m {script_file}")
    if not hasattr(c, 'diag') or c.diag is None:
        print("no diagnostic defined in config, exiting")
        return

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

    run_job(commands, job_name="diag", run_dir=cycle_dir(c, c.time), nproc=c.nproc, **job_submit_opts)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    print_1p = by_rank(c.comm, 0)(print_with_cache)
    print_1p('\nRunning diagnostics\n')
    if not hasattr(c, 'diag') or c.diag is None:
        print_1p('no diagnostic defined in config, exiting\n\n')
        exit()

    timer(c)(diagnose)(c)
