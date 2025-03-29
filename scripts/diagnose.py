import os
import sys
import importlib
import importlib.util
from utils.conversion import dt1h, ensure_list
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.progress import timer, print_with_cache, progress_bar
from utils.shell_utils import run_job

def diagnose(c):
    """Perform diagnostic methods collectively by mpi ranks"""
    assert c.nproc==c.comm.Get_size(), f"Error: nproc {c.nproc} not equal to mpi size {c.comm.Get_size()}"

    ##get task list for each rank
    task_list = bcast_by_root(c.comm)(distribute_diag_tasks)(c)

    ##the processor with most work load will show progress messages
    c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]

    ##init file locks for collective i/o
    init_file_locks(c)

    ntask = len(task_list[c.pid])
    for task_id, rec in enumerate(task_list[c.pid]):
        if c.debug:
            print(f"PID {c.pid:4} running diagnostics '{rec['method']}'", flush=True)
        else:
            c.print_1p(progress_bar(task_id, ntask))

        method_name = f"diag.{rec['method']}"
        mod = importlib.import_module(method_name)

        ##perform the diag task
        mod.run(c, **rec)

    c.comm.Barrier()
    print_1p(' done.\n\n')
    c.comm.cleanup_file_locks()

def distribute_diag_tasks(c):
    """Build the full task list for the diagnostics part of the config"""
    task_list_full = []
    for rec in ensure_list(c.diag):
        ##load the module for the given method
        method_name = f"diag.{rec['method']}"
        mod = importlib.import_module(method_name)
        ##module returns a list of tasks to be done by each processor
        if not hasattr(mod, 'get_task_list'):
            task_list_full.append(rec)
            continue
        task_list_rec = mod.get_task_list(c, **rec)
        for task in task_list_rec:
            task_list_full.append(task)
    ##collected full list of tasks is evenly distributed across the mpi communicator
    task_list = distribute_tasks(c.comm, task_list_full)
    return task_list

def init_file_locks(c):
    """Build the full task list for the diagnostics part of the config"""
    for rec in ensure_list(c.diag):
        ##load the module for the given method
        method_name = f"diag.{rec['method']}"
        mod = importlib.import_module(method_name)
        ##module get_file_list returns a list of files for collective i/o
        if not hasattr(mod, 'get_file_list'):
            continue
        files = mod.get_file_list(c, **rec)
        for file in files:
            ##create the file lock across mpi ranks for this file
            c.comm.init_file_lock(file)

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

    run_job(commands, job_name="diag", run_dir=c.cycle_dir(c.time), nproc=c.nproc, **job_submit_opts)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    print_1p = by_rank(c.comm, 0)(print_with_cache)
    print_1p('\nRunning diagnostics\n')
    if not hasattr(c, 'diag') or c.diag is None:
        print_1p('no diagnostic defined in config, exiting\n\n')
        exit()

    timer(c)(diagnose)(c)
