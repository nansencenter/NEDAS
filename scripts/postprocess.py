import os
import sys
from utils.progress import timer
from utils.parallel import Scheduler
from utils.dir_def import forecast_dir, cycle_dir
from utils.shell_utils import makedir, run_job

def postprocess(c, model_name):
    """
    This function postpares the necessary files for an ensemble forecast
    """
    print(f"\nPostprocessing files for {model_name} ensemble", flush=True)
    model = c.model_config[model_name]
    if c.time==c.time_start:
        restart_dir = model.ens_init_dir
    else:
        restart_dir = forecast_dir(c, c.prev_time, model_name)
    print(f"previous cycle restart files in {restart_dir}", flush=True)

    path = forecast_dir(c, c.time, model_name)
    makedir(path)

    if not c.job_submit:
        c.job_submit = {}

    if c.job_submit.get('run_separate_jobs', False):
        nproc_avail = os.cpu_count()
        nworker = min(c.nens, nproc_avail)
    else:
        nworker = c.nproc // model.nproc_per_util
    scheduler = Scheduler(nworker, debug=c.debug)

    for mem_id in range(c.nens):
        job_name = f'postproc_{model_name}_mem{mem_id+1}'
        job_opt = {
            'restart_dir': restart_dir,
            'path': path,
            'member': mem_id,
            'time': c.time,
            'forecast_period': c.cycle_period,
            'time_start': c.time_start,
            **c.job_submit,
            }
        scheduler.submit_job(job_name, model.postprocess, **job_opt)  ##add job to the queue

    scheduler.start_queue() ##start the job queue
    scheduler.shutdown()
    print(' done.', flush=True)

def run(c):
    script_file = os.path.abspath(__file__)
    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    print(f"\033[1;33mRUNNING\033[0m {script_file}")

    ##build run commands for the postprocess script
    commands = f"source {c.python_env}; "
    commands += f"{sys.executable} {script_file} -c {config_file}"

    job_submit_opts = {}
    if c.job_submit:
        job_submit_opts = c.job_submit

    run_job(commands, job_name="postprocess", run_dir=cycle_dir(c, c.time), nproc=c.nproc, **job_submit_opts)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    for model_name, model in c.model_config.items():
        timer(c)(postprocess)(c, model_name)
