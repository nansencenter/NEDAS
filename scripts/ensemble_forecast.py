import os
import sys
import subprocess
from utils.progress import timer
from utils.conversion import t2s
from utils.parallel import Scheduler
from utils.shell_utils import makedir, run_job

def ensemble_forecast_scheduler(c, model_name):
    """
    This function runs ensemble forecasts to advance to the next cycle
    """
    model = c.model_config[model_name]
    path = c.forecast_dir(c.time, model_name)
    makedir(path)
    print(f"\nRunning {model_name} ensemble forecast in {path}", flush=True)

    if not c.job_submit:
        c.job_submit = {}

    if c.job_submit.get('run_separate_jobs', False):
        ##all jobs will be submitted to external scheduler's queue
        ##just assign a worker to each ensemble member
        nworker = c.nens
    else:
        ##Scheduler will use nworkers to spawn ensemble member runs to
        ##the available nproc processors
        nworker = c.nproc // model.nproc_per_run
    if hasattr(model, 'walltime'):
        walltime = model.walltime
    else:
        walltime = None
    scheduler = Scheduler(nworker, walltime, debug=c.debug)

    for mem_id in range(c.nens):
        job_name = f'forecast_{model_name}_mem{mem_id+1}'

        job_opt = {
            'path': path,
            'member': mem_id,
            'time': c.time,
            'forecast_period': c.cycle_period,
            **c.job_submit,
            }
        scheduler.submit_job(job_name, model.run, **job_opt)  ##add job to the queue

    scheduler.start_queue() ##start the job queue
    scheduler.shutdown()
    print(' done.', flush=True)

def ensemble_forecast_batch(c, model_name):
    """
    This functions runs ensemble forecast in one processor
    Requires the model.run to propagate an ensemble state forward in time
    in a simultaneous manner
    """
    model = c.model_config[model_name]
    path = c.forecast_dir(c.time, model_name)
    makedir(path)
    print(f"\nRunning {model_name} ensemble forecast in {path}", flush=True)

    job_opt = {
        'path': path,
        'nens': c.nens,
        'time': c.time,
        'forecast_period': c.cycle_period,
        **c.job_submit,
        }
    model.run_batch(**job_opt)
    print('done.', flush=True)

def run(c):
    script_file = os.path.abspath(__file__)
    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    print(f"\033[1;33mRUNNING\033[0m {script_file}")

    ##build run commands for the ensemble forecast script
    commands = f"source {c.python_env}; "
    commands += f"{sys.executable} {script_file} -c {config_file}"

    job_submit_opts = {}
    if c.job_submit:
        job_submit_opts = c.job_submit

    #run_job(commands, job_name="ensemble_forecast", run_dir=cycle_dir(c, c.time), nproc=c.nproc, **job_submit_opts)
    subprocess.run(commands, shell=True, stdout=sys.stdout, stderr=sys.stderr)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    for model_name, model in c.model_config.items():
        if model.ens_run_type == 'batch':
            timer(c)(ensemble_forecast_batch)(c, model_name)
        elif model.ens_run_type == 'scheduler':
            timer(c)(ensemble_forecast_scheduler)(c, model_name)
        else:
            raise TypeError("unknown ensemble forecast type: '"+model.ens_run_type+"' for "+model_name)
