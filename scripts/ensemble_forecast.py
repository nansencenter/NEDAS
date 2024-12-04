import os

from utils.progress import timer
from utils.conversion import t2s
from utils.parallel import Scheduler
from utils.dir_def import forecast_dir
from utils.shell_utils import makedir

def ensemble_forecast_scheduler(c, model_name):
    """
    This function runs ensemble forecasts to advance to the next cycle
    """
    model = c.model_config[model_name]
    path = forecast_dir(c, c.time, model_name)
    makedir(path)
    print(f"\n\033[1;33mRunning {model_name} ensemble forecast\033[0m in {path}", flush=True)

    ##if using external scheduler, c.job_submit['run_separate_jobs']
    ##all jobs will run directly, nworker is nens
    nworker = c.nproc // model.nproc_per_run
    scheduler = Scheduler(nworker, model.walltime, debug=c.debug)

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
    if scheduler.error_jobs:
        raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')
    scheduler.shutdown()
    print(' done.', flush=True)

def ensemble_forecast_batch(c, model_name):
    """
    This functions runs ensemble forecast in one processor
    Requires the model.run to propagate an ensemble state forward in time
    in a simultaneous manner
    """
    model = c.model_config[model_name]
    path = forecast_dir(c, c.time, model_name)
    makedir(path)
    print(f"\n\033[1;33munning {model_name} ensemble forecast\033[0m in {path}", flush=True)

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
    for model_name, model in c.model_config.items():
        if model.ens_run_type == 'batch':
            timer(c)(ensemble_forecast_batch)(c, model_name)
        elif model.ens_run_type == 'scheduler':
            timer(c)(ensemble_forecast_scheduler)(c, model_name)
        else:
            raise TypeError("unknown ensemble forecast type: '"+model.ens_run_type+"' for "+model_name)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    run(c)

