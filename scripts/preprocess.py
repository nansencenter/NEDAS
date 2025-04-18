import os

from utils.progress import timer
from utils.conversion import t2s
from utils.parallel import Scheduler
from utils.dir_def import forecast_dir

def preprocess_model_state(c, model_name):
    """
    This function prepares the necessary files for an ensemble forecast
    """
    print(f"\n\033[1;33mPreprocessing {model_name} ensemble members\033[0m", flush=True)
    model = c.model_config[model_name]
    if c.time==c.time_start:
        restart_dir = model.ens_init_dir
    else:
        restart_dir = forecast_dir(c, c.prev_time, model_name)
    print(f"using restart files in {restart_dir}", flush=True)

    scheduler = Scheduler(c.nproc // model.nproc_per_util, debug=c.debug)

    for mem_id in range(c.nens):
        job_name = f'preproc_{model_name}_mem{mem_id+1}'
        job_opt = {
            'job_submit_cmd': c.job_submit_cmd,
            'task_nproc': model.nproc_per_util,
            'restart_dir': restart_dir,
            'path': forecast_dir(c, c.time, model_name),
            'member': mem_id,
            'time': c.time,
            'forecast_period': c.cycle_period,
            }
        scheduler.submit_job(job_name, model.preprocess, **job_opt)  ##add job to the queue

    scheduler.start_queue() ##start the job queue
    if scheduler.error_jobs:
        raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')
    scheduler.shutdown()
    print(' done.', flush=True)

def preprocess(c):
    for model_name, model in c.model_config.items():
        timer(c)(preprocess_model_state)(c, model_name)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    preprocess(c)

