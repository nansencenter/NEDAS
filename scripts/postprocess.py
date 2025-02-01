import os

from utils.progress import timer
from utils.parallel import Scheduler
from utils.dir_def import forecast_dir
from utils.shell_utils import makedir

def postprocess(c, model_name):
    """
    This function prepares the necessary files for an ensemble forecast
    """
    ##copy the necessary files for each model
    print(f"\n\033[1;33mPostprocessing files for {model_name} ensemble\033[0m", flush=True)
    model = c.model_config[model_name]
    path = forecast_dir(c, c.time, model_name)
    makedir(path)

    if c.job_submit.get('run_separate_jobs', False):
        nproc_avail = os.cpu_count()
        nworker = min(c.nens, nproc_avail)
    else:
        nworker = c.nproc // model.nproc_per_util
    scheduler = Scheduler(nworker, debug=c.debug)

    for mem_id in range(c.nens):
        job_name = f'postproc_{model_name}_mem{mem_id+1}'
        job_opt = {
            'path': path,
            'member': mem_id,
            'time': c.time,
            'forecast_period': c.cycle_period,
            'time_start': c.time_start,
            'time_end': c.time_end,
            'debug': c.debug,
            **c.job_submit,
            }
        scheduler.submit_job(job_name, model.postprocess, **job_opt)  ##add job to the queue

    scheduler.start_queue() ##start the job queue
    if scheduler.error_jobs:
        raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')
    scheduler.shutdown()
    print(' done.', flush=True)

def run(c):
    for model_name, model in c.model_config.items():
        timer(c)(postprocess)(c, model_name)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    run(c)

