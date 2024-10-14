import os

from utils.progress import timer
from utils.conversion import t2s
from utils.parallel import Scheduler
from utils.dir_def import forecast_dir

def postprocess_model_state(c, model_name):
    """
    This function prepares the necessary files for an ensemble forecast
    """
    ##copy the necessary files for each model
    print(f"\nPostprocessing files for {model_name} ensemble", flush=True)
    model = c.model_config[model_name]
    path = forecast_dir(c, c.time, model_name)

    scheduler = Scheduler(c.nproc)
    os.system("mkdir -p "+path)

    for mem_id in range(c.nens):
        job_name = model_name+f'_mem{mem_id+1}'
        job_opt = {
                    'path': path,
                    'member': mem_id,
                    'time_start': c.time_start,
                    'time_end': c.time_end,
                    'cycle_period': c.cycle_period,
                    }
        scheduler.submit_job(job_name, model.postprocess, **job_opt)  ##add job to the queue

    scheduler.start_queue() ##start the job queue
    if scheduler.error_jobs:
        raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')
    scheduler.shutdown()
    print(' done.', flush=True)

def postprocess(c):
    for model_name, model in c.model_config.items():
        timer(c)(postprocess_model_state)(c, model_name)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    postprocess(c)

