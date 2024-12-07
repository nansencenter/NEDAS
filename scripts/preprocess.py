import os
from utils.progress import timer
from utils.parallel import Scheduler
from utils.dir_def import forecast_dir
from utils.shell_utils import makedir

def preprocess(c, model_name):
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

    path = forecast_dir(c, c.time, model_name)
    makedir(path)

    if c.job_submit.get('run_separate_jobs', False):
        ##ideally, if in preprocess method jobs are submitted through run_job, then
        ##here nworker should be c.nens
        ## model preprocess method typically contain a lot of serial subprocess.run
        ##not necessarily using run_job and JobSubmitter class to run the job
        ##so temporarily use os.cpu_count to limit the resource here
        nproc_avail = os.cpu_count()
        nworker = min(c.nens, nproc_avail)
    else:
        ##Scheduler will use nworkers to spawn preprocess task for each member
        ##to the available nproc processors
        nworker = c.nproc // model.nproc_per_util
    scheduler = Scheduler(nworker, debug=c.debug)

    for mem_id in range(c.nens):
        job_name = f'preproc_{model_name}_mem{mem_id+1}'
        job_opt = {
            'restart_dir': restart_dir,
            'path': path,
            'member': mem_id,
            'time': c.time,
            'forecast_period': c.cycle_period,
            **c.job_submit,
            }
        scheduler.submit_job(job_name, model.preprocess, **job_opt)  ##add job to the queue

    scheduler.start_queue() ##start the job queue
    if scheduler.error_jobs:
        raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')
    scheduler.shutdown()
    print(' done.', flush=True)

def run(c):
    for model_name, model in c.model_config.items():
        timer(c)(preprocess)(c, model_name)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    run(c)

