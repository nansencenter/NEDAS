import os

from utils.conversion import t2s
from utils.parallel import Scheduler

def ensemble_forecast(c):
    """
    This function runs ensemble forecasts to advance to the next cycle
    """
    for model_name, model in c.model_config.items():
        print(f"start {model_name} ensemble forecast", flush=True)

        nproc_per_job = c.model_def[model_name].get('nproc_per_mem', 1)
        walltime = c.model_def[model_name].get('walltime', 10000)
        nworker = c.nproc // nproc_per_job
        scheduler = Scheduler(nworker, walltime)

        path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        for mem_id in range(c.nens):

            job_name = model_name+f'_mem{mem_id+1}'
            job = model

            path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)

            job_opt = {'host': c.host,
                       'nedas_dir': c.nedas_dir,
                       'path': path,
                       'member': mem_id,
                       'time': c.time,
                       }
            # print(job_opt)

            scheduler.submit_job(job_name, job, **job_opt)  ##add job to the queue

        scheduler.start_queue() ##start the job queue

        if scheduler.error_jobs:
            raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')

        print(' done.', flush=True)


def ensemble_forecast_batch(c):
    for model_name, model in c.model_config.items():
        print(f"start {model_name} ensemble forecast ...", flush=True)

        path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        model.run(nens=c.nens, path=path, time=c.time)

        print('done.', flush=True)


if __name__ == "__main__":

    from config import Config
    from utils.progress import timer

    c = Config(parse_args=True)  ##get config from runtime args

    timer(c)(ensemble_forecast)(c)

