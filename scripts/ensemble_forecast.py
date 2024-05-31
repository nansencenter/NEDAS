import os

from utils.parallel import Scheduler
from utils.conversion import t2s
from utils.log import message, timer

def ensemble_forecast(c):
    """
    This function runs ensemble forecasts to advance to the next cycle
    """
    for model_name, model in c.model_config.items():

        message(c.comm, f"start {model_name} ensemble forecast\n")

        nproc_per_job = c.model_def[model_name].get('nproc_per_mem', 1)
        walltime = c.model_def[model_name].get('walltime', 10000)
        nworker = c.nproc // nproc_per_job
        scheduler = Scheduler(c.comm, nworker, walltime)

        path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)

        for mem_id in range(c.nens):

            job_name = model_name+f'_mem{mem_id+1}'
            job = model

            if c.time == c.time_start:
                icbc_path = c.model_def[model_name]['icbc_path']
                input_file = model.filename(path=icbc_path, member=mem_id, time=c.time)
            else:
                prev_path = os.path.join(c.work_dir, 'cycle', t2s(c.prev_time), model_name)
                input_file = model.filename(path=prev_path, member=mem_id, time=c.time)
            output_file = model.filename(path=path, member=mem_id, time=c.next_time)

            job_opt = {'host': c.host,
                       'nedas_dir': c.nedas_dir,
                       'path': path,
                       'member': mem_id,
                       'time': c.time,
                       'input_file': input_file,
                       'output_file': output_file,
                       }

            scheduler.submit_job(job_name, job, **job_opt)  ##add job to the queue

        scheduler.start_queue() ##start the job queue

        if scheduler.error_jobs:
            raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')

        message(c.comm, ' done.\n', c.pid_show)


if __name__ == "__main__":

    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    timer(c.comm)(ensemble_forecast)(c)

