import importlib
import os
import time

from utils.parallel import run_by_root, Scheduler
from utils.conversion import t2s
from utils.log import message, timer


def ensemble_forecast(c):
    """
    This function runs ensemble forecasts to advance to the next cycle
    """
    for model_config in c.model_def:

        ##load the Model class
        model_name = model_config['name']

        message(c.comm, f"start {model_name} ensemble forecast\n")

        module = importlib.import_module('models.'+model_name)
        Model = getattr(module, 'Model')
        path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)

        nproc_per_job = model_config['nproc_per_mem']
        nworker = c.nproc // nproc_per_job
        walltime = model_config['walltime']

        scheduler = Scheduler(nworker, walltime)

        for mem_id in range(c.nens):
            job = Model()
            job_name = model_name+f'_{mem_id}'
            scheduler.submit_job(job_name, job, c, path, member=mem_id, time=c.time)  ##add job to the queue

        scheduler.start_queue() ##start the job queue

        message(c.comm, model_name+' model ensemble runs complete\n\n', c.pid_show)


if __name__ == "__main__":

    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    timer(c.comm, 0)(run_by_root(c.comm)(ensemble_forecast))(c)

