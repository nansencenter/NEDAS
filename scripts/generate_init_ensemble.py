import os

from utils.conversion import t2s
from utils.parallel import Scheduler

def generate_init_ensemble(c, model_name):
    """
    This function generates the initial ensemble states at c.time_start
    """
    if c.pid == 0:
        model = c.model_config[model_name]
        print(f"generate {model_name} initial ensemble", flush=True)

        nproc_per_job = c.model_def[model_name].get('nproc_per_mem', 1)
        walltime = c.model_def[model_name].get('walltime')
        nworker = c.nproc // nproc_per_job
        scheduler = Scheduler(nworker, walltime)

        path = os.path.join(c.work_dir, 'cycle', t2s(c.time_start), model_name)
        os.system("mkdir -p "+path)

        for mem_id in range(c.nens):
            job_name = model_name+f'_mem{mem_id+1}'
            path = os.path.join(c.work_dir, 'cycle', t2s(c.time_start), model_name)

            job_opt = {'nedas_dir': c.nedas_dir,
                       'job_submit_cmd': c.job_submit_cmd,
                       'model_code_dir': c.model_def[model_name].get('model_code_dir'),
                       'model_data_dir': c.model_def[model_name].get('model_data_dir'),
                       'ens_init_dir': c.model_def[model_name].get('ens_init_dir'),
                       'path': path,
                       'member': mem_id,
                       'time': c.time_start,
                      }
            scheduler.submit_job(job_name, model.generate_initial_condition, **job_opt)  ##add job to the queue

        scheduler.start_queue() ##start the job queue
        if scheduler.error_jobs:
            raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')
        scheduler.shutdown()
        print(' done.', flush=True)


if __name__ == "__main__":
    from config import Config
    from utils.progress import timer
    c = Config(parse_args=True)  ##get config from runtime args

    for model_name, model in c.model_config.items():
        timer(c)(generate_init_ensemble)(c, model_name)

