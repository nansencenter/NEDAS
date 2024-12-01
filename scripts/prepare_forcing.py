import os

from utils.conversion import t2s
from utils.parallel import Scheduler

def prepare_forcing(c, model_name) -> None:
    """
    This function prepare forcing before advancing to the next cycle using NEDAS scheduler.

    The shceduler allows for the parallelised preparation of forcing used by each ensemble member.
    This function requires that the model module has a method `prepare_forcing`.

    Parameters
    ----------
    c : Config
        Configuration object.
        This object should contain the property `time` and `next_time`,
        and the `model_def` property with a dictionary key of `model_name`.
        Under the `model_name` section, the following properties are required:
        - `nproc_per_mem`: number of processors required for each ensemble member.
                           This is optional and defaults to 1.
        - `walltime`: limit on the walltime for each job.
        - `nens`: number of ensemble members.
    model_name : str
        Name of the model to prepare forcing for.
        This is used to create model specific directories.
    """
    # only the parent process should submit jobs
    if c.pid != 0: return
    # get the module handle of the model
    model = c.model_config[model_name]
    print(f"start {model_name} forcing setup for next forecast cycle", flush=True)

    # get the number of processors required for each worker
    # this information can be used in the function called in the scheduler
    # where each thread can request a number of processors
    nproc_per_job = c.model_def[model_name].get('nproc_per_mem', 1)
    # a limit on the walltime for each job limited by the scheduler
    walltime = c.model_def[model_name].get('walltime')
    nworker = c.nproc // nproc_per_job
    # The scheduler is a multithreading scheduler
    # each worker in the scheduler only uses one thread
    # without special handling
    scheduler = Scheduler(nworker, walltime)

    # create the directory for the model at current model time
    path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
    os.system("mkdir -p "+path)

    # create a list of jobs for each ensemble member
    for mem_id in range(c.nens):
        job_name = model_name+f'_mem{mem_id+1}'

        # job options are passed to the function called by the scheduler
        job_opt = {'task_nproc': nproc_per_job,
                   'path': path,
                   'member': mem_id,
                   'time': c.time,
                   'next_time': c.next_time,
                   **c.model_def[model_name],
                   }
        # add job to the queue
        scheduler.submit_job(job_name, model.prepare_forcing, **job_opt)

    # running the jobs
    scheduler.start_queue()
    if scheduler.error_jobs:
        raise RuntimeError(f'scheduler: there are jobs with errors: {scheduler.error_jobs}')
    scheduler.shutdown()
    print(' done.', flush=True)

if __name__ == "__main__":
    from config import Config
    from utils.progress import timer
    c = Config(parse_args=True)  ##get config from runtime args

    for model_name, model in c.model_config.items():
        timer(c)(prepare_forcing)(c, model_name)

