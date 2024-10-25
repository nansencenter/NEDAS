import os

from utils.conversion import t2s
from utils.parallel import Scheduler

def ensemble_forecast_scheduler(c, model_name):
    """
    This function runs ensemble forecasts to advance to the next cycle
    """
    if c.pid == 0:
        model = c.model_config[model_name]
        print(f"start {model_name} ensemble forecast", flush=True)

        nproc_per_job = c.model_def[model_name].get('nproc_per_mem', 1)
        walltime = c.model_def[model_name].get('walltime')
        nworker = c.nproc // nproc_per_job
        scheduler = Scheduler(nworker, walltime)

        path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
        os.system("mkdir -p "+path)

        for mem_id in range(c.nens):
            job_name = model_name+f'_mem{mem_id+1}'
            path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
            output_dir = os.path.join(c.work_dir, 'cycle', t2s(c.next_time), model_name)

            job_opt = {'task_nproc': nproc_per_job,
                       'job_submit_cmd': c.job_submit_cmd,
                       'path': path,
                       'member': mem_id,
                       'time': c.time,
                       'forecast_period': c.cycle_period,
                       'output_dir': output_dir,
                       **c.model_def[model_name],
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
    if c.pid == 0:
        model = c.model_config[model_name]
        print(f"start {model_name} ensemble forecast ...", flush=True)

        path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
        output_dir = os.path.join(c.work_dir, 'cycle', t2s(c.next_time), model_name)
        os.system("mkdir -p "+path)

        job_opt = {'job_submit_cmd': c.job_submit_cmd,
                   'job_submit_node': c.job_submit_node,
                   'path': path,
                   'time': c.time,
                   'n_ens': c.nens,
                   'forecast_period': c.cycle_period,
                   'output_dir': output_dir,
                   **c.model_def[model_name],
                  }
        model.run(nens=c.nens, **job_opt)
        print('done.', flush=True)


def ensemble_forecast(c, model_name):
    ens_run_type = c.model_def[model_name]['ens_run_type']
    if ens_run_type == 'batch':
        ensemble_forecast_batch(c, model_name)
    elif ens_run_type == 'scheduler':
        ensemble_forecast_scheduler(c, model_name)
    else:
        raise TypeError("unknown ensemble forecast type: '"+ens_run_type+"' for "+model_name)


if __name__ == "__main__":
    from config import Config
    from utils.progress import timer
    c = Config(parse_args=True)  ##get config from runtime args

    for model_name, model in c.model_config.items():
        timer(c)(ensemble_forecast)(c, model_name)

