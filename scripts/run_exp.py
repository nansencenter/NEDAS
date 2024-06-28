import os
import sys
import importlib
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import Config
from perturb import random_field_gaussian
from utils.progress import timer
from utils.conversion import t2s, s2t, dt1h
from scripts.ensemble_forecast import ensemble_forecast_scheduler, ensemble_forecast_batch


def copy_file(file1, file2):
    # print('copy', file1, file2)
    os.system(f"mkdir -p {os.path.dirname(file2)}; cp {file1} {file2}")


c = Config(parse_args=True)
c.show_summary()

if not os.path.exists(c.work_dir):
    os.makedirs(c.work_dir)

if c.use_synthetic_obs:
    os.system(f"cd {c.work_dir}; ln -fs {c.scratch_dir}/qg/truth .")

print("Cycling start...", flush=True)

c.prev_time = c.time
while c.time < c.time_end:
    c.next_time = c.time + c.cycle_period * dt1h

    print(60*'-'+f"\ncurrent cycle: {c.time} => {c.next_time}", flush=True)

    cycle_dir = os.path.join(c.work_dir, 'cycle', t2s(c.time))
    if not os.path.exists(cycle_dir):
        os.makedirs(cycle_dir)

    if c.time == c.time_start:
        ##copy initial ensemble
        ##TODO: figure out a way to do this initialization in model class?
        ##      hard-wiring this code in the top-level script is not so good
        ##TODO: 128 processors available, but it depends on env/machine
        with ProcessPoolExecutor(max_workers=128) as executor:
            futures = []
            for mem_id in range(c.nens):
                for model_name, model in c.model_config.items():
                    path1 = os.path.join(c.scratch_dir, 'qg_ens_runs')
                    path2 = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
                    file1 = model.filename(path=path1, member=mem_id, time=c.time)
                    file2 = model.filename(path=path2, member=mem_id, time=c.time)
                    future = executor.submit(copy_file, file1, file2)
                    futures.append(future)
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f'An error occurred: {e}')

    ###run the ensemble forecasts, use batch or scheduler approach
    for model_name, model in c.model_config.items():
        ens_run_type = c.model_def[model_name]['ens_run_type']
        if ens_run_type == 'batch':
            timer(c)(ensemble_forecast_batch)(c, model_name)
        elif ens_run_type == 'scheduler':
            timer(c)(ensemble_forecast_scheduler)(c, model_name)

    ##collectively copy forecast files to next_cycle
    with ProcessPoolExecutor(max_workers=128) as executor:
        futures = []
        for mem_id in range(c.nens):
            for model_name, model in c.model_config.items():
                path1 = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
                path2 = os.path.join(c.work_dir, 'cycle', t2s(c.next_time), model_name)
                file1 = model.filename(path=path1, member=mem_id, time=c.next_time)
                file2 = model.filename(path=path2, member=mem_id, time=c.next_time)
                future = executor.submit(copy_file, file1, file2)
                futures.append(future)
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f'An error occurred: {e}')

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

    config_file = os.path.join(cycle_dir, 'config.yml')
    c.dump_yaml(config_file)

    ###assimilation is run by a parallel call to python assimilate.py
    if c.run_assim:
        print('start assimilation...', flush=True)
        job_submitter = os.path.join(c.nedas_dir, 'config', 'env', c.host, 'job_submit.sh')
        assimilate_code = os.path.join(c.nedas_dir, 'scripts', 'assimilate.py')
        shell_cmd = job_submitter+f" {c.nproc} {0}"
        shell_cmd += " python -m mpi4py "+assimilate_code
        shell_cmd += " --config_file "+config_file
        try:
            p = subprocess.Popen(shell_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
            p.wait()

            if p.returncode != 0:
                print(f"{p.stderr}")
                exit()
        except Exception as e:
            print(f"subprocess.run raised Exception {e}")
            exit()


print("Cycling complete.", flush=True)

