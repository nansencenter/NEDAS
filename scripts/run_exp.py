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


# def perturb_member(path, model, mem_id, time, k, name=None, amp=0, hcorr=1):
#     print('perturb', mem_id, time)
#     fld = model.read_var(path=path, name=name, member=mem_id, time=c.time, k=k)
#     fld += random_field_gaussian(model.grid.nx, model.grid.ny, amp, hcorr)
#     model.write_var(fld, path=path, name=name, member=mem_id, time=c.time, k=k)


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

        ##perturb the initial ensemble
        # for model_name, model in c.model_config.items():
        #     path2 = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
        #     for mem_id in range(c.nens):
        #         for k in range(model.nz):
        #             state_init = model.read_var(path=path1, name='temperature', time=c.time, k=k)
        #             state = state_init + random_field_gaussian(model.grid.nx, model.grid.ny, 0.1, 50)
        #             file2 = model.filename(path=path2, member=mem_id, time=c.time)
        #             os.system(f"mkdir -p {os.path.dirname(file2)}; touch {file2}")
        #             model.write_var(state, path=path2, name='temperature', member=mem_id, time=c.time, k=k)

    #add perturbation to the ensemble state
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     futures = []
    # for model_name, model in c.model_config.items():
    #     for mem_id in range(c.nens):
    #         for k in range(model.nz):
    #             path = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
    #             #TODO: set perturb opt in config
    #             perturb_opt = {'name': 'streamfunc',
    #                         'amp': 1,
    #                         'hcorr': 20,}
    #             perturb_member(path, model, mem_id, c.time, k, **perturb_opt)
    #                 future = executor.submit(perturb_member, path, model, mem_id, c.time, k, **perturb_opt)
    #                 futures.append(future)
    #     for future in as_completed(futures):
    #         try:
    #             result = future.result()
    #         except Exception as e:
    #             print(f'An error occurred: {e}')

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

