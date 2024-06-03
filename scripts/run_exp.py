import os
import sys
import subprocess

from config import Config
from utils.progress import timer
from utils.conversion import t2s, s2t, dt1h

c = Config(parse_args=True)
c.show_summary()

from scripts.ensemble_forecast import ensemble_forecast

if not os.path.exists(c.work_dir):
    os.makedirs(c.work_dir)

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
        for mem_id in range(c.nens):
            for model_name, model in c.model_config.items():
                path1 = os.path.join(c.scratch_dir, 'qg_ens_runs')
                path2 = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
                file1 = model.filename(path=path1, member=mem_id, time=c.time)
                file2 = model.filename(path=path2, member=mem_id, time=c.time)
                os.system(f"mkdir -p {os.path.dirname(file2)}; cp {file1} {file2}")

    ###
    timer(c)(ensemble_forecast)(c)

    ##copy forecast files to next_time
    for mem_id in range(c.nens):
        for model_name, model in c.model_config.items():
            path1 = os.path.join(c.work_dir, 'cycle', t2s(c.time), model_name)
            path2 = os.path.join(c.work_dir, 'cycle', t2s(c.next_time), model_name)
            file1 = model.filename(path=path1, member=mem_id, time=c.next_time)
            file2 = model.filename(path=path2, member=mem_id, time=c.next_time)
            os.system(f"mkdir -p {os.path.dirname(file2)}; cp {file1} {file2}")

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

    config_file = os.path.join(cycle_dir, 'config.yml')
    c.dump_yaml(config_file)

    ###
    analysis_dir = os.path.join(c.work_dir,'cycle', t2s(c.time), 'analysis', c.s_dir)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

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

