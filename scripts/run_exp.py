import os
import sys
import subprocess

from config import Config
from utils.progress import timer
from utils.conversion import t2s, s2t, dt1h
from scripts.ensemble_forecast import ensemble_forecast

c = Config(parse_args=True)
c.show_summary()
os.system("mkdir -p "+c.work_dir)

print("Cycling start...", flush=True)

c.prev_time = c.time
while c.time < c.time_end:
    c.next_time = c.time + c.cycle_period * dt1h
    print(60*'-'+f"\ncurrent cycle: {c.time} => {c.next_time}", flush=True)

    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    cycle_dir = os.path.join(c.work_dir, 'cycle', t2s(c.time))
    os.system("mkdir -p "+cycle_dir)

    ###assimilation is run by a parallel call to python assimilate.py
    if c.time > c.time_start and c.run_assim:
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

    ###run the ensemble forecasts, use batch or scheduler approach
    for model_name, model in c.model_config.items():
        timer(c)(ensemble_forecast)(c, model_name)

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

print("Cycling complete.", flush=True)

