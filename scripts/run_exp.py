import os
import subprocess

from config import Config
from utils.progress import timer
from utils.conversion import t2s, s2t, dt1h
from scripts.ensemble_forecast import ensemble_forecast
# from scripts.assimilate import assimilate

c = Config(parse_args=True)
c.show_summary()

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

    config_file = os.path.join(cycle_dir, 'config.yml')
    c.dump_yaml(config_file)

    ###
    timer()(ensemble_forecast)(c)

    ###
    job_submitter = os.path.join(c.nedas_dir, 'config', 'env', c.host, 'job_submit.sh')
    assimilate_code = os.path.join(c.nedas_dir, 'scripts', 'assimilate.py')
    shell_cmd = job_submitter+f" {c.nproc} {0}"
    shell_cmd += " python -m mpi4py "+assimilate_code
    shell_cmd += " --config_file "+config_file
    subprocess.run(shell_cmd, shell=True, check=True, capture_output=True, text=True)

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

print("Cycling complete.", flush=True)

