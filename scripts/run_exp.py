import sys, os, shutil
import subprocess

from config import Config
from utils.progress import timer
from utils.conversion import t2s, s2t, dt1h
from scripts.generate_init_ensemble import generate_init_ensemble
from scripts.ensemble_forecast import ensemble_forecast

c = Config(parse_args=True)
c.show_summary()
#os.system("mkdir -p "+c.work_dir)
os.makedirs(c.work_dir, exist_ok=True)

print("Cycling start...", flush=True)

c.prev_time = c.time
while c.time < c.time_end:
    c.next_time = c.time + c.cycle_period * dt1h
    print(60*'-'+f"\ncurrent cycle: {c.time} => {c.next_time}", flush=True)

    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    cycle_dir = os.path.join(c.work_dir, 'cycle', t2s(c.time))
    os.makedirs(cycle_dir, exist_ok=True)

    ##at first cycle, generate the initial ensemble
    if c.time == c.time_start:
        for model_name, model in c.model_config.items():
            timer(c)(generate_init_ensemble)(c, model_name)

    ###assimilation step
    ##this is run by a parallel call to python assimilate.py
    if c.time > c.time_start and c.run_assim:
        print('start assimilation...', flush=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assimilate_code = os.path.join(script_dir, 'assimilate.py')
        shell_cmd = c.job_submit_cmd+f" {c.nproc} {0}"
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

