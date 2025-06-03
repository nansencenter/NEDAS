import os
import subprocess
from datetime import timedelta
from NEDAS.config import Config

def generate_truth(c: Config) -> None:
    model = c.models['qg']

    truth_dir = model.truth_dir
    os.system("mkdir -p "+truth_dir)
    print(f"Creating truth run for qg model in {truth_dir}")

    run_dir = os.path.join(truth_dir, 'run')
    init_file = f"output_{c.time_start:%Y%m%d_%H}.bin"
    print(f"Running the model for spinup period to get initial condition: {init_file}")
    opt = {
        'path': run_dir,
        'member': 999,  ##make truth run different from the ensemble members
        'time': c.time_start - model.spinup_hours * timedelta(hours=1),
        'forecast_period': model.spinup_hours,
        'time_start': c.time_start,
        'time_end': c.time_end,
        'debug': c.debug,
        **c.job_submit,
        }
    model.run(**opt)

    c.time = c.time_start
    while c.time < c.time_end:
        opt['time'] = c.time
        opt['forecast_period'] = c.cycle_period

        file = f"output_{c.time:%Y%m%d_%H}.bin"
        next_file = f"output_{c.next_time:%Y%m%d_%H}.bin"
        print(f"Running the model from condition {file} to reach {next_file}")
        model.run(**opt)

        c.time = c.next_time
    print("done.")

    subprocess.run(f"mv -v {run_dir}/*/output*.bin {truth_dir}/.", shell=True, check=True)
    print(f"removing temporary run directory: {run_dir}")
    os.system(f"rm -rf {run_dir}")

if __name__ == '__main__':
    c = Config(parse_args=True)
    generate_truth(c)
