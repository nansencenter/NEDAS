##top-level control script
##usage: python run_expt.py -c <your config.yml file>

import os
from config import Config
from utils.conversion import dt1h
from utils.dir_def import cycle_dir
from scripts import preprocess, postprocess,perturb, assimilate, ensemble_forecast, diagnose

c = Config(parse_args=True)
c.show_summary()
os.system("mkdir -p "+c.work_dir)

print("Cycling start...", flush=True)

if c.time == c.time_start:
    c.prev_time = c.time

while c.time < c.time_end:
    c.next_time = c.time + c.cycle_period * dt1h
    print(f"\n\033[1;33mCURRENT CYCLE\033[0m: {c.time} => {c.next_time}", flush=True)

    os.system("mkdir -p "+cycle_dir(c, c.time))

    preprocess.run(c)

    perturb.run(c)

    ##assimilation step
    if c.run_assim and c.time >= c.time_assim_start and c.time <= c.time_assim_end:
        assimilate.run(c)
        postprocess.run(c)

    ##advance model state to next analysis cycle
    ensemble_forecast.run(c)

    ##compute diagnostics
    if c.run_diag:
        diagnose.run(c)

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

print("Cycling complete.", flush=True)
