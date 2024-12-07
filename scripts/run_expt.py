##top-level control script
##usage: python run_expt.py -c <your config.yml file>

import os
from config import Config
from utils.progress import timer
from utils.conversion import t2s, s2t, dt1h
from utils.dir_def import cycle_dir
from scripts import preprocess, postprocess,perturb, assimilate, diag, ensemble_forecast

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

    #preprocess.run(c)

    #perturb.run(c)

    ##assimilation step
    if c.run_assim and c.time >= c.time_assim_start and c.time <= c.time_assim_end:
        ##multiscale approach: loop over scale components and perform assimilation on each scale
        for c.scale_id in range(c.nscale):
            assimilate.run(c)

    #    postprocess.run(c)

    ##forecast step
    #ensemble_forecast.run(c)

    ##compute diagnostics
    if c.run_diag:
        diag.run(c)

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

print("Cycling complete.", flush=True)

