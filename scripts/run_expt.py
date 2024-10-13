##top-level control script
##usage: python run_expt.py -c <your config.yml file>

import os
from config import Config
from utils.progress import timer
from utils.conversion import t2s, s2t, dt1h
from utils.dir_def import cycle_dir
from scripts.preprocess import preprocess
from scripts.postprocess import postprocess
from scripts.perturb import perturb
from scripts.assimilate import assimilate
from scripts.ensemble_forecast import ensemble_forecast
from scripts.diag import diag

c = Config(parse_args=True)
c.show_summary()
os.system("mkdir -p "+c.work_dir)

print("Cycling start...", flush=True)

c.prev_time = c.time
while c.time < c.time_end:
    c.next_time = c.time + c.cycle_period * dt1h
    print(f"\nCURRENT CYCLE: {c.time} => {c.next_time}", flush=True)

    os.system("mkdir -p "+cycle_dir(c, c.time))

    ##preparation of initial ensemble (linking files, perturbing icbc)
    preprocess(c)
    perturb(c)

    ##assimilation step
    if c.run_assim and c.time >= c.time_assim_start and c.time <= c.time_assim_end:
        ##multiscale approach: loop over scale components and perform assimilation on each scale
        for c.scale_id in range(c.nscale):
            assimilate(c)

    postprocess(c)

    ##forecast step
    ensemble_forecast(c)

    ##compute diagnostics
    if c.run_diag:
        diag(c)

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

print("Cycling complete.", flush=True)

