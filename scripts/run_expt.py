##top-level control script
##usage: python run_expt.py -c <your config.yml file>

import os
from config import Config
from utils.progress import timer
from utils.conversion import t2s, s2t, dt1h
from utils.dir_def import cycle_dir
from scripts.prepare_ensemble import prepare_ensemble
from scripts.assimilate import assimilate
from scripts.ensemble_forecast import ensemble_forecast

c = Config(parse_args=True)
c.show_summary()
os.system("mkdir -p "+c.work_dir)

print("Cycling start...", flush=True)

c.prev_time = c.time
while c.time < c.time_end:
    c.next_time = c.time + c.cycle_period * dt1h
    print(60*'-'+f"\ncurrent cycle: {c.time} => {c.next_time}", flush=True)

    cycle_dir = cycle_dir(c, c.time)
    os.system("mkdir -p "+cycle_dir)

    ##preparation of initial ensemble (linking files, perturbing icbc)
    preprocess(c)
    perturb(c)

    ##assimilation step
    ##multiscale approach: loop over scale components scale_id=0,...,nscale
    for c.scale_id in range(c.nscale):
        assimilate(c)

    postprocess(c)

    ##forecast step
    ensemble_forecast(c)

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

print("Cycling complete.", flush=True)

