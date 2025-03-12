##top-level control script
##usage: python run_expt.py -c <your config.yml file>

import os
from config import Config
from utils.conversion import dt1h
from scripts import analysis, preprocess, postprocess, perturb, ensemble_forecast, diagnose

c = Config(parse_args=True)
c.show_summary()
os.system("mkdir -p "+c.work_dir)

print("Cycling start...", flush=True)

while c.time < c.time_end:
    print(f"\n\033[1;33mCURRENT CYCLE\033[0m: {c.time} => {c.next_time}", flush=True)

    os.system("mkdir -p "+c.cycle_dir(c.time))

    preprocess.run(c)

    perturb.run(c)

    ##assimilation step
    if c.run_analysis and c.time >= c.time_analysis_start and c.time <= c.time_analysis_end:
        analysis.run(c)
        postprocess.run(c)

    ##advance model state to next analysis cycle
    if c.run_forecast:
        ensemble_forecast.run(c)

    ##compute diagnostics
    if c.run_diagnose:
        diagnose.run(c)

    ##advance to next cycle
    c.time = c.next_time

print("Cycling complete.", flush=True)
