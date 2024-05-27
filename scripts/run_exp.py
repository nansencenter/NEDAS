from datetime import datetime, timedelta

from config import Config
from utils.log import message, timer
from utils.parallel import run_by_root
# from scripts.prepare_icbc import prepare_icbc
from scripts.ensemble_forecast import ensemble_forecast
# from scripts.prepare_state import prepare_state

import time

c = Config()

message(c.comm, "Cycling start...\n\n", c.pid_show)

prev_time = c.time
while c.time < c.time_end:
    next_time = c.time + timedelta(hours=c.cycle_period)
    message(c.comm, f"*** cycle {c.time} => {next_time} *** \n\n", c.pid_show)

    #prepare_icbc()

    timer(c.comm, c.pid_show)(run_by_root(c.comm)(ensemble_forecast))(c)

    # prepare_state(c, data)



    ##advance to next cycle
    prev_time = c.time
    c.time = next_time

message(c.comm, "Cycling complete.\n", c.pid_show)

