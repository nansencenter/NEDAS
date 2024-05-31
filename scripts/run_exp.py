from config import Config
from utils.log import message, timer
from utils.conversion import t2s, s2t, dt1h
from utils.parallel import bcast_by_root
from scripts.ensemble_forecast import ensemble_forecast
from scripts.assimilate import assimilate

c = Config(parse_args=True)

message(c.comm, "Cycling start...\n\n", c.pid_show)

c.prev_time = c.time
while c.time < c.time_end:
    c.next_time = c.time + c.cycle_period * dt1h
    message(c.comm, f"*** cycle {c.time} => {c.next_time} *** \n\n", c.pid_show)

    timer(c.comm)(ensemble_forecast)(c)

    assimilate(c)

    ##advance to next cycle
    c.prev_time = c.time
    c.time = c.next_time

message(c.comm, "Cycling complete.\n", c.pid_show)

