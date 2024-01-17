##run forecast using the vort2d model
##this program is mimicking the run script of a real model, it is called by
##top level control script run_cycle.sh to spawm several runs simultaneously
##as if running ensemble forecasts

import numpy as np
import sys
from datetime import datetime, timedelta
import config as c
import models.vort2d as model
from conversion import s2t, t2s

##current cycle time
time = s2t(sys.argv[1])

##member index, start from 0
if len(sys.argv) > 2:
    mem_id = int(sys.argv[2]) - 1
else:
    mem_id = None

##time at next cycle, where current forecast will end
next_time = time + timedelta(hours=1) * c.cycle_period

path = './'

##read the initial condition
state = model.read_var(path, c.grid, name='velocity', is_vector=True, time=time, member=mem_id)

t = time
print('vort2d model forecast start at', t)
while t < next_time:
    t += timedelta(hours=1) * model.restart_dt

    ##run the model
    state = model.advance_time(state, model.dx, model.restart_dt, model.dt, model.gen, model.diss)

    ##save restart file
    model.write_var(path, c.grid, state, name='velocity', is_vector=True, time=t, member=mem_id)

    print(t)

print('vort2d model forecast finished successfully')

