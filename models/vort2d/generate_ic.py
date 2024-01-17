import numpy as np
import models.vort2d as model
import config as c
import sys
from conversion import s2t

time = s2t(sys.argv[1])

##input random seed is provided, set it
if len(sys.argv) > 2:
    np.random.seed(int(sys.argv[2]))

state = model.initialize(c.grid, model.Vmax, model.Rmw, model.Vbg, model.Vslope)

model.write_var('./', c.grid, state, name='velocity', is_vector=True, time=time)

