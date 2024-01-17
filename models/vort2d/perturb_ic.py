import numpy as np
import models.vort2d as model
import config as c
import sys
from conversion import s2t

time = s2t(sys.argv[1])
nens = int(sys.argv[2])

for m in range(nens):
    state = model.initialize(c.grid, model.Vmax, model.Rmw, model.Vbg, model.Vslope, loc_sprd=model.loc_sprd)

    model.write_var('./', c.grid, state, name='velocity', is_vector=True, time=time, member=m)

