import numpy as np
import struct
import importlib
import sys
from datetime import datetime, timedelta
from conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from parallel import distribute_tasks

###top-level routine to apply the analysis increment to the original model restart (for next forecast)
##additive increment, just add post-prior state increment, interpolated to model grid, back to restart files
def add_increment(c):
    pass


##alignment technique
def alignment():
    pass

def warp():
    pass

##additive increments

