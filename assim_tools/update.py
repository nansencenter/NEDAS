import numpy as np
import struct
import importlib
import sys
from datetime import datetime, timedelta
from .common import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from .parallel import distribute_tasks

###top-level routine to apply the analysis increment to the original model restart (for next forecast)
def update_restart(c, comm):
    pass


##alignment technique
def optical_flow():
    pass

def warp():
    pass

##additive increments

