#!/usr/bin/env python
##run forecast using the vort2d model
##this program is mimicking the run script of a real model, it is called by
##top level control script run_cycle.sh to spawm several runs simultaneously
##as if running ensemble forecasts

import sys, io
import numpy as np
from models.vort2d import read_var, write_var, advance_time


print(sys.argv[1])
