###a simple 2d vortex dynamic model

import numpy as np
from numba import njit
from assim_tools.multiscale import fft2, ifft2, get_wn

