import numpy as np

def regular_grid(xstart, xend, ystart, yend, dx, centered=False):
    xcoord = np.arange(xstart, xend, dx)
    ycoord = np.arange(ystart, yend, dx)
    x, y = np.meshgrid(xcoord, ycoord)
    if centered:
        x += 0.5*dx  ##move coords to center of grid box
        y += 0.5*dx
    return x, y

from .grid import Grid
from .converter import Converter
from .multiscale import get_scale_comp
