import numpy as np
import config as cc

from .define_grid import uniform_grid, theta, corners

##Get reference grid projection from config file
##defined as proj.4 string, see https://proj.org/usage/projections.html
from pyproj import Proj
proj = Proj(cc.PROJ)
x_ref, y_ref = uniform_grid(cc.XSTART, cc.XEND, cc.YSTART, cc.YEND, cc.DX)

##grid transform operators
from .irregular_grid_interpolator import IrregularGridInterpolator
from .transform_vectors import transform_vectors

##io operator for netcdf format
from .io_netcdf import nc_read, nc_write
