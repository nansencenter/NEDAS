import numpy as np
from config import proj, xmin, xmax, ymin, ymax, maskfile
from grid import Grid
import pyproj

##high-res target grid
grid = Grid.regular_grid(pyproj.Proj(proj), xmin, xmax, ymin, ymax, 3000, centered=True)

##get mask from nextsim
from models.nextsim import read_grid_from_msh
ngrid = read_grid_from_msh('/cluster/work/users/yingyue/mesh/small_arctic_10km.msh')
ngrid.dst_grid = grid
mask = np.isnan(ngrid.convert(ngrid.x)).astype(int)

##save the mask to file
np.savez(maskfile, x=grid.x, y=grid.y, mask=mask)
