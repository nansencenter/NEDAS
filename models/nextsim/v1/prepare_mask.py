import numpy as np
import config as c
from grid import Grid

##get mask from nextsim
from models.nextsim.v1 import read_grid_from_msh
ngrid = read_grid_from_msh(c.data_dir+'/mesh/small_arctic_10km.msh')
ngrid.set_destination_grid(c.grid)
mask = np.isnan(ngrid.convert(ngrid.x)).astype(int)

##save the mask to file
np.savez(c.maskfile, x=c.grid.x, y=c.grid.y, mask=mask)
