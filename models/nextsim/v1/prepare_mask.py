import numpy as np
from config import proj, xmin, xmax, ymin, ymax, maskfile
from grid import Grid

##high-res target grid
grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, 3000, centered=True)

##get mask from nextsim
# from models.nextsim import read_grid_from_msh
# ngrid = read_grid_from_msh('/cluster/work/users/yingyue/mesh/small_arctic_10km.msh')
# ngrid.dst_grid = grid
# mask = np.isnan(ngrid.convert(ngrid.x)).astype(int)

##get mask from topaz v5
from models.topaz.v5 import read_grid, read_mask
path = '/cluster/work/users/yingyue/data/test_Jul26958'
mgrid = read_grid(path)
mmask = read_mask(path, mgrid).astype(int)
mgrid.dst_grid = grid
mask = mgrid.convert(mmask)
mask[np.where(np.isnan(mask))] = 1
mask[np.where(mask>0)] = 1

##save the mask to file
np.savez(maskfile, x=grid.x, y=grid.y, mask=mask)
