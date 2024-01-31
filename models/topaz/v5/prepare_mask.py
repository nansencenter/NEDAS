import numpy as np
from grid import Grid
import config as c

##get mask from topaz v5
from models.topaz.v5 import read_grid, read_mask
path = c.work_dir
mgrid = read_grid(path)
mmask = read_mask(path, mgrid).astype(int)
mgrid.dst_grid = grid
mask = mgrid.convert(mmask)
mask[np.where(np.isnan(mask))] = 1
mask[np.where(mask>0)] = 1

##save the mask to file
np.savez(maskfile, x=grid.x, y=grid.y, mask=mask)

