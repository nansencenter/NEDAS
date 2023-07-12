import config as c
import numpy as np
from models.nextsim import get_grid_from_msh
grid = get_grid_from_msh(c.MESH_FILE)
grid.dst_grid = c.ref_grid
mask = np.isnan(grid.convert(grid.x))
np.save('mask.npy', mask)

