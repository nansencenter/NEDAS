import numpy as np
from pynextsim import NextsimBin
from pynextsim.gridding import Grid

##output grid (uniform)
nb = NextsimBin('output/control_run/field_final.bin')
grid_params = nb.mesh_info.define_grid(extent=[-2.5e6, 2e6, -2e6, 2.5e6], resolution=5000)
grid = Grid.init_from_grid_params(grid_params, projection=nb.mesh_info.projection)

np.save('output/grid.npy', grid.xy)
