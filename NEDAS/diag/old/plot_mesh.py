import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy.crs as ccrs
from grid_util import *

###making uniform analysis grid parameters from comparing with nextsim mesh (from msh file)

##grid 1 from msh file
msh_file = '../diag/output/mesh/small_arctic_5km.msh'
x1, y1, z1 = get_grid_from_msh(msh_file)

##grid 2 uniform
x2, y2, z2 = gen_uniform_grid(-2.5e6, 2e6, -2e6, 2.5e6, 5, -45)

lon1, lat1 = xyz_to_lonlat(x1, y1, z1)
lon2, lat2 = xyz_to_lonlat(x2, y2, z2)

d=1
plt.figure(figsize=(12,12))
ax = plt.subplot(1,1,1,projection=ccrs.NorthPolarStereo())
ax.scatter(np.squeeze(lon2)[::d], np.squeeze(lat2)[::d], s=1, c='red', transform=ccrs.PlateCarree())
ax.scatter(lon1[::d], lat1[::d], s=1, c='blue', transform=ccrs.PlateCarree())
ax.coastlines('50m')
plt.show()
