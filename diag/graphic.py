import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cmocean.cm import ice
import grid

plot_crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=60)

def plot_var_on_mesh(f, dat):
    return 
