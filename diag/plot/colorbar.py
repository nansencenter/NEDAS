import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

def adjust_ax_size(ax):
    """
    Make ax a little smaller on right hand side to make room for colorbar
    For plots without colorbar, it is still useful to call this function
    so that the axes will align with those with colorbars
    """
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left, bottom, width*0.9, height])

def add_colorbar(fig, ax, cmap, vmin, vmax, nlevels=10, fontsize=12, units=None):
    """
    Add a colorbar to thwe right hand side of ax
    Inputs:
    - fig: matplotlib.pyplot figure object
    - ax: matplotlib.pyplot axes object
    - cmap: matplotlib colormap object
    - vmin, vmax: float
      Min and Max value bound
    - nlevels: int (default 10)
      Number of color levels
    - fontsize: int (default 12)
      Font size for the ticklabel and units label
    - units: str (optional)
      Unit string to be shwon on colorbar title
    """
    dv = (vmax - vmin) / nlevels
    bounds = np.arange(vmin, vmax+dv, dv)

    norm = BoundaryNorm(bounds, ncolors=256, extend='both')

    ##adjust the main plot ax to make room for colorbar
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left, bottom, width*0.9, height])

    ##add colorbar ax to the right
    cax = fig.add_axes([left+width*0.95, bottom+height*0.15, width*0.03, height*0.7])
    cax.tick_params(labelsize=fontsize)

    ##draw the colorbar
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=bounds)
    if units is not None:
        cbar.ax.set_title(units, fontsize=fontsize, loc='left', pad=25)