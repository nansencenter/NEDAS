"""Utility functions for misc. graphics"""

import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.path import Path
from matplotlib.patches import PathPatch

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

def arrowhead_xy(x1, x2, y1, y2, hw, hl):
    """Given line segment [x1,y1]-[x2,y2], return the segments for its arrow head (making it a vector)"""
    np.seterr(invalid='ignore')
    ll = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    sinA = (y2 - y1)/ll
    cosA = (x2 - x1)/ll
    h1x = x1 - 0.2*hl*cosA
    h1y = y1 - 0.2*hl*sinA
    h2x = x1 + 0.8*hl*cosA - 0.5*hw*sinA
    h2y = y1 + 0.8*hl*sinA + 0.5*hw*cosA
    h3x = x1 + 0.5*hl*cosA
    h3y = y1 + 0.5*hl*sinA
    h4x = x1 + 0.8*hl*cosA + 0.5*hw*sinA
    h4y = y1 + 0.8*hl*sinA - 0.5*hw*cosA
    return [h1x, h2x, h3x, h4x, h1x], [h1y, h2y, h3y, h4y, h1y]

def draw_reference_vector_legend(ax, xr, yr, V, L, hw, hl, refcolor, linecolor, ref_units=''):
    """Draw a legend box with reference vector and units string"""
    ##draw a box
    xb = [xr-L*1.3, xr-L*1.3, xr+L*1.3, xr+L*1.3, xr-L*1.3]
    yb = [yr+L/2, yr-L, yr-L, yr+L/2, yr+L/2]
    ax.fill(xb, yb, color=refcolor, zorder=6)
    ax.plot(xb, yb, color='k', zorder=6)
    ##draw the reference vector
    ax.plot([xr-L/2, xr+L/2], [yr, yr], color=linecolor, zorder=8)
    ax.fill(*arrowhead_xy(xr+L/2, xr-L/2, yr, yr, hw, hl), color=linecolor, zorder=8)
    ##add unit string annotation below the vector
    ax.text(xr, yr-L/2, f"{V} {ref_units}", color='k', ha='center', va='center', zorder=8)

def draw_line(ax, data, linecolor, linewidth, linestyle, zorder):
    xy = data['xy']
    parts = data['parts']
    for i in range(len(xy)):
        for j in range(len(parts[i])-1): ##plot separate segments if multi-parts
            ax.plot(*zip(*xy[i][parts[i][j]:parts[i][j+1]]), color=linecolor, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
        ax.plot(*zip(*xy[i][parts[i][-1]:]), color=linecolor, linewidth=linewidth, linestyle=linestyle, zorder=zorder)

def draw_patch(ax, data, color, zorder):
    xy = data['xy']
    parts = data['parts']
    for i in range(len(xy)):
        code = [Path.LINETO] * len(xy[i])
        for j in parts[i]:  ##make discontinuous patch if multi-parts
            code[j] = Path.MOVETO
        ax.add_patch(PathPatch(Path(xy[i], code), facecolor=color, edgecolor=color, linewidth=0.1, zorder=zorder))
