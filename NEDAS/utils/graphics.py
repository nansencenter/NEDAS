import numpy as np
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def get_cmap(cmap_name: str):
    """
    Get colormap object based on the input name string.

    Args:
        cmap_name (str): The name of the color map. For `cmocean` colormaps, the name
            should be in the format 'cmocean.<cmap_name>'

    Returns:
        Colormap: A colormap object corresponding to the given name.

    Raises:
        KeyError: If the colormap name is not found.
    """
    if cmap_name.split('.')[0] == 'cmocean':
        import cmocean
        cmap = getattr(cmocean.cm, cmap_name.split('.')[-1])
    else:
        cmap = colormaps[cmap_name]
    return cmap

def adjust_ax_size(ax):
    """
    Make plot axes a little smaller on right hand side to make room for colorbar.

    Even for axes without colorbar in a multi-pane plot, it is still useful to call this function
    so that the axes will align with those with colorbars.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes object.
    """
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left, bottom, width*0.9, height])

def add_colorbar(fig, ax, cmap, vmin, vmax, nlevels=10, fontsize=12, units=None):
    """
	Add a colorbar to the right-hand side of an axes.

    This function adds a colorbar to the provided axes, using the specified
    colormap and value range. The number of levels, font size, and unit label
    can be customized.

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure object.
        ax (matplotlib.axes.Axes): Matplotlib axes object.
        cmap (matplotlib.colors.Colormap): Colormap to use for the colorbar.
        vmin (float): Minimum value for the colorbar.
        vmax (float): Maximum value for the colorbar.
        nlevels (int, optional): Number of color levels. Defaults to 10.
        fontsize (int, optional): Font size for tick labels and unit label. Defaults to 12.
        units (str, optional): Unit label to display as the colorbar title. Defaults to None.

    Returns:
        matplotlib.colorbar.Colorbar: The created colorbar object.
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
    return cbar

def arrowhead_xy(x1, x2, y1, y2, hw, hl):
    """
	Given a line segment from (x1,y1) to (x2,y2), return the segments that draw an arrow head (for plotting vectors).

    Args:
        x1 (float): X-coordinate of the start point of line segment.
        x2 (float): X-coordinate of the end point of line segment.
        y1 (float): Y-coordinate of the start point of line segment.
        y2 (float): Y-coordinate of the end point of line segment.
        hw (float): Width of the arrow head.
        hl (float): Length of the arrow head.

    Returns:
        list: X-coordinates of the additional line segments forming the arrowhead.
        list: Y-coordinates of the additional line segments forming the arrowhead.
    """
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
    """
    Draw a legend box with reference vector and units string.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes object.
        xr (float): X-coordinate of the center of the legend box
        yr (float): Y-coordinate of the center of the legend box
        V (float): Velocity scale to be shown as the reference vector.
        L (float): Length of the reference vector.
        hw (float): Width of the arrowhead.
        hl (float): Length of the arrowhead.
        refcolor (str or tuple): Color of the background in the legend box.
        linecolor (str or tuple): Color of the reference vector.
        ref_units (str, optional): Unit label to be shown, default is ''.
    """
    ##draw a box
    xb = [xr-L*1.3, xr-L*1.3, xr+L*1.3, xr+L*1.3, xr-L*1.3]
    yb = [yr+L/2, yr-L, yr-L, yr+L/2, yr+L/2]
    ax.fill(xb, yb, color=refcolor, zorder=6)
    ax.plot(xb, yb, color='k', zorder=6)
    ##draw the reference vector
    ax.plot([xr-L/2, xr+L/2], [yr, yr], color=linecolor, zorder=8)
    ax.fill(*arrowhead_xy(xr+L/2, xr-L/2, yr, yr, hw, hl), color=linecolor, zorder=8)
    ##add unit string annotation below the vector
    ax.text(xr, yr-L/2, f"{V} {ref_units}", color=linecolor, ha='center', va='center', zorder=8)

def draw_line(ax, data, linecolor, linewidth, linestyle, zorder):
    """
    Draw line segments.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to draw the lines.
        data (dict): Dictionary containing:
            - 'xy' (list of arrays): Coordinates of points.
            - 'parts' (list of lists): Indices indicating segment divisions.
        linecolor (str or tuple): Color of the line.
        linewidth (float): Width of the line.
        linestyle (str): Style of the line (e.g., '-', '--', ':').
        zorder (int): Drawing order (higher numbers are drawn on top).
    """
    xy = data['xy']
    parts = data['parts']
    for i in range(len(xy)):
        for j in range(len(parts[i])-1): ##plot separate segments if multi-parts
            ax.plot(*zip(*xy[i][parts[i][j]:parts[i][j+1]]), color=linecolor, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
        ax.plot(*zip(*xy[i][parts[i][-1]:]), color=linecolor, linewidth=linewidth, linestyle=linestyle, zorder=zorder)

def draw_patch(ax, data, color, zorder):
    """
    Draw a filled patch.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to draw the lines.
        data (dict): Dictionary containing:
            - 'xy' (list of arrays): Coordinates of points.
            - 'parts' (list of lists): Indices indicating segment divisions.
        color (str or tuple): Color of the patch
        zorder (int): Drawing order (higher numbers are drawn on top).
    """
    xy = data['xy']
    parts = data['parts']
    for i in range(len(xy)):
        code = [Path.LINETO] * len(xy[i])
        for j in parts[i]:  ##make discontinuous patch if multi-parts
            code[j] = Path.MOVETO
        ax.add_patch(PathPatch(Path(xy[i], code), facecolor=color, edgecolor=color, linewidth=0.1, zorder=zorder))
