import numpy as np

###convert field from native projection to reference grid projection
from .converter import Converter

###Define reference grid for analysis
def uniform_grid(xstart, xend, ystart, yend, dx):
    '''
    Make a uniform grid in Polar stereographic grid
    inputs: xstart, xend, ystart, yend: start and end points in x and y directions (in meters)
            dx: grid spacing in meters
    output: x[:, :], y[:, :] in meters
    '''
    xcoord = np.arange(xstart, xend, dx)
    ycoord = np.arange(ystart, yend, dx)
    y, x = np.meshgrid(ycoord, xcoord)
    x += 0.5*dx  ##move coords to center of grid box
    y += 0.5*dx
    return x, y

def theta(x, y):
    nx, ny = x.shape
    theta = np.zeros((nx, ny))
    for j in range(ny):
        dx = x[1,j] - x[0,j]
        dy = x[1,j] - y[0,j]
        theta[0,j] = np.arctan2(dy,dx)
        for i in range(1, nx-1):
            dx = x[i+1,j] - x[i-1,j]
            dy = y[i+1,j] - y[i-1,j]
            theta[i,j] = np.arctan2(dy,dx)
        dx = x[nx-1,j] - x[nx-2,j]
        dy = y[nx-1,j] - y[nx-2,j]
        theta[nx-1,j] = np.arctan2(dy,dx)
    return theta

def corners(x):
    nx, ny = x.shape
    xt = np.zeros((nx+1, ny+1))
    ##use linear interp in interior
    xt[1:nx, 1:ny] = 0.25*(x[1:nx, 1:ny] + x[1:nx, 0:ny-1] + x[0:nx-1, 1:ny] + x[0:nx-1, 0:ny-1])
    ##use 2nd-order polynomial extrapolat along borders
    xt[0, :] = 3*xt[1, :] - 3*xt[2, :] + xt[3, :]
    xt[nx, :] = 3*xt[nx-1, :] - 3*xt[nx-2, :] + xt[nx-3, :]
    xt[:, 0] = 3*xt[:, 1] - 3*xt[:, 2] + xt[:, 3]
    xt[:, ny] = 3*xt[:, ny-1] - 3*xt[:, ny-2] + xt[:, ny-3]
    ##make corners into new dimension
    x_corners = np.zeros((nx, ny, 4))
    x_corners[:, :, 0] = xt[0:nx, 0:ny]
    x_corners[:, :, 1] = xt[0:nx, 1:ny+1]
    x_corners[:, :, 2] = xt[1:nx+1, 1:ny+1]
    x_corners[:, :, 3] = xt[1:nx+1, 0:ny]
    return x_corners

##some basic map plotting without the need for installing cartopy
def coastline_xy(proj):
    ##prepare data to show the land area (with plt.fill/plt.plot)
    ##  usage: for xy in coastline_xy:
    ##             x, y = zip(*xy)
    ##             ax.fill(x, y, color=fillcolor) #(optional fill)
    ##             ax.plot(x, y, 'k', linewidth=0.5)  ##solid coast line
    import shapefile
    import os

    ## downloaded from https://www.naturalearthdata.com
    sf = shapefile.Reader(os.path.join(__path__[0],'ne_50m_coastline.shp'))
    shapes = sf.shapes()

    ##Some cosmetic treaks of the shapefile:
    ## get rid of the Caspian Sea
    shapes[1387].points = shapes[1387].points[391:]
    ## merge some Canadian coastlines shape
    shapes[1200].points = shapes[1200].points + shapes[1199].points[1:]
    shapes[1199].points = []
    shapes[1230].points = shapes[1230].points + shapes[1229].points[1:] + shapes[1228].points[1:] + shapes[1227].points[1:]
    shapes[1229].points = []
    shapes[1228].points = []
    shapes[1227].points = []
    shapes[1233].points = shapes[1233].points + shapes[1234].points
    shapes[1234].points = []

    coastline_xy = []

    for shape in shapes:
        xy = []
        for point in shape.points[:]:
            lon, lat = point
            if lat>20:  ## only process northern region
                xy.append(proj(lon, lat))
        if len(xy)>0:
            coastline_xy.append(xy)

    return coastline_xy

def lonlat_grid_xy(proj, dlon, dlat):
    ##prepare a lat/lon grid to plot as guidelines
    ##  dlon, dlat: spacing of lon/lat grid
    import numpy as np

    lonlat_grid_xy = []
    for lon in np.arange(-180, 180, dlon):
        xy = []
        for lat in np.arange(0, 90, 0.1):
            xy.append(proj(lon, lat))
        lonlat_grid_xy.append(xy)
    for lat in np.arange(0, 90+dlat, dlat):
        xy = []
        for lon in np.arange(-180, 180, 0.1):
            xy.append(proj(lon, lat))
        lonlat_grid_xy.append(xy)

    return lonlat_grid_xy

def add_land(ax, proj, facecolor=[.7, .7, .7], linecolor='k', linewidth=1):
    for coast_xy in coastline_xy(proj):
        if facecolor != None:
            ax.fill(*zip(*coast_xy), color=facecolor)
        if linecolor != None:
            ax.plot(*zip(*coast_xy), color=linecolor, linewidth=linewidth)

def add_lonlat_grid(ax, proj, dlon=20, dlat=5, linecolor='k', linewidth=0.5, linestyle=':'):
    for grid_xy in lonlat_grid_xy(proj, dlon, dlat):
        ax.plot(*zip(*grid_xy), color=linecolor, linewidth=linewidth, linestyle=linestyle)

def set_extent(ax, xstart, xend, ystart, yend):
    ax.set_xlim(xstart, xend)
    ax.set_ylim(ystart, yend)

##
