import numpy as np

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

def plot_var_on_grid(ax):
    print(ax)
    ###ax: matplotlib.pyplot.axes


