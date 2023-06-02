import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
from matplotlib.tri import Triangulation

###handling 2d horizontal grid/mesh`
###1. regular grid, x, y (1d coordinates), proj
###  cyclic_dims (tuple)
###  fld[0:nx, 0:ny] defined on x[0:nx] and y[0:nx]
###2. irregular grid, triangular mesh nodes; x, y, tri, proj
###  fld[0:nn] defined on x[0:nn], y[0:nn], nn = number of nodes
###  triangulation tri.triangles[num_triangles, 3] for indices for each vertex
##TODO:
###map_factor (mx, my) = (dx, dy)/(grid spacing on earth)
###landmask (from model)
##get distance?


class Grid(object):
    def __init__(self,
                 proj,              ##pyproj.Proj object
                 x, y,              ##x, y coordinates, same shape as a 2D field
                 dx, dy,            ##grid spacing
                 regular=True,      ##if grid is regular, if not it's unstructured mesh
                 cyclic_dim=None,   ##dimension(s) with cyclic boundary: 'x', 'y' or 'xy'
                 pole_dim=None,     ##specify the dimension with poles: 'x' or 'y'
                 pole_index=None,   ##tuple for the pole index(s) in grid
                 triangles=None,    ##triangles for unstructured mesh
                 test=None
                 ):
        assert isinstance(proj, Proj), "proj is not pyproj.Proj instance"
        assert x.shape == y.shape, "x, y shape does not match"
        self.proj = proj
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.regular = regular
        self.cyclic_dim = cyclic_dim
        self.pole_dim = pole_dim
        self.pole_index = pole_index

        if not regular:
            ##Generate triangulation, if tiangles are provided its very quick,
            ##otherwise Triangulation will generate one, but slower.
            self.tri = Triangulation(x, y, triangles=triangles)

        ##some properties for stereographic grid
        if self.proj.name == 'stere' and self.regular:
            self._set_theta()

        # self.landmask
        self._set_land_xy()  ##prepare land data for plot_var_on_map


    def init_regular_grid(xstart, xend, ystart, yend, dx):
        xcoord = np.arange(xstart, xend, dx)
        ycoord = np.arange(ystart, yend, dx)
        y, x = np.meshgrid(ycoord, xcoord)
        x += 0.5*dx  ##move coords to center of grid box
        y += 0.5*dx
        return x, y


###TODO: nx, ny swap
    def _set_theta(self):
        x = self.x
        y = self.y
        nx, ny = x.shape
        self.theta = np.zeros((nx, ny))
        for j in range(ny):
            dx = x[1,j] - x[0,j]
            dy = x[1,j] - y[0,j]
            self.theta[0,j] = np.arctan2(dy,dx)
            for i in range(1, nx-1):
                dx = x[i+1,j] - x[i-1,j]
                dy = y[i+1,j] - y[i-1,j]
                self.theta[i,j] = np.arctan2(dy,dx)
            dx = x[nx-1,j] - x[nx-2,j]
            dy = y[nx-1,j] - y[nx-2,j]
            self.theta[nx-1,j] = np.arctan2(dy,dx)

    def get_corners(self, fld):
        assert fld.shape == self.x.shape, "fld shape does not match x,y"
        nx, ny = fld.shape
        fld_ = np.zeros((nx+1, ny+1))
        ##use linear interp in interior
        fld_[1:nx, 1:ny] = 0.25*(fld[1:nx, 1:ny] + fld[1:nx, 0:ny-1] + fld[0:nx-1, 1:ny] + fld[0:nx-1, 0:ny-1])
        ##use 2nd-order polynomial extrapolat along borders
        fld_[0, :] = 3*fld_[1, :] - 3*fld_[2, :] + fld_[3, :]
        fld_[nx, :] = 3*fld_[nx-1, :] - 3*fld_[nx-2, :] + fld_[nx-3, :]
        fld_[:, 0] = 3*fld_[:, 1] - 3*fld_[:, 2] + fld_[:, 3]
        fld_[:, ny] = 3*fld_[:, ny-1] - 3*fld_[:, ny-2] + fld_[:, ny-3]
        ##make corners into new dimension
        fld_corners = np.zeros((nx, ny, 4))
        fld_corners[:, :, 0] = fld_[0:nx, 0:ny]
        fld_corners[:, :, 1] = fld_[0:nx, 1:ny+1]
        fld_corners[:, :, 2] = fld_[1:nx+1, 1:ny+1]
        fld_corners[:, :, 3] = fld_[1:nx+1, 0:ny]
        return fld_corners


    ##some basic map plotting without the need for installing cartopy
    def _set_land_xy(self):
        ##prepare data to show the land area (with plt.fill/plt.plot)
        import shapefile
        import os, inspect
        path = os.path.split(inspect.getfile(self.__class__))[0]
        sf = shapefile.Reader(os.path.join(path, 'ne_50m_coastline.shp'))
        ## downloaded from https://www.naturalearthdata.com
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

        self.land_xy= []
        for shape in shapes:
            xy = []
            for point in shape.points[:]:
                lon, lat = point
                if lat>20:  ## only process northern region
                    xy.append(self.proj(lon, lat))
            if len(xy)>0:
                self.land_xy.append(xy)

    def plot_var_on_map(self, ax, var, vmin, vmax, cmap,
                        showland=True,
                        landcolor=None, landlinecolor='k', landlinewidth=1,
                        showgrid=True,
                        gridlinecolor='k', gridlinewidth=0.5, gridlinestyle=':',
                        dlon=20, dlat=5):

        if self.regular:
            ax.pcolor(self.x, self.y, var, vmin=vmin, vmax=vmax, cmap=cmap)

        else:
            ax.tripcolor(self.tri, var, vmin=vmin, vmax=vmax, cmap=cmap)

        ###plot the coastline to indicate land area
        if showland:
            for xy in self.land_xy:
                if landcolor != None:
                    ax.fill(*zip(*xy), color=landcolor)
                if landlinecolor != None:
                    ax.plot(*zip(*xy), color=landlinecolor, linewidth=landlinewidth)

        ###add reference lonlat grid on map
        if showgrid:
            ##prepare a lat/lon grid to plot as guidelines
            ##  dlon, dlat: spacing of lon/lat grid
            grid_xy = []
            for lon in np.arange(-180, 180, dlon):
                xy = []
                for lat in np.arange(0, 90, 0.1):
                    xy.append(self.proj(lon, lat))
                grid_xy.append(xy)
            for lat in np.arange(0, 90+dlat, dlat):
                xy = []
                for lon in np.arange(-180, 180, 0.1):
                    xy.append(self.proj(lon, lat))
                grid_xy.append(xy)

            for xy in grid_xy:
                ax.plot(*zip(*xy), color=gridlinecolor, linewidth=gridlinewidth, linestyle=gridlinestyle)

        ##set extent of plot
        ax.set_xlim(np.min(self.x), np.max(self.x))
        ax.set_ylim(np.min(self.y), np.max(self.y))

