import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

class Grid(object):
    def __init__(self,
                 proj,              ##pyproj, from lon,lat to x,y
                 x, y,              ##x, y coords, same shape as a 2D field
                 regular=True,      ##regular grid or unstructured mesh
                 cyclic_dim=None,   ##cyclic dimension(s): 'x', 'y' or 'xy'
                 pole_dim=None,     ##dimension with poles: 'x' or 'y'
                 pole_index=None,   ##tuple for the pole index(s) in pole_dim
                 triangles=None     ##triangles for unstructured mesh
                 ):
        assert x.shape == y.shape, "x, y shape does not match"
        self.proj = proj
        self.x = x
        self.y = y
        self.regular = regular
        self.cyclic_dim = cyclic_dim
        self.pole_dim = pole_dim
        self.pole_index = pole_index

        if self.proj_name() == 'longlat':
            assert (np.min(x) > -180. and np.max(x) <= 180.), "for interpolation to work, lon should be within (-180, 180]"

        if not regular:
            ##Generate triangulation, if tiangles are provided its very quick,
            ##otherwise Triangulation will generate one, but slower.
            self.tri = Triangulation(x, y, triangles=triangles)
        self._set_grid_spacing()

        self._set_map_factor()

        # self.landmask
        self._set_land_xy()  ##prepare land data for plot_var_on_map

    def init_regular_grid(xstart, xend, ystart, yend, dx):
        xcoord = np.arange(xstart, xend, dx)
        ycoord = np.arange(ystart, yend, dx)
        x, y = np.meshgrid(xcoord, ycoord)
        x += 0.5*dx  ##move coords to center of grid box
        y += 0.5*dx
        return x, y

    def proj_name(self):
        if hasattr(self.proj, 'name'):
            return self.proj.name
        return ''

    def proj_ellps(self):
        if hasattr(self.proj, 'definition'):
            for e in self.proj.definition.split():
                es = e.split('=')
                if es[0]=='ellps':
                    return es[1]
        return 'WGS84'

    def _set_grid_spacing(self):
        if self.regular:
            xi = self.x[0, :]
            yi = self.y[:, 0]
            self.dx = (np.max(xi) - np.min(xi)) / (len(xi) - 1)
            self.dy = (np.max(yi) - np.min(yi)) / (len(yi) - 1)
        else:
            t = self.tri.triangles
            s1 = np.sqrt((self.x[t][:,0]-self.x[t][:,1])**2+(self.y[t][:,0]-self.y[t][:,1])**2)
            s2 = np.sqrt((self.x[t][:,0]-self.x[t][:,2])**2+(self.y[t][:,0]-self.y[t][:,2])**2)
            s3 = np.sqrt((self.x[t][:,2]-self.x[t][:,1])**2+(self.y[t][:,2]-self.y[t][:,1])**2)
            sa = (s1 + s2 + s3)/3
            e = 0.3
            inds = np.logical_and(np.logical_and(np.abs(s1-sa) < e*sa, np.abs(s2-sa) < e*sa), np.abs(s3-sa) < e*sa)
            dx = np.mean(sa[inds])
            self.dx = dx
            self.dy = dx

    def _set_map_factor(self):
        if self.proj_name() == 'longlat':
            ##long/lat grid doesn't have units in meters, so will not use map factors
            self.mfx = np.ones(self.x.shape)
            self.mfy = np.ones(self.x.shape)
        else:
            ##map factor: ratio of (dx, dy) to their actual distances on the earth.
            from pyproj import Geod
            geod = Geod(ellps=self.proj_ellps())
            lon, lat = self.proj(self.x, self.y, inverse=True)
            lon1x, lat1x = self.proj(self.x+self.dx, self.y, inverse=True)
            lon1y, lat1y = self.proj(self.x, self.y+self.dy, inverse=True)
            _,_,gcdx = geod.inv(lon, lat, lon1x, lat1x)
            _,_,gcdy = geod.inv(lon, lat, lon1y, lat1y)
            self.mfx = self.dx / gcdx
            self.mfy = self.dy / gcdy

    def wrap_cyclic_xy(self, x_, y_):
        if self.cyclic_dim != None:
            xi = self.x[0, :]
            yi = self.y[:, 0]
            for d in self.cyclic_dim:
                if d=='x':
                    Lx = self.dx * len(xi)
                    x_ = np.mod(x_ - np.min(xi), Lx) + np.min(xi)
                elif d=='y':
                    Ly = self.dy * len(yi)
                    y_ = np.mod(y_ - np.min(yi), Ly) + np.min(yi)
        return x_, y_

    def find_index(self, x, y, x_, y_):
        ##find indices of x_,y_ in the coordinates x,y
        assert self.regular, "Grid.find_index only works for regular grids, use triangle_map for irregular"
        xi = x[0, :]
        yi = y[:, 0]
        assert np.all(np.diff(xi) > 0) or np.all(np.diff(xi) < 0), "x index not monotonic"
        assert np.all(np.diff(yi) > 0) or np.all(np.diff(yi) < 0), "y index not monotonic"
        ###account for cyclic dim, when points drop "outside" then wrap around
        x_, y_ = self.wrap_cyclic_xy(x_, y_)
        ###find the index, using left/right sided search when index sequence is decreasing/increasing.
        if np.all(np.diff(xi) > 0):
            id_x = np.searchsorted(xi, x_, side='right')
        else:
            id_x = len(xi) - np.searchsorted(xi[::-1], x_, side='left')
        if np.all(np.diff(yi) > 0):
            id_y = np.searchsorted(yi, y_, side='right')
        else:
            id_y = len(yi) - np.searchsorted(yi[::-1], y_, side='left')
        return id_x, id_y

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

        ##boundary corners of the grid
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        ymin = np.min(self.y)
        ymax = np.max(self.y)

        self.land_xy= []
        for shape in shapes:
            xy = []
            inside = []
            for point in shape.points[:]:
                lon, lat = point
                x, y = self.proj(lon, lat)
                xy.append((x, y))
                inside.append((xmin <= x <= xmax) and (ymin <= y <= ymax))
            ##if any point in the polygon lies inside the grid, need to plot it.
            if len(xy)>0 and any(inside):
                self.land_xy.append(xy)

    def plot_field(self, ax, fld,  vmin=None, vmax=None, cmap='jet'):
        if vmin == None:
            vmin = np.nanmin(fld)
        if vmax == None:
            vmax = np.nanmax(fld)

        if self.regular:
            c = ax.pcolor(self.x, self.y, fld, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            c = ax.tripcolor(self.tri, fld, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(c, fraction=0.025, pad=0.015, location='right')

    def plot_vectors(self, ax, vec_fld, V=None, L=None, spacing=0.5, num_steps=10, 
                     linecolor='k', linewidth=1,
                     showref=False, ref_xy=(0, 0), refcolor='w',
                     showhead=True, headwidth=0.1, headlength=0.3):
        ##plot vector field, replacing quiver
        ##options:
        ## V = velocity scale, typical velocity value in vec_fld units
        ## L = length scale, how long in x,y units do vectors with velocity V show
        ## spacing: interval among vectors, relative to L
        ## num_steps: along the vector length, how many times does velocity gets updated
        ##            with new position. =1 gets straight vectors, >1 gets curly trajs
        ## linecolor, linewidth: styling for vector lines
        ## showref, ref_xy: if plotting reference vector and where to put it
        ## showhead, ---: if plotting vector heads, and their sizes relative to L
        assert vec_fld.shape == (2,)+self.x.shape, "vector field shape mismatch with x,y"
        x = self.x
        y = self.y
        u = vec_fld[0,:]
        v = vec_fld[1,:]

        ##set typicall L, V if not defined
        if V == None:
            V = 0.33 * np.nanmax(np.abs(u))
        if L == None:
            L = 0.05 * (np.max(x) - np.min(x))

        ##start trajectories on a regular grid with spacing d
        if isinstance(spacing, tuple):
            d = (spacing[0]*L, spacing[1]*L)
        else:
            d = (spacing*L, spacing*L)
        dt = L / V / num_steps
        xo, yo = np.mgrid[x.min():x.max():d[0], y.min():y.max():d[1]]
        npoints = xo.flatten().shape[0]
        xtraj = np.full((npoints, num_steps+1,), np.nan)
        ytraj = np.full((npoints, num_steps+1,), np.nan)
        leng = np.zeros(npoints)
        xtraj[:, 0] = xo.flatten()
        ytraj[:, 0] = yo.flatten()
        for t in range(num_steps):
            ###find velocity ut,vt at traj position for step t
            if self.regular:
                idx, idy = self.find_index(x, y, xtraj[:,t], ytraj[:,t])
                inside = ~np.logical_or(np.logical_or(idy==x.shape[0], idy==0),
                                        np.logical_or(idx==x.shape[1], idx==0))
                ut = u[idy[inside], idx[inside]]
                vt = v[idy[inside], idx[inside]]
                mfx = self.mfx[idy[inside], idx[inside]]
                mfy = self.mfy[idy[inside], idx[inside]]
            else:
                tri_finder = self.tri.get_trifinder()
                triangle_map = tri_finder(xtraj[:, t], ytraj[:, t])
                inside = (triangle_map >= 0)
                ut = np.mean(u[self.tri.triangles[triangle_map[inside], :]], axis=1)
                vt = np.mean(v[self.tri.triangles[triangle_map[inside], :]], axis=1)
                mfx = np.mean(self.mfx[self.tri.triangles[triangle_map[inside], :]], axis=1)
                mfy = np.mean(self.mfy[self.tri.triangles[triangle_map[inside], :]], axis=1)
            ###velocity should be in physical units, to plot the right length on projection
            ###we use the map factors to scale distance units
            ut = ut * mfx
            vt = vt * mfy
            ###update traj position
            xtraj[inside, t+1] = xtraj[inside, t] + ut * dt
            ytraj[inside, t+1] = ytraj[inside, t] + vt * dt
            leng[inside] = leng[inside] + np.sqrt(ut**2 + vt**2) * dt

        ##plot the vector lines
        hl = headlength * L
        hw = headwidth * L
        def arrowhead_xy(x1, x2, y1, y2):
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

        for t in range(xtraj.shape[0]):
            ##plot trajectory at one output location
            ax.plot(xtraj[t, :], ytraj[t, :], color=linecolor, linewidth=linewidth, zorder=4)

            ##add vector head if traj is long and straight enough
            dist = np.sqrt((xtraj[t,0]-xtraj[t,-1])**2 + (ytraj[t,0]-ytraj[t,-1])**2)
            if showhead and hl < leng[t] < 1.6*dist:
                ax.fill(*arrowhead_xy(xtraj[t,-1], xtraj[t,-2], ytraj[t,-1],ytraj[t,-2]), color=linecolor, zorder=5)

        ##add reference vector
        if showref:
            xr, yr = ref_xy
            ##find the map factor for the ref point
            mfxr = 1.
            if self.regular:
                ix, iy = self.find_index(x, y, xr, yr)
                if 0<ix<x.shape[0] and 0<iy<y.shape[0]:
                    mfxr = self.mfx[ix, iy]
            else:
                tri_finder = self.tri.get_trifinder()
                triangle_map = tri_finder(xr, yr)
                if triangle_map >=0:
                    mfxr = self.mfx[self.tri.triangles[triangle_map, 0]]
            Lr = L * mfxr
            ##draw a box
            xb = [xr-Lr*1.3, xr-Lr*1.3, xr+Lr*1.3, xr+Lr*1.3, xr-Lr*1.3]
            yb = [yr+Lr/2, yr-Lr, yr-Lr, yr+Lr/2, yr+Lr/2]
            ax.fill(xb, yb, color=refcolor, zorder=6)
            ax.plot(xb, yb, color='k', zorder=6)
            ##draw the reference vector
            ax.plot([xr-Lr/2, xr+Lr/2], [yr, yr], color=linecolor, zorder=7)
            ax.fill(*arrowhead_xy(xr+Lr/2, xr-Lr/2, yr, yr), color=linecolor, zorder=8)

    def plot_land(self, ax, color=None, linecolor='k', linewidth=1,
                  showgrid=True, dlon=20, dlat=5):
        ###plot the coastline to indicate land area
        for xy in self.land_xy:
            if color != None:
                ax.fill(*zip(*xy), color=color, zorder=0)
            if linecolor != None:
                ax.plot(*zip(*xy), color=linecolor, linewidth=linewidth, zorder=8)

        ###add reference lonlat grid on map
        if showgrid:
            ##prepare a lat/lon grid to plot as guidelines
            ##  dlon, dlat: spacing of lon/lat grid
            grid_xy = []
            for lon in np.arange(-180, 180, dlon):
                xy = []
                for lat in np.arange(-89.9, 90, 0.1):
                    xy.append(self.proj(lon, lat))
                grid_xy.append(xy)
            for lat in np.arange(-90, 90+dlat, dlat):
                xy = []
                for lon in np.arange(-180, 180, 0.1):
                    xy.append(self.proj(lon, lat))
                grid_xy.append(xy)

            for xy in grid_xy:
                ax.plot(*zip(*xy), color='k', linewidth=0.5, linestyle=':', zorder=4)

        ax.set_xlim(np.min(self.x), np.max(self.x))
        ax.set_ylim(np.min(self.y), np.max(self.y))

