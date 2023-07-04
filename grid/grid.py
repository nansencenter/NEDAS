###  Yue Ying 2023
###  Grid class to handle 2D field defined on a regular grid or unstructured mesh
###
###  "rotate_vector", "interp" methods for converting a field to dst_grid
###  To speed up, the rotate and interpolate weights are computed once and stored.
###  Some functions are adapted from nextsim-tools/pynextsim:
###  lib.py:transform_vectors, irregular_grid_interpolator.py
###
###  Some basic map plotting to visualize the field
###  "plot_field", "plot_vector", and "plot_land"
###
###  See NEDAS/tutorials/grid_convert for some examples.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.tri import Triangulation
from pyproj import Proj, Geod
import shapefile
import os, inspect
from functools import cached_property

class Grid(object):
    def __init__(self,
                 proj,              ##pyproj, from lon,lat to x,y
                 x, y,              ##x, y coords, same shape as a 2D field
                 regular=True,      ##regular grid or unstructured mesh
                 cyclic_dim=None,   ##cyclic dimension(s): 'x', 'y' or 'xy'
                 pole_dim=None,     ##dimension with poles: 'x' or 'y'
                 pole_index=None,   ##tuple for the pole index(s) in pole_dim
                 triangles=None,    ##triangles for unstructured mesh
                 dst_grid=None      ##Grid obj to convert a field towards
                 ):
        assert x.shape == y.shape, "x, y shape does not match"
        self.proj = proj
        self.x = x
        self.y = y
        self.regular = regular
        self.cyclic_dim = cyclic_dim
        self.pole_dim = pole_dim
        self.pole_index = pole_index

        if self.proj_name == 'longlat':
            self.x = np.mod(self.x + 180., 360.) - 180.

        ##boundary corners of the grid
        self.xmin = np.min(self.x)
        self.xmax = np.max(self.x)
        self.ymin = np.min(self.y)
        self.ymax = np.max(self.y)

        if not regular:
            ##Generate triangulation, if tiangles are provided its very quick,
            ##otherwise Triangulation will generate one, but slower.
            self.tri = Triangulation(x, y, triangles=triangles)

        if dst_grid is not None:
            self.dst_grid = dst_grid

    @classmethod
    def regular_grid(cls, proj, xstart, xend, ystart, yend, dx, centered=False, **kwargs):
        self = cls.__new__(cls)
        xcoord = np.arange(xstart, xend, dx)
        ycoord = np.arange(ystart, yend, dx)
        x, y = np.meshgrid(xcoord, ycoord)
        if centered:
            x += 0.5*dx  ##move coords to center of grid box
            y += 0.5*dx
        self.__init__(proj, x, y, regular=True, **kwargs)
        return self

    @cached_property
    def proj_name(self):
        if hasattr(self.proj, 'name'):
            return self.proj.name
        return ''

    @cached_property
    def proj_ellps(self):
        if hasattr(self.proj, 'definition'):
            for e in self.proj.definition.split():
                es = e.split('=')
                if es[0]=='ellps':
                    return es[1]
        return 'WGS84'

    def _mesh_dx(self):
        t = self.tri.triangles
        s1 = np.sqrt((self.x[t][:,0]-self.x[t][:,1])**2+(self.y[t][:,0]-self.y[t][:,1])**2)
        s2 = np.sqrt((self.x[t][:,0]-self.x[t][:,2])**2+(self.y[t][:,0]-self.y[t][:,2])**2)
        s3 = np.sqrt((self.x[t][:,2]-self.x[t][:,1])**2+(self.y[t][:,2]-self.y[t][:,1])**2)
        sa = (s1 + s2 + s3)/3
        e = 0.3
        inds = np.logical_and(np.logical_and(np.abs(s1-sa) < e*sa, np.abs(s2-sa) < e*sa), np.abs(s3-sa) < e*sa)
        return np.mean(sa[inds])

    @cached_property
    def nx(self):
        return self.x.shape[1]

    @cached_property
    def ny(self):
        return self.x.shape[0]

    @cached_property
    def Lx(self):
        return self.nx * self.dx

    @cached_property
    def Ly(self):
        return self.ny * self.dy

    @cached_property
    def npoints(self):
        if self.regular:
            return self.nx * self.ny
        else:
            return self.tri.triangles.shape[0]

    @cached_property
    def dx(self):
        if self.regular:
            return (self.xmax - self.xmin) / (self.nx - 1)
        else:
            return self._mesh_dx()

    @cached_property
    def dy(self):
        if self.regular:
            return (self.ymax - self.ymin) / (self.ny - 1)
        else:
            return self._mesh_dx()

    @cached_property
    def mfx(self):
        if self.proj_name == 'longlat':
            ##long/lat grid doesn't have units in meters, so will not use map factors
            return np.ones(self.x.shape)
        else:
            ##map factor: ratio of (dx, dy) to their actual distances on the earth.
            geod = Geod(ellps=self.proj_ellps)
            lon, lat = self.proj(self.x, self.y, inverse=True)
            lon1x, lat1x = self.proj(self.x+self.dx, self.y, inverse=True)
            _,_,gcdx = geod.inv(lon, lat, lon1x, lat1x)
            return self.dx / gcdx

    @cached_property
    def mfy(self):
        if self.proj_name == 'longlat':
            ##long/lat grid doesn't have units in meters, so will not use map factors
            return np.ones(self.x.shape)
        else:
            ##map factor: ratio of (dx, dy) to their actual distances on the earth.
            geod = Geod(ellps=self.proj_ellps)
            lon, lat = self.proj(self.x, self.y, inverse=True)
            lon1y, lat1y = self.proj(self.x, self.y+self.dy, inverse=True)
            _,_,gcdy = geod.inv(lon, lat, lon1y, lat1y)
            return self.dy / gcdy

    ##destination grid for convert, interp, rotate_vector methods
    @property
    def dst_grid(self):
        return self._dst_grid

    @dst_grid.setter
    def dst_grid(self, grid):
        assert isinstance(grid, Grid), "dst_grid should be a Grid instance"
        self._dst_grid = grid

        ##rotation of vector field from self.proj to dst_grid.proj
        self._set_rotation_matrix()

        ##prepare inside, vertices and weights for interpolation
        ##dst_grid points projected back to self.proj, call it x,y
        x, y = self._proj_from(grid.x, grid.y)
        x_ = x.flatten()
        y_ = y.flatten()
        inside, vertices, in_coords = self._find_index(x_, y_)
        self.inside = inside
        self.vertices = vertices
        self.interp_weights = self._interp_weights(inside, vertices, in_coords)

    def wrap_cyclic_xy(self, x_, y_):
    ##if input x_,y_ is outside of domain, wrap around for cyclic boundary condition
        if self.cyclic_dim is not None:
            xi = self.x[0, :]
            yi = self.y[:, 0]
            for d in self.cyclic_dim:
                if d=='x':
                    x_ = np.mod(x_ - xi.min(), self.Lx) + xi.min()
                elif d=='y':
                    y_ = np.mod(y_ - yi.min(), self.Ly) + yi.min()
        return x_, y_

    def _find_index(self, x_, y_):
        ##for each point x,y find the grid box vertices that it falls in
        ##and the internal coordinate that pinpoint its location
        x_ = np.array(x_).flatten()
        y_ = np.array(y_).flatten()

        if self.regular:
            xi = self.x[0, :]
            yi = self.y[:, 0]
            idx = np.arange(self.nx)
            idy = np.arange(self.ny)

            ##lon: proj works only for lon=-180:180
            if self.proj_name == 'longlat':
                xi = np.mod(xi + 180., 360.) - 180.
                x_ = np.mod(x_ + 180., 360.) - 180.

            ###account for cyclic dim, when points drop "outside" then wrap around
            x_, y_ = self.wrap_cyclic_xy(x_, y_)

            ##sort the index to monoticially increasing
            idx_ = np.argsort(xi)
            xi_ = xi[idx_]
            idy_ = np.argsort(yi)
            yi_ = yi[idy_]

            ##pad cyclic dimensions with additional row for the wrap-around point
            if self.cyclic_dim is not None:
                for d in self.cyclic_dim:
                    if d=='x':
                        if xi_[0]+self.Lx not in xi_:
                            xi_ = np.hstack((xi_, xi_[0] + self.Lx))
                            idx_ = np.hstack((idx_, idx_[0]))
                    elif d=='y':
                        if yi_[0]+self.Ly not in yi_:
                            yi_ = np.hstack((yi_, yi_[0] + self.Ly))
                            idy_ = np.hstack((idy_, idy_[0]))

            ##now find the index near the given x_,y_ coordinates
            i = np.array(np.searchsorted(xi_, x_, side='right'))
            j = np.array(np.searchsorted(yi_, y_, side='right'))
            inside = ~np.logical_or(np.logical_or(i==len(xi_), i==0), np.logical_or(j==len(yi_), j==0))

            ##vertices (A, B, C, D) for the rectangular grid box
            ##internal coordinate (in_x, in_y) pinpoint the x_,y_ location inside the grid box
            ##with values range [0, 1)
            ##(0,1) D----+------C (1,1)
            ##      |    |      |
            ##      |    |      |
            ##      +----*------in_y
            ##      |    |      |
            ##(0,0) A---in_x----B (1,0)
            vertices = np.zeros(x_[inside].shape+(4,), dtype=int)
            vertices[:, 0] = idy_[j[inside]-1] * self.nx + idx_[i[inside]-1]
            vertices[:, 1] = idy_[j[inside]-1] * self.nx + idx_[i[inside]]
            vertices[:, 2] = idy_[j[inside]] * self.nx + idx_[i[inside]]
            vertices[:, 3] = idy_[j[inside]] * self.nx + idx_[i[inside]-1]
            in_coords = np.zeros(x_[inside].shape+(2,), dtype=np.float64)
            in_coords[:, 0] = (x_[inside] - xi_[i[inside]-1]) / (xi_[i[inside]] - xi_[i[inside]-1])
            in_coords[:, 1] = (y_[inside] - yi_[j[inside]-1]) / (yi_[j[inside]] - yi_[j[inside]-1])

        else:
            ##for irregular mesh, use tri_finder to find index
            tri_finder = self.tri.get_trifinder()
            triangle_map = tri_finder(x_, y_)
            inside = ~(triangle_map < 0)

            ##internal coords are the barycentric coords (in1, in2, in3) in a triangle
            ##note: in1+in2+in3=1 so only in1 and in2 stored in in_coords
            ##     (0,0,1) C\
            ##            / | \
            ##          in3 |  .in2
            ##          /  :* .   \
            ##         / .   |   .  \
            ##(1,0,0) A-----in1------B (0,1,0)
            vertices = self.tri.triangles[triangle_map[inside], :]

            ##transform matrix for barycentric coords computation
            a = self.x[vertices[:,0]] - self.x[vertices[:,2]]
            b = self.x[vertices[:,1]] - self.x[vertices[:,2]]
            c = self.y[vertices[:,0]] - self.y[vertices[:,2]]
            d = self.y[vertices[:,1]] - self.y[vertices[:,2]]
            det = a*d-b*c
            t_matrix = np.zeros((len(vertices), 3, 2))
            t_matrix[:,0,0] = d/det
            t_matrix[:,0,1] = -b/det
            t_matrix[:,1,0] = -c/det
            t_matrix[:,1,1] = a/det
            t_matrix[:,2,0] = self.x[vertices[:,2]]
            t_matrix[:,2,1] = self.y[vertices[:,2]]
            # ##get barycentric coords, according to https://en.wikipedia.org/wiki/
            # ##Barycentric_coordinate_system#Barycentric_coordinates_on_triangles,
            delta = np.array([x_[inside], y_[inside]]).T - t_matrix[:,2,:]
            in_coords = np.einsum('njk,nk->nj', t_matrix[:,:2,:], delta)
        return inside, vertices, in_coords

    def _proj_to(self, x, y):
        lon, lat = self.proj(x, y, inverse=True)
        x_, y_ = self._dst_grid.proj(lon, lat)
        x_, y_ = self._dst_grid.wrap_cyclic_xy(x_, y_)
        return x_, y_

    def _proj_from(self, x, y):
        lon, lat = self._dst_grid.proj(x, y, inverse=True)
        x_, y_ = self.proj(lon, lat)
        x_, y_ = self.wrap_cyclic_xy(x_, y_)
        return x_, y_

    def _set_rotation_matrix(self):
        self.rotate_matrix = np.zeros((4,)+self.x.shape)
        ##self.x,y corresponding coordinates in dst_proj, call them x,y
        x, y = self._proj_to(self.x, self.y)

        ##find small increments in x,y due to small changes in self.x,y in dst_proj
        eps = 0.1 * self.dx    ##grid spacing is specified in Grid object
        xu, yu = self._proj_to(self.x + eps, self.y      )  ##move a bit in x dirn
        xv, yv = self._proj_to(self.x      , self.y + eps)  ##move a bit in y dirn

        np.seterr(invalid='ignore')  ##will get nan at poles
        dxu = xu-x
        dyu = yu-y
        dxv = xv-x
        dyv = yv-y
        hu = np.hypot(dxu, dyu)
        hv = np.hypot(dxv, dyv)
        self.rotate_matrix[0, :] = dxu/hu
        self.rotate_matrix[1, :] = dxv/hv
        self.rotate_matrix[2, :] = dyu/hu
        self.rotate_matrix[3, :] = dyv/hv

    def _fill_pole_void(self, fld):
        if self.pole_dim == 'x':
            for i in self.pole_index:
                if i==0:
                    fld[:, 0] = np.mean(fld[:, 1])
                if i==-1:
                    fld[:, -1] = np.mean(fld[:, -2])
        if self.pole_dim == 'y':
            for i in self.pole_index:
                if i==0:
                    fld[0, :] = np.mean(fld[1, :])
                if i==-1:
                    fld[-1, :] = np.mean(fld[-2, :])
        return fld

    def rotate_vectors(self, vec_fld):
        u = vec_fld[0, :]
        v = vec_fld[1, :]

        rw = self.rotate_matrix
        u_rot = rw[0, :]*u + rw[1, :]*v
        v_rot = rw[2, :]*u + rw[3, :]*v

        u_rot = self._fill_pole_void(u_rot)
        v_rot = self._fill_pole_void(v_rot)

        vec_fld_rot = np.full(vec_fld.shape, np.nan)
        vec_fld_rot[0, :] = u_rot
        vec_fld_rot[1, :] = v_rot
        return vec_fld_rot

    ###utility functions for interpolation (low->high resolution; refine)
    def _interp_weights(self, inside, vertices, in_coords):
        if self.regular:
            ##compute bilinear interp weights
            interp_weights = np.zeros(vertices.shape)
            interp_weights[:, 0] =  (1-in_coords[:, 0]) * (1-in_coords[:, 1])
            interp_weights[:, 1] =  in_coords[:, 0] * (1-in_coords[:, 1])
            interp_weights[:, 2] =  in_coords[:, 0] * in_coords[:, 1]
            interp_weights[:, 3] =  (1-in_coords[:, 0]) * in_coords[:, 1]
        else:
            interp_weights = np.hstack((in_coords, 1-in_coords.sum(axis=1, keepdims=True)))
        return interp_weights

    def interp(self, fld, x, y, calc_weight=True, method='linear'):
        ##use precalculated weights from _set_interp_weights
        ##otherwise compute the weights for the given x,y
        if calc_weight:
            inside, vertices, in_coords = self._find_index(x, y)
            weights = self._interp_weights(inside, vertices, in_coords)
        else:
            inside = self.inside
            vertices = self.vertices
            weights = self.interp_weights
            x = self._dst_grid.x
            y = self._dst_grid.y

        fld_interp = np.full(np.array(x).flatten().shape, np.nan)

        if method == 'nearest':
            # find the node of the triangle with the maximum weight
            v = vertices[np.arange(len(weights), dtype=int), np.argmax(weights, axis=1)]
            fld_interp[inside] = fld.flatten()[v]

        elif method == 'linear':
            # sum over the weights for each node of triangle
            fld_interp[inside] = np.einsum('nj,nj->n', np.take(fld.flatten(), vertices), weights)

        else:
            raise ValueError("'method' should be 'nearest' or 'linear'")

        return fld_interp.reshape(np.array(x).shape)

    ###utility functions for convolution/smoothing (high->low resolution; coarsen)


    ### Method to convert from self.proj, x, y to dst_grid coordinate systems:
    ###  Steps: 1. rotate vectors in self.proj to dst_grid.proj
    ###         2. interpolate fld from (self.x, self.y) to dst_grid.(x, y)->self.proj
    def convert(self, fld, is_vector=False, method='linear'):
        x, y = self._proj_from(self._dst_grid.x, self._dst_grid.y)

        if is_vector:
            assert fld.shape[0] == 2, "vector field should have first dim==2, for u,v component"
            assert fld.shape[1:] == self.x.shape, "vector field shape does not match x1"

            ##vector field needs to rotate to dst_grid.proj before interp
            fld = self.rotate_vectors(fld)

            fld_out = np.full((2,)+x.shape, np.nan)
            ##interp for u, v component
            fld_out[0, :] = self.interp(fld[0, :], x, y, False, method)
            fld_out[1, :] = self.interp(fld[1, :], x, y, False, method)

        else:
            ##if fld is scalar, just interpolate
            if fld.shape == self.x.shape:
                fld_out = self.interp(fld, x, y, False, method)

            elif not self.regular and len(fld) == self.npoints:
                tri_finder = self.tri.get_trifinder()
                triangle_map = tri_finder(x, y)
                inside = ~(triangle_map < 0)
                fld_out = np.full(x.shape, np.nan)
                fld_out[inside] = fld[triangle_map[inside]]

            else:
                raise ValueError("field shape does not match x1.shape, or number of triangle elements in proj1")
        return fld_out

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
    def collect_shape_data(self, shapes):
        data = {'xy':[], 'parts':[]}
        for shape in shapes:
            if len(shape.points) > 0:
                xy = []
                inside = []
                for lon, lat in shape.points:
                    x, y = self.proj(lon, lat)
                    xy.append((x, y))
                    inside.append((self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax))
                ##if any point in the polygon lies inside the grid, need to plot it.
                if any(inside):
                    data['xy'].append(xy)
                    data['parts'].append(shape.parts)
        return data

    @cached_property
    def land_data(self):
        ##prepare data to show the land area (with plt.fill/plt.plot)
        ##downloaded from https://www.naturalearthdata.com
        path = os.path.split(inspect.getfile(self.__class__))[0]
        sf = shapefile.Reader(os.path.join(path, 'ne_50m_coastline.shp'))
        shapes = sf.shapes()

        ##Some cosmetic tweaks of the shapefile for some Canadian coastlines
        shapes[1200].points = shapes[1200].points + shapes[1199].points[1:]
        shapes[1199].points = []
        shapes[1230].points = shapes[1230].points + shapes[1229].points[1:] + shapes[1228].points[1:] + shapes[1227].points[1:]
        shapes[1229].points = []
        shapes[1228].points = []
        shapes[1227].points = []
        shapes[1233].points = shapes[1233].points + shapes[1234].points
        shapes[1234].points = []
        shapes[1234].points = []

        return self.collect_shape_data(shapes)

    @cached_property
    def river_data(self):
        ##river features
        sf = shapefile.Reader(os.path.join(path, 'ne_50m_rivers.shp'))
        shapes = sf.shapes()
        return self.collect_shape_data(shapes)

    @cached_property
    def lake_data(self):
        ##lake features
        sf = shapefile.Reader(os.path.join(path, 'ne_50m_lakes.shp'))
        shapes = sf.shapes()
        return self.collect_shape_data(shapes)

    def llgrid_xy(self, dlon, dlat):
        self.dlon = dlon
        self.dlat = dlat
        ##prepare a lat/lon grid to plot as guidelines
        ##  dlon, dlat: spacing of lon/lat grid
        llgrid_xy = []
        for lon in np.arange(-180, 180, dlon):
            xy = []
            inside = []
            for lat in np.arange(-89.9, 90, 0.1):
                x, y = self.proj(lon, lat)
                xy.append((x, y))
                inside.append((self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax))
            if any(inside):
                llgrid_xy.append(xy)
        for lat in np.arange(-90, 90+dlat, dlat):
            xy = []
            inside = []
            for lon in np.arange(-180, 180, 0.1):
                x, y = self.proj(lon, lat)
                xy.append((x, y))
                inside.append((self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax))
            if any(inside):
                llgrid_xy.append(xy)
        return llgrid_xy

    def plot_field(self, ax, fld,  vmin=None, vmax=None, cmap='viridis'):
        if vmin is None:
            vmin = np.nanmin(fld)
        if vmax is None:
            vmax = np.nanmax(fld)

        if self.regular:
            x = self.x
            y = self.y
            ##in case of lon convention 0:360, need to reorder so that x is monotonic
            if self.proj_name == 'longlat':
                ind = np.argsort(x[0,:])
                x = np.take(x, ind, axis=1)
                fld = np.take(fld, ind, axis=1)
            c = ax.pcolor(x, y, fld, vmin=vmin, vmax=vmax, cmap=cmap)
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
        if V is None:
            V = 0.33 * np.nanmax(np.abs(u))
        if L is None:
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
            ut = self.interp(u, xtraj[:,t], ytraj[:,t])
            vt = self.interp(v, xtraj[:,t], ytraj[:,t])
            ###velocity should be in physical units, to plot the right length on projection
            ###we use the map factors to scale distance units
            mfx = self.interp(self.mfx, xtraj[:,t], ytraj[:,t])
            mfy = self.interp(self.mfy, xtraj[:,t], ytraj[:,t])
            ut = ut * mfx
            vt = vt * mfy
            ###update traj position
            xtraj[:, t+1] = xtraj[:, t] + ut * dt
            ytraj[:, t+1] = ytraj[:, t] + vt * dt
            leng = leng + np.sqrt(ut**2 + vt**2) * dt

        ##plot the vector lines
        hl = headlength * L
        hw = headwidth * L
        def arrowhead_xy(x1, x2, y1, y2):
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
            ##find the length scale at the ref point
            Lr = L
            mfxr = self.interp(self.mfx, xr, yr)
            if not np.isnan(mfxr):
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
                  showriver=False, rivercolor='c',
                  showgrid=True, dlon=20, dlat=5):

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

        ###plot the coastline to indicate land area
        if color is not None:
            draw_patch(ax, self.land_data, color=color, zorder=0)
        if linecolor is not None:
            draw_line(ax, self.land_data, linecolor=linecolor, linewidth=linewidth, linestyle='-', zorder=8)
        if showriver:
            draw_line(ax, self.river_data, linecolor=rivercolor, linewidth=0.5, linestyle='-', zorder=1)
            draw_patch(ax, self.lake_data, color=rivercolor, zorder=1)

        ###add reference lonlat grid on map
        if showgrid:
            for xy in self.llgrid_xy(dlon, dlat):
                ax.plot(*zip(*xy), color='k', linewidth=0.5, linestyle=':', zorder=4)

        ##set the correct extent of plot
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

