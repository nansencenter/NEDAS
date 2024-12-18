import numpy as np
import os
import inspect
import copy
from functools import cached_property
from matplotlib import colormaps
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.tri import Triangulation
from pyproj import Proj, Geod
import shapefile

class Grid(object):
    """
    Grid class to handle 2D field defined on a regular grid or unstructured mesh

    Regular grid can handle cyclic boundary conditions (longitude, e.g.) and
    the existance of poles (latitude)
    Irregular mesh variables can be defined on nodal points (vertices of triangles)
    or elements (the triangle itself)

    "rotate_vector", "interp", "coarsen" methods for converting a field to dst_grid.
    To speed up, the rotate and interpolate weights are computed once and stored.
    Some functions are adapted from nextsim-tools/pynextsim:
    lib.py:transform_vectors, irregular_grid_interpolator.py

    Grid provides some basic map plotting methods to visualize a 2D field:
    "plot_field", "plot_vector", and "plot_land"

    See NEDAS/tutorials/grid_convert.ipynb for some examples.
    """
    def __init__(self, proj, x, y, bounds=None, regular=True,
                 cyclic_dim=None, pole_dim=None, pole_index=None,
                 triangles=None, neighbors=None, dst_grid=None):
        """
        Initialize a Grid object.

        Parameters:
        - proj: pyproj.Proj
          Projection from lon,lat to x,y.

        - x, y: np.array(field.shape)
          x, y coordinates for each element in the field.

        - bounds: list[float], optional
          [xmin, xmax, ymin, ymax] boundary limits

        - regular: bool, optional
          Whether grid is regular or unstructured. Default is True (regular).

        - cyclic_dim: str, optional
          Cyclic dimension(s): 'x', 'y', or 'xy'.

        - pole_dim: str, optional
          Dimension with poles: 'x' or 'y'.

        - pole_index: tuple, optional
          Tuple for the pole index(s) in pole_dim.

        - triangles: np.array, optional
          Triangle indices for an unstructured mesh.

        - neighbors: np.array[2, 4, field.shape], optional
          Neighbor indices to handle special grid geometry
          first dim: j and i component; second dim: east, north, west, south

        - dst_grid: Grid, optional
          Grid object to convert a field towards.
        """

        assert x.shape == y.shape, "x, y shape does not match"

        if proj is None:
            self.proj = Proj('+proj=stere') ##default projection
        else:
            self.proj = proj

        ##name of the projection
        if hasattr(proj, 'name'):
            self.proj_name = proj.name
        else:
            self.proj_name = ''

        ##proj info, ellps is used in Geod for distance calculation
        self.proj_ellps = 'WGS84'
        self.proj_lon0 = 0
        self.proj_lat0 = 0
        if hasattr(proj, 'definition'):
            for e in proj.definition.split():
                es = e.split('=')
                if es[0]=='ellps':
                    self.proj_ellps = es[1]
                if es[0]=='lat_0':
                    self.proj_lat0 = np.float32(es[1])
                if es[0]=='lon_0':
                    self.proj_lon0 = np.float32(es[1])

        ##coordinates and properties of the 2D grid
        self.x = x
        self.y = y
        self.regular = regular
        self.cyclic_dim = cyclic_dim
        self.pole_dim = pole_dim
        self.pole_index = pole_index
        self.neighbors = neighbors

        if self.neighbors is not None and self.cyclic_dim is not None:
            print('neighbors already implemented, discarding cyclic_dim')
            self.cyclic_dim = None

        ##internally we use -180:180 convention for longitude
        if self.proj_name == 'longlat':
            self.x = np.mod(self.x + 180., 360.) - 180.

        ##boundary corners of the grid
        if bounds is not None:
            self.xmin, self.xmax, self.ymin, self.ymax = bounds
        else:
            self.xmin = np.min(self.x)
            self.xmax = np.max(self.x)
            self.ymin = np.min(self.y)
            self.ymax = np.max(self.y)

        if regular:
            self.nx = self.x.shape[1]
            self.ny = self.x.shape[0]
            self.dx = (self.xmax - self.xmin) / (self.nx - 1)
            self.dy = (self.ymax - self.ymin) / (self.ny - 1)
            self.Lx = self.nx * self.dx
            self.Ly = self.ny * self.dy
            self.npoints = self.nx * self.ny

        else:
            ##Generate triangulation, if tiangles are provided its very quick,
            ##otherwise Triangulation will generate one, but slower.
            self.x = self.x.flatten()
            self.y = self.y.flatten()
            self.npoints = self.x.size
            self.tri = Triangulation(self.x, self.y, triangles=triangles)
            self.tri.inds = np.arange(self.npoints)
            dx = self._mesh_dx()
            self.dx = dx
            self.dy = dx
            self.x_elem = np.mean(self.tri.x[self.tri.triangles], axis=1)
            self.y_elem = np.mean(self.tri.y[self.tri.triangles], axis=1)
            self.Lx = self.xmax - self.xmin
            self.Ly = self.ymax - self.ymin
            if self.cyclic_dim is not None:
                self._pad_cyclic_mesh_bounds()
            self._triangle_properties()

        self._dst_grid = None
        if dst_grid is not None:
            self.set_destination_grid(dst_grid)

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        if self.proj != other.proj:
            return False
        if self.x.shape != other.x.shape:
            return False
        if not np.allclose(self.x, other.x):
            return False
        if not np.allclose(self.y, other.y):
            return False
        return True

    @classmethod
    def regular_grid(cls, proj, xstart, xend, ystart, yend, dx, centered=False, **kwargs):
        """
        Create a regular grid within specified boundaries.

        Parameters:
        - proj: pyproj.Proj
          Projection from lon,lat to x,y.

        - xstart, xend, ystart, yend: float
          Boundaries of the grid in the x and y directions.

        - dx: float
          Resolution of the grid.

        - centered: bool, optional
          Toggle for grid points to be on vertices (False) or in the middle of each grid box (True).
          Default is False.

        - **kwargs:
          Additional keyword arguments.

        Returns:
        - Grid:
          A Grid object representing the regular grid.
        """
        self = cls.__new__(cls)
        dx = float(dx)
        xcoord = np.arange(xstart, xend, dx)
        ycoord = np.arange(ystart, yend, dx)
        x, y = np.meshgrid(xcoord, ycoord)
        if centered:
            x += 0.5*dx  ##move coords to center of grid box
            y += 0.5*dx
        self.__init__(proj, x, y, regular=True, **kwargs)
        return self

    @classmethod
    def random_grid(cls, proj, xstart, xend, ystart, yend, npoints, min_dist=None, **kwargs):
        """
        Create a grid with randomly positioned points within specified boundaries.

        Parameters:
        - proj: pyproj.Proj
          Projection from lon,lat to x,y.

        - xstart, xend, ystart, yend: float
          Boundaries of the grid in the x and y directions.

        - npoints: int
          Number of grid points.

        - min_dist: float, optional
          Minimal distance allowed between each pair of grid points.

        - **kwargs:
          Additional keyword arguments.

        Returns:
        - Grid:
          A Grid object representing the randomly positioned grid.
        """
        self = cls.__new__(cls)
        points = []
        ntry = 0
        while len(points) < npoints:
            xp = np.random.uniform(0, 1) * (xend - xstart) + xstart
            yp = np.random.uniform(0, 1) * (yend - ystart) + ystart
            if min_dist is not None:
                near = [p for p in points if abs(p[0]-xp) <= min_dist and abs(p[1]-yp) <= min_dist]
                ntry += 1
                if not near:
                    points.append((xp, yp))
                    ntry = 0
                if ntry > 10:
                    raise RuntimeError(f"tried 10 times, unable to insert more grid points so that dist<{min_dist}")
            else:
                points.append((xp, yp))
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        bounds = [xstart, xend, ystart, yend]
        self.__init__(proj, x, y, bounds=bounds, regular=False, **kwargs)

        return self

    def change_resolution_level(self, nlevel):
        """
        Generate a new grid with changed resolution
        Input:
         - nlevel: int
           positive: downsample grid |nlevel| times, each time doubling the grid spacing
           negative: upsample grid |nlevel| times, each time halving the grid spacing
        Return:
         - new_grid: Grid object
        """
        if not self.regular:
            raise NotImplementedError("change_resolution only works for regular grid now")
        if nlevel == 0:
            return self
        else:
            new_grid = copy.deepcopy(self)
            fac = 2**nlevel
            new_grid.dx = self.dx * fac
            new_grid.dy = self.dy * fac
            new_grid.nx = int(np.round(self.Lx / new_grid.dx))
            new_grid.ny = int(np.round(self.Ly / new_grid.dy))
            assert min(new_grid.nx, new_grid.ny) > 1, "Grid.change_resolution_level: new resolution too low, try smaller nlevel"
            new_grid.x, new_grid.y = np.meshgrid(self.xmin + np.arange(new_grid.nx) * new_grid.dx, self.ymin + np.arange(new_grid.ny) * new_grid.dy)
            return new_grid

    def _mesh_dx(self):
        """
        computes averaged edge length for irregular mesh, used in self.dx, dy
        """
        t = self.tri.triangles
        s1 = np.sqrt((self.x[t][:,0]-self.x[t][:,1])**2+(self.y[t][:,0]-self.y[t][:,1])**2)
        s2 = np.sqrt((self.x[t][:,0]-self.x[t][:,2])**2+(self.y[t][:,0]-self.y[t][:,2])**2)
        s3 = np.sqrt((self.x[t][:,2]-self.x[t][:,1])**2+(self.y[t][:,2]-self.y[t][:,1])**2)
        sa = (s1 + s2 + s3)/3

        ##try to remove very elongated triangles, so that mesh dx is more accurate
        e = 0.3
        valid = np.logical_and(np.abs(s1-sa) < e*sa, np.abs(s2-sa) < e*sa, np.abs(s3-sa) < e*sa)
        if ~valid.all():
            ##all triangles are elongated, just take their mean size
            return np.mean(sa)
        else:
            ##take the mean of triangle size, excluding the elongated ones
            return np.mean(sa[valid])

    def _triangle_properties(self):
        t = self.tri.triangles
        x = self.x[self.tri.inds]
        y = self.y[self.tri.inds]
        s1 = np.hypot(x[t[:,0]] - x[t[:,1]], y[t[:,0]] - y[t[:,1]])
        s2 = np.hypot(x[t[:,0]] - x[t[:,2]], y[t[:,0]] - y[t[:,2]])
        s3 = np.hypot(x[t[:,2]] - x[t[:,1]], y[t[:,2]] - y[t[:,1]])
        s = 0.5*(s1+s2+s3)
        self.tri.p = 2.0 * s  ##circumference
        self.tri.a = np.sqrt(s*(s-s1)*(s-s2)*(s-s3))  ##area
        ##circumference-to-area ratio
        ##(1: equilateral triangle, ~0: very elongated)
        self.tri.ratio =  self.tri.a / s**2 * 3**(3/2)

    @cached_property
    def mfx(self):
        """
        Map scaling factors in x direction (mfx), since on the projection plane dx is not exactly
        the distance on Earth. The mfx is defined as ratio between dx and the actual distance.
        """
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
        """
        Map scaling factors in y direction (mfy), since on the projection plane dy is not exactly
        the distance on Earth. The mfy is defined as ratio between dy and the actual distance.
        """
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

    @property
    def dst_grid(self):
        """
        Destination grid for convert, interp, rotate_vector methods
        once specified a dst_grid, the setter will compute corresponding rotation_matrix and interp_weights
        """
        return self._dst_grid

    @dst_grid.setter
    def dst_grid(self, grid):
        assert isinstance(grid, Grid), "dst_grid should be a Grid instance"
        if grid == self.dst_grid:  ##the same grid is set before
            return

        self._dst_grid = grid

        ##rotation of vector field from self.proj to dst_grid.proj
        self._set_rotation_matrix()

        ##prepare indices and weights for interpolation
        ##when dst_grid is set, these info are prepared and stored to avoid recalculating
        ##too many times, when applying the same interp to a lot of flds
        x, y = self._proj_from(grid.x, grid.y)
        inside, indices, vertices, in_coords, nearest = self.find_index(x, y)
        self.interp_inside = inside
        self.interp_indices = indices
        self.interp_vertices = vertices
        self.interp_nearest = nearest
        self.interp_weights = self._interp_weights(inside, vertices, in_coords)

        ##prepare indices for coarse-graining
        x, y = self._proj_to(self.x, self.y)
        inside, _, _, _, nearest = self.dst_grid.find_index(x, y)
        self.coarsen_inside = inside
        self.coarsen_nearest = nearest
        if not self.regular: ## for irregular mesh, find indices for elements too
            x, y = self._proj_to(self.x_elem, self.y_elem)
            inside, _, _, _, nearest = self.dst_grid.find_index(x, y)
            self.coarsen_inside_elem = inside
            self.coarsen_nearest_elem = nearest

    def set_destination_grid(self, grid):
        """
        Set method for self.dst_grid the destination Grid object to convert to.
        """
        self.dst_grid = grid

    def _wrap_cyclic_xy(self, x_, y_):
        """
        When interpolating for point x_,y_, if the coordinates falls outside of the domain,
        we wrap around and make then inside again, if the boundary condition is cyclic (self.cyclic_dim)

        Inputs:
        - x_, y_: np.array
          x, y coordinates of a grid

        Returns:
        - x_, y_: np.array
          Same as input but x, y values are wrapped to be within the boundaries again.
        """
        if self.cyclic_dim is not None:
            for d in self.cyclic_dim:
                if d=='x':
                    x_ = np.mod(x_ - self.xmin, self.Lx) + self.xmin
                elif d=='y':
                    y_ = np.mod(y_ - self.ymin, self.Ly) + self.ymin
        return x_, y_

    def _pad_cyclic_mesh_bounds(self):
        ##repeat the mesh in x and y directions if cyclic, to form the wrap around geometry
        x = self.x
        y = self.y
        inds = np.arange(self.npoints)
        if 'x' in self.cyclic_dim:
            x = np.hstack([x, x-self.Lx, x+self.Lx])
            y = np.hstack([y, y, y])
            inds = np.hstack([inds, inds, inds])
        if 'y' in self.cyclic_dim:
            x = np.hstack([x, x, x])
            y = np.hstack([y, y-self.Ly, y+self.Ly])
            inds = np.hstack([inds, inds, inds])
        if 'x' in self.cyclic_dim and 'y' in self.cyclic_dim:
            x = np.hstack([x, x-self.Lx, x+self.Lx, x-self.Lx, x+self.Lx])
            y = np.hstack([y, y-self.Ly, y-self.Ly, y+self.Ly, y+self.Ly])
            inds = np.hstack([inds, inds, inds, inds, inds])
        tri = Triangulation(x, y)

        ##find the triangles that covers the domain and keep them
        triangles = []
        for i in range(tri.triangles.shape[0]):
            t = tri.triangles[i,:]
            if np.logical_and(x[t]>=self.xmin, x[t]<=self.xmax).any() and np.logical_and(y[t]>=self.ymin, y[t]<=self.ymax).any():
                triangles.append(t)
        triangles = np.array(triangles)

        ##collect the uniq grid point indices (in self.x and self.y) and assign them to the triangles
        uniq_inds = list(np.unique(triangles.reshape(-1)))
        for i in np.ndindex(triangles.shape):
            triangles[i] = uniq_inds.index(triangles[i])
        self.tri = Triangulation(x[uniq_inds], y[uniq_inds], triangles=triangles)
        self.tri.inds = inds[uniq_inds]

    def find_index(self, x_, y_):
        """
        Finding indices of self.x,y corresponding to the given x_,y_

        Inputs:
        - x_, y_: float or np.array
          x, y coordinates of target point(s)

        Returns:
        - inside: bool, np.array with x_.flatten().size
          Whether x_,y_ points are inside the grid.
          Note that the following returned properties only have the inside points

        - indices: int, np.array with inside_size
          Indices of the grid elements that x_,y_ falls in.
          For regular grid, it is None since vertices can pinpoint the grid box already.
          For irregular mesh, it is the index for tri.triangles from tri_finder.

        - vertices: int, np.array with shape (inside_size, n),
          n = 4 for regular grid boxes, or 3 for mesh elements
          The grid indices in self.x,y (flattened) for the nodes of each
          grid element that x_,y_ falls in.

        - in_coords: float, np.array with shape (inside_size, n),
          n = 2 for regular grid boxes, or 3 for mesh elements
          The internal coordinates for x_,y_ within the grid box/element,
          used in computing interp_weights, see illustration below.

        - nearest: int, np.array with inside_size
          The indices for the nodes in self.x,y that are closest to x_,y_
        """
        x_ = np.array(x_).flatten()
        y_ = np.array(y_).flatten()

        ##lon: pyproj.Proj works only for lon=-180:180
        if self.proj_name == 'longlat':
            x_ = np.mod(x_ + 180., 360.) - 180.

        ###account for cyclic dim, when points drop "outside" then wrap around
        x_, y_ = self._wrap_cyclic_xy(x_, y_)

        if self.regular:
            xi = self.x[0, :]
            yi = self.y[:, 0]

            ##sort the index to monoticially increasing
            ##x_,y_ are the sorted coordinates of grid points
            ##i_,j_ are their original grid index
            i_ = np.argsort(xi)
            xi_ = xi[i_]
            j_ = np.argsort(yi)
            yi_ = yi[j_]

            ##pad cyclic dimensions with additional grid point for the wrap-around
            if self.cyclic_dim is not None:
                for d in self.cyclic_dim:
                    if d=='x':
                        if xi_[0]+self.Lx not in xi_:
                            xi_ = np.hstack((xi_, xi_[0] + self.Lx))
                            i_ = np.hstack((i_, i_[0]))
                    elif d=='y':
                        if yi_[0]+self.Ly not in yi_:
                            yi_ = np.hstack((yi_, yi_[0] + self.Ly))
                            j_ = np.hstack((j_, j_[0]))

            ##if neighbors indices are provided, the search range is extended by 1 grid on both sides
            if self.neighbors is not None:
                xi_ = np.hstack((xi_[0]-self.dx, xi_, xi_[-1]+self.dx))
                i_ = np.hstack((i_[0]-1, i_, i_[-1]+1))
                yi_ = np.hstack((yi_[0]-self.dy, yi_, yi_[-1]+self.dy))
                j_ = np.hstack((j_[0]-1, j_, j_[-1]+1))

            ##now find the position near the given x_,y_ coordinates
            ##pi,pj are the index in the padded array, right side of the given x_,y_
            ##only the positions inside the grid will be kept
            pi = np.array(np.searchsorted(xi_, x_, side='right'))
            pj = np.array(np.searchsorted(yi_, y_, side='right'))
            inside = ~np.logical_or(np.logical_or(pi==len(xi_), pi==0),
                                    np.logical_or(pj==len(yi_), pj==0))
            pi, pj = pi[inside], pj[inside]

            ##vertices (p1, p2, p3, p4) for the rectangular grid box
            ##p3 is the point found by the search index (pj,pi),
            ##internal coordinates (in_x, in_y) pinpoint the x_,y_ location inside
            ##the rectangle with values range [0, 1)
            ##(pj,pi-1)   p4----+------p3 (pj,pi)
            ##             |    |      |
            ##             +in_x*------+
            ##             |    in_y   |
            ##(pj-1,pi-1) p1----+------p2 (pj-1,pi)
            indices = None #for regular grid, the element indices are not used

            if self.neighbors is not None:
                ##find the right indices for each vertex grid point
                j1,i1 = j_[pj-1], i_[pi-1]
                j2,i2 = np.zeros(pj.shape, dtype=int), np.zeros(pj.shape, dtype=int)
                j3,i3 = np.zeros(pj.shape, dtype=int), np.zeros(pj.shape, dtype=int)
                j4,i4 = np.zeros(pj.shape, dtype=int), np.zeros(pj.shape, dtype=int)

                ind = np.where(np.logical_and(j1>=0, i1>=0)) ##p1 is the anchor in neighbors
                j2[ind], i2[ind] = self.neighbors[0,0,j1[ind],i1[ind]], self.neighbors[1,0,j1[ind],i1[ind]]
                j3[ind], i3[ind] = self.neighbors[0,1,j2[ind],i2[ind]], self.neighbors[1,1,j2[ind],i2[ind]]
                j4[ind], i4[ind] = self.neighbors[0,1,j1[ind],i1[ind]], self.neighbors[1,1,j1[ind],i1[ind]]

                ind = np.where(np.logical_and(j1>=0, i1<0)) ##p2 is the anchor in neighbors
                j2[ind], i2[ind] = j_[pj-1][ind], i_[pi][ind]
                j1[ind], i1[ind] = self.neighbors[0,2,j2[ind],i2[ind]], self.neighbors[1,2,j2[ind],i2[ind]]
                j3[ind], i3[ind] = self.neighbors[0,1,j2[ind],i2[ind]], self.neighbors[1,1,j2[ind],i2[ind]]
                j4[ind], i4[ind] = self.neighbors[0,2,j3[ind],i3[ind]], self.neighbors[1,2,j3[ind],i3[ind]]

                ind = np.where(np.logical_and(j1<0, i1<0)) ##p3 is the anchor in neighbors
                j3[ind], i3[ind] = j_[pj][ind], i_[pi][ind]
                j2[ind], i2[ind] = self.neighbors[0,3,j3[ind],i3[ind]], self.neighbors[1,3,j3[ind],i3[ind]]
                j4[ind], i4[ind] = self.neighbors[0,2,j3[ind],i3[ind]], self.neighbors[1,2,j3[ind],i3[ind]]
                j1[ind], i1[ind] = self.neighbors[0,2,j2[ind],i2[ind]], self.neighbors[1,2,j2[ind],i2[ind]]

                ind = np.where(np.logical_and(j1<0, i1>=0)) ##p4 is the anchor in neighbors
                j4[ind], i4[ind] = j_[pj][ind], i_[pi-1][ind]
                j1[ind], i1[ind] = self.neighbors[0,3,j4[ind],i4[ind]], self.neighbors[1,3,j4[ind],i4[ind]]
                j3[ind], i3[ind] = self.neighbors[0,0,j4[ind],i4[ind]], self.neighbors[1,0,j4[ind],i4[ind]]
                j2[ind], i2[ind] = self.neighbors[0,0,j1[ind],i1[ind]], self.neighbors[1,0,j1[ind],i1[ind]]

            else:
                ##use normal rectangle grid indices
                j1, i1 = j_[pj-1], i_[pi-1]
                j2, i2 = j_[pj-1], i_[pi]
                j3, i3 = j_[pj],   i_[pi]
                j4, i4 = j_[pj],   i_[pi-1]

            ##assign the points to vertices
            vertices = np.zeros(pi.shape+(4,), dtype=int)
            vertices[:, 0] = j1 * self.nx + i1
            vertices[:, 1] = j2 * self.nx + i2
            vertices[:, 2] = j3 * self.nx + i3
            vertices[:, 3] = j4 * self.nx + i4

            ##internal coordinates inside rectangles
            in_coords = np.zeros(pi.shape+(2,), dtype=np.float64)
            in_coords[:, 0] = (x_[inside] - xi_[pi-1]) / (xi_[pi] - xi_[pi-1])
            in_coords[:, 1] = (y_[inside] - yi_[pj-1]) / (yi_[pj] - yi_[pj-1])

            ##index of grid point nearest to (x_,y_)
            j_near = np.zeros(pj.shape, dtype=int)
            i_near = np.zeros(pj.shape, dtype=int)
            ind = np.where(np.logical_and(in_coords[:,0]<0.5, in_coords[:,1]<0.5))
            j_near[ind], i_near[ind] = j1[ind], i1[ind]
            ind = np.where(np.logical_and(in_coords[:,0]>=0.5, in_coords[:,1]<0.5))
            j_near[ind], i_near[ind] = j2[ind], i2[ind]
            ind = np.where(np.logical_and(in_coords[:,0]>=0.5, in_coords[:,1]>=0.5))
            j_near[ind], i_near[ind] = j3[ind], i3[ind]
            ind = np.where(np.logical_and(in_coords[:,0]<0.5, in_coords[:,1]>=0.5))
            j_near[ind], i_near[ind] = j4[ind], i4[ind]
            nearest = j_near * self.nx + i_near

        else:
            ##for irregular mesh, use tri_finder to find index
            tri_finder = self.tri.get_trifinder()
            triangle_map = tri_finder(x_, y_)
            inside = ~(triangle_map < 0)
            indices = triangle_map[inside]

            ##internal coords are the barycentric coords (in1, in2, in3) in a triangle
            ##note: larger in1 means closer to the vertice 1!
            ##     (0,0,1) p3\
            ##            / | \
            ##           / in3. \
            ##          /  :* .   \
            ##         /in1  | in2  \
            ##(1,0,0) p1-------------p2 (0,1,0)
            vertices = self.tri.triangles[triangle_map[inside], :]

            ##transform matrix for barycentric coords computation
            a = self.tri.x[vertices[:,0]] - self.tri.x[vertices[:,2]]
            b = self.tri.x[vertices[:,1]] - self.tri.x[vertices[:,2]]
            c = self.tri.y[vertices[:,0]] - self.tri.y[vertices[:,2]]
            d = self.tri.y[vertices[:,1]] - self.tri.y[vertices[:,2]]
            det = a*d-b*c
            t_matrix = np.zeros((len(vertices), 3, 2))
            t_matrix[:,0,0] = d/det
            t_matrix[:,0,1] = -b/det
            t_matrix[:,1,0] = -c/det
            t_matrix[:,1,1] = a/det
            t_matrix[:,2,0] = self.tri.x[vertices[:,2]]
            t_matrix[:,2,1] = self.tri.y[vertices[:,2]]

            ##get barycentric coords, according to https://en.wikipedia.org/wiki/
            ##Barycentric_coordinate_system#Barycentric_coordinates_on_triangles,
            delta = np.array([x_[inside], y_[inside]]).T - t_matrix[:,2,:]
            in12 = np.einsum('njk,nk->nj', t_matrix[:,:2,:], delta)
            in_coords = np.hstack((in12, 1.-in12.sum(axis=1, keepdims=True)))

            ##index of grid nearest to (x_,y_)
            nearest = vertices[np.arange(len(in_coords), dtype=int), np.argmax(in_coords, axis=1)]

        return inside, indices, vertices, in_coords, nearest

    def _proj_to(self, x, y):
        """
        Transform coordinates from self.proj to dst_grid.proj
        """
        if self.dst_grid.proj != self.proj:
            lon, lat = self.proj(x, y, inverse=True)
            x, y = self.dst_grid.proj(lon, lat)
        x, y = self.dst_grid._wrap_cyclic_xy(x, y)
        return x, y

    def _proj_from(self, x, y):
        """
        transform coordinates from dst_grid.proj to self.proj
        """
        if self.dst_grid.proj != self.proj:
            lon, lat = self.dst_grid.proj(x, y, inverse=True)
            x, y = self.proj(lon, lat)
        x, y = self._wrap_cyclic_xy(x, y)
        return x, y

    def _set_rotation_matrix(self):
        """
        setting the rotation matrix for converting vector fields from self to dst_grid
        """
        self.rotate_matrix = np.zeros((4,)+self.x.shape)
        if self.proj != self.dst_grid.proj:
            ##self.x,y corresponding coordinates in dst_proj, call them x,y
            x, y = self._proj_to(self.x, self.y)

            ##find small increments in x,y due to small changes in self.x,y in dst_proj
            eps = 0.1 * self.dx    ##grid spacing is specified in Grid object
            xu, yu = self._proj_to(self.x + eps, self.y      )  ##move a bit in x dirn
            xv, yv = self._proj_to(self.x      , self.y + eps)  ##move a bit in y dirn

            np.seterr(invalid='ignore')  ##will get nan at poles due to singularity, fill_pole_void takes care later
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
        else:
            ##if no change in proj, we can skip the calculation
            self.rotate_matrix[0, :] = 1.
            self.rotate_matrix[1, :] = 0.
            self.rotate_matrix[2, :] = 0.
            self.rotate_matrix[3, :] = 1.

    def _fill_pole_void(self, fld):
        """
        if rotation of vectors (or other reasons) generates nan at the poles
        we fill in the void using surrounding values for each pole defined by self.pole_dim and pole_index
        """
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
        """
        Apply the rotate_matrix to a vector field

        Inputs:
        - vec_fld: float, np.array with shape (2, self.x.shape)
          The input vector field defined on self.

        Returns:
        - vec_fld: float, same shape as the input vec_fld
          The vector field rotated to the dst_grid, still defined on self.
        """
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

    def get_corners(self, fld):
        """
        given fld defined on a regular grid, obtain its value on the 4 corners/vertices
        """
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

    def _interp_weights(self, inside, vertices, in_coords):
        """
        Compute interpolation weights from the outputs of find_index
        the interp_weights are the weights (sums to 1) given to each grid vertex in self.x,y
        based on their distance to the x_,y_ points (as specified by the in_coords)

        Inputs:
        - inside, vertices, in_coords: from the output of self.find_index

        Output:
        - interp_weights: float, np.array with vertices.shape
        """
        if self.regular:
            ##compute bilinear interp weights
            interp_weights = np.zeros(vertices.shape)
            interp_weights[:, 0] =  (1-in_coords[:, 0]) * (1-in_coords[:, 1])
            interp_weights[:, 1] =  in_coords[:, 0] * (1-in_coords[:, 1])
            interp_weights[:, 2] =  in_coords[:, 0] * in_coords[:, 1]
            interp_weights[:, 3] =  (1-in_coords[:, 0]) * in_coords[:, 1]
        else:
            ##use barycentric coordinates as interp weights
            interp_weights = in_coords
        return interp_weights

    def interp(self, fld, x=None, y=None, method='linear'):
        """
        Interpolation of 2D field data (fld) from one grid (self or given x,y) to another (dst_grid).
        This can be used for grid refining (low->high resolution) or grid thinning (high->low resolution).
        This also converts between different grid geometries.

        Inputs:
        - fld: np.array
          Input field defined on self, should have same shape as self.x,y

        - x,y: float or np.array, optional
          If x,y are specified, the function computes the weights and apply them to fld
          If x,y are None, the self.dst_grid.x,y are used.
          Since their interp_weights are precalculated by dst_grid.setter it will be efficient
          to run interp for many different input flds quickly.

        - method: str
          Interpolation method, can be 'nearest' or 'linear'

        Returns:
        - fld_interp: float, np.array
          The interpolated field defined on the destination grid
        """
        if x is None or y is None:
            ##use precalculated weights for self.dst_grid
            inside = self.interp_inside
            indices = self.interp_indices
            vertices = self.interp_vertices
            nearest = self.interp_nearest
            weights = self.interp_weights
            x = self.dst_grid.x
        else:
            ##otherwise compute the weights for the given x,y
            inside, indices, vertices, in_coords, nearest = self.find_index(x, y)
            weights = self._interp_weights(inside, vertices, in_coords)

        fld_interp = np.full(np.array(x).flatten().shape, np.nan)
        if fld.shape == self.x.shape:
            if not self.regular:
                fld = fld[self.tri.inds]
            if method == 'nearest':
                # find the node of the triangle with the maximum weight
                fld_interp[inside] = fld.flatten()[nearest]
            elif method == 'linear':
                # sum over the weights for each node of triangle
                fld_interp[inside] = np.einsum('nj,nj->n', np.take(fld.flatten(), vertices), weights)
            else:
                raise NotImplementedError(f"interp method {method} is not yet available")
        elif not self.regular and fld.shape == self.x_elem.shape:
            fld_interp[inside] = fld[indices]
        else:
            raise ValueError("field shape does not match grid shape, or number of triangle elements")
        return fld_interp.reshape(np.array(x).shape)

    ###utility function for coarse-graining (high->low resolution)
    def coarsen(self, fld):
        """
        Coarse-graining is sometimes needed when the dst_grid is at lower resolution than self.
        Since many points of self.x,y falls in one dst_grid box/element, it is better to
        average them to represent the field on the low-res grid, instead of interpolating
        only from the nearest points that will cause representation errors.

        Inputs:
        - fld: float, np.array
          Input field to perform coarse-graining on, it is defined on self.

        Outputs:
        - fld_coarse: float, np.array
          The coarse-grained field defined on self.dst_grid.
        """

        ##find which location x_,y_ falls in in dst_grid
        if fld.shape == self.x.shape:
            inside = self.coarsen_inside
            nearest = self.coarsen_nearest
        elif not self.regular and fld.shape == self.x_elem.shape:
            inside = self.coarsen_inside_elem
            nearest = self.coarsen_nearest_elem
        else:
            raise ValueError("field shape does not match grid shape, or number of triangle elements")

        fld_coarse = np.zeros(self.dst_grid.x.flatten().shape)
        count = np.zeros(self.dst_grid.x.flatten().shape)
        fld_inside = fld.flatten()[inside]
        valid = ~np.isnan(fld_inside)  ##filter out nan

        ##average the fld points inside each dst_grid box/element
        np.add.at(fld_coarse, nearest[valid], fld_inside[valid])
        np.add.at(count, nearest[valid], 1)

        valid = (count>1)  ##do not coarse grain if only one point near by
        fld_coarse[valid] /= count[valid]
        fld_coarse[~valid] = np.nan

        return fld_coarse.reshape(self.dst_grid.x.shape)

    def convert(self, fld, is_vector=False, method='linear', coarse_grain=False):
        """
        Main method to convert from self.proj, x, y to dst_grid coordinate systems:
        Steps: 1. if projection changes and is_vector, rotate vectors from self.proj to dst_grid.proj
               2.1 interpolate fld components from self.x,y to dst_grid.x,y
               2.2 if dst_grid is low-res, coarse_grain=True will perform coarse-graining

        Inputs:
        - fld: float, np.array
          Input field to perform convertion on.

        - is_vector: bool, optional
          If False (default) the input fld is a scalar field,
          otherwise the input fld is a vector field.

        - method: str, optional
          Interpolation method, 'linear' (default) or 'nearest'

        - coarse_grain: bool, optional
          If True, the coarse-graining will be applied using self.coarsen(). The default is False.

        Outputs:
        - fld_out: float, np.array
          The converted field defined on the destination grid self.dst_grid.
        """
        if self.dst_grid != self:
            if is_vector:
                assert fld.shape[0] == 2, "vector field should have first dim==2, for u,v component"
                ##vector field needs to rotate to dst_grid.proj before interp
                fld = self.rotate_vectors(fld)

                fld_out = np.full((2,)+self.dst_grid.x.shape, np.nan)
                for i in range(2):
                    ##interp each component: u, v
                    fld_out[i, :] = self.interp(fld[i, :], method=method)
                    if coarse_grain:
                        ##coarse-graining if more points fall in one grid
                        fld_coarse = self.coarsen(fld[i, :])
                        ind = ~np.isnan(fld_coarse)
                        fld_out[i, ind] = fld_coarse[ind]
            else:
                ##scalar field, just interpolate
                fld_out = np.full(self.dst_grid.x.shape, np.nan)
                fld_out = self.interp(fld, method=method)
                if coarse_grain:
                    ##coarse-graining if more points fall in one grid
                    fld_coarse = self.coarsen(fld)
                    ind = ~np.isnan(fld_coarse)
                    fld_out[ind] = fld_coarse[ind]
        else:
            fld_out = fld
        return fld_out

    def distance_in_x(self, ref_x, x):
        dist_x = np.abs(x - ref_x)
        if self.cyclic_dim is not None:
            if 'x' in self.cyclic_dim:
                dist_x = np.minimum(dist_x, self.Lx - dist_x)
        return dist_x

    def distance_in_y(self, ref_y, y):
        dist_y = np.abs(y - ref_y)
        if self.cyclic_dim is not None:
            if 'y' in self.cyclic_dim:
                dist_y = np.minimum(dist_y, self.Ly - dist_y)
        return dist_y

    def distance(self, ref_x, x, ref_y, y, p=2):
        """
        Compute distance for points (x,y) to the reference point
        Input:
        - ref_x, ref_y: float
          reference point coordinates
        - x, y: np.array(float)
          points whose distance to the reference points will be computed
        - p: int
          Minkowski p-norm order, default is 2
        Output:
        - dist: np.array(float)
        """
        ##TODO: account for other geometry (neighbors) here

        ##normal cartesian distances in x and y
        dist_x = self.distance_in_x(ref_x, x)
        dist_y = self.distance_in_y(ref_y, y)
        if p == 1:
            dist = dist_x + dist_y  ##Manhattan distance, order 1
        elif p == 2:
            dist = np.hypot(dist_x, dist_y)   ##Euclidean distance, order 2
        else:
            raise NotImplementedError(f"grid.distance: p-norm order {p} is not implemented for 2D grid")
        return dist

    ### Some methods for basic data visulisation and map plotting
    def _collect_shape_data(self, shapes):
        """
        This collects the x,y coordinates from shapes read from .shp files for later plotting
        filter the points not inside the grid domain
        """
        data = {'xy':[], 'parts':[]}
        for shape in shapes:
            if len(shape.points) > 0:
                xy = []
                inside = []
                lon, lat = [np.array(x) for x in zip(*shape.points)]
                x, y = self.proj(lon, lat)
                inside = np.logical_and(np.logical_and(x >= self.xmin, x <= self.xmax),
                                        np.logical_and(y >= self.ymin, y <= self.ymax))

                ##when showing global maps, the lines leave the domain and re-enter
                ##from the other side, the cross-over lines are visible on the plot
                ##temporary solution: make a pause when lines wrap around cut meridian
                ##lines work fine now but filled patches do not
                if self.proj_name in ['longlat', 'tripolar', 'bipolar']:
                    x[~inside] = np.nan
                    y[~inside] = np.nan

                xy = [(x[i], y[i]) for i in range(x.size)]

                ##if any point in the polygon lies inside the grid, need to plot it.
                if any(inside):
                    data['xy'].append(xy)
                    data['parts'].append(shape.parts)

        return data

    @cached_property
    def land_data(self):
        """
        prepare data to show the land area, the shp file ne_50m_coastlines is
        downloaded from https://www.naturalearthdata.com
        """
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

        return self._collect_shape_data(shapes)

    @cached_property
    def river_data(self):
        """
        prepare data to show river features
        """
        path = os.path.split(inspect.getfile(self.__class__))[0]
        sf = shapefile.Reader(os.path.join(path, 'ne_50m_rivers.shp'))
        shapes = sf.shapes()
        return self._collect_shape_data(shapes)

    @cached_property
    def lake_data(self):
        """
        prepare data to show lake features
        """
        path = os.path.split(inspect.getfile(self.__class__))[0]
        sf = shapefile.Reader(os.path.join(path, 'ne_50m_lakes.shp'))
        shapes = sf.shapes()
        return self._collect_shape_data(shapes)

    def llgrid_xy(self, dlon, dlat):
        """
        Prepare a lon/lat grid to plot as reference lines

        Inputs:
        - dlon, dlat: spacing of lon/lat grid in degrees
        """
        self.dlon = dlon
        self.dlat = dlat

        llgrid_xy = []
        for lon_r in np.arange(-180, 180, dlon):
            xy = []
            inside = []
            lat = np.arange(-89.9, 90, 0.1)
            lon = np.ones(lat.size) * lon_r
            x, y = self.proj(lon, lat)
            inside = np.logical_and(np.logical_and(x >= self.xmin, x <= self.xmax),
                                    np.logical_and(y >= self.ymin, y <= self.ymax))
            x[~inside] = np.nan
            y[~inside] = np.nan
            xy = [(x[i], y[i]) for i in range(x.size)]
            if any(inside):
                llgrid_xy.append(xy)

        for lat_r in np.arange(-90, 90+dlat, dlat):
            xy = []
            inside = []
            lon = np.arange(-180., 180., 0.1)
            lat = np.ones(lon.size) * lat_r
            x, y = self.proj(lon, lat)
            inside = np.logical_and(np.logical_and(x >= self.xmin, x <= self.xmax),
                                    np.logical_and(y >= self.ymin, y <= self.ymax))
            x[~inside] = np.nan
            y[~inside] = np.nan
            xy = [(x[i], y[i]) for i in range(x.size)]
            if any(inside):
                llgrid_xy.append(xy)

        return llgrid_xy

    def plot_field(self, ax, fld,  vmin=None, vmax=None, cmap='viridis', **kwargs):
        """
        Plot a scalar field using pcolor/tripcolor

        Inputs:
        - ax: matplotlib.pyplot.Axes
          Handle for plotting

        - fld: float, np.array
          The scalar field for plotting

        - vmin, vmax: float, optional
          The minimum and maximum value range for the colormap, if not specified (None)
          the np.min, np.max of the input fld will be used.

        - cmap: matplotlib colormap, optional
          Colormap used in the plot, default is 'viridis'
        """
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
            im = ax.pcolor(x, y, fld, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

        else:
            if fld.shape == self.x.shape:
                fld = fld[self.tri.inds]
            im = ax.tripcolor(self.tri, fld, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

        self.set_xylim(ax)
        return im

    def plot_scatter(self, ax, fld, vmin=None, vmax=None, nlevel=20, cmap='viridis', markersize=10, **kwargs):
        """
        Same as plot_field, but showing individual scattered points instead
        This is more suitable for plotting observations in space
        """
        if vmin is None:
            vmin = np.nanmin(fld)
        if vmax is None:
            vmax = np.nanmax(fld)
        dv = (vmax - vmin) / nlevel

        v = np.array(fld)
        vbound = np.maximum(np.minimum(v, vmax), vmin)

        cmap = np.array([colormaps[cmap](x)[0:3] for x in np.linspace(0, 1, nlevel+1)])
        cind = ((vbound - vmin) / dv).astype(int)

        ax.scatter(self.x, self.y, markersize, color=cmap[cind], **kwargs)

        self.set_xylim(ax)

    def plot_vectors(self, ax, vec_fld, V=None, L=None, spacing=0.5, num_steps=10,
                     linecolor='k', linewidth=1,
                     showref=False, ref_xy=(0, 0), refcolor='w',
                     showhead=True, headwidth=0.1, headlength=0.3):
        """
        Plot vector fields (improved version of matplotlib quiver)

        Inputs:
        - ax: matplotlib.pyplot.Axes
          Handle for plotting

        - vec_fld: float, np.array
          The vector field for plotting

        - V: float, optional
          Velocity scale, typical velocity value in vec_fld units. If not specified (None)
          a typical value 0.33*max(abs(vec_fld[0,:])) will be used.

        - L: float, optional
          Length scale, how long in x,y units do vectors with velocity V show in the plot
          If not specified (None), a typical value 0.05*self.Lx will be used.

        - spacing: float, optional
          Distance between vectors in both directions is given by spacing*L. Default is 0.5.
          This controls the density of vectors in the plot.
          You can provide a tuple (float, float) for spacings in (x, y) if you want them
          to be set differently.

        - num_steps: int, optional
          Default is 10. If num_steps=1, straight vectors (as in quiver) will be displayed.
          num_steps>1 lets you display curved trajectories, at each sub-step the velocity is
          re-interpolated at the new position along the trajectories. As num_steps get larger
          the trajectories are more detailed.

        - linecolor: matplotlib color, optional
          Line color for the vector lines, default is 'k'

        - linewidth: float, optional
          Line width for the vector lines, default is 1.

        - showref: bool, optional
          If True, show a legend box with a reference vector (size L) inside. Default is False.

        - ref_xy: tuple (x,y) float, optional
          The x,y coordinates for the reference vector box

        - ref_color: matplotlib color, optional
          Background color for the reference vector box, default is 'w' (white).

        - showhead: bool, optional
          If True (default), show the arrow head of the vectors

        - headwidth: float, optional
          Width of arrow heads relative to L, default is 0.1.

        - headlength: float, optional
          Length of arrow heads relative to L, default is 0.3.
        """
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

        xo, yo = np.mgrid[x.min()+0.5*d[0]:x.max():d[0], y.min()+0.5*d[1]:y.max():d[1]]
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
            ut = ut * self.interp(self.mfx, xtraj[:,t], ytraj[:,t])
            vt = vt * self.interp(self.mfy, xtraj[:,t], ytraj[:,t])

            ###update traj position
            xtraj[:, t+1] = xtraj[:, t] + ut * dt
            ytraj[:, t+1] = ytraj[:, t] + vt * dt

            ##update length
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

        for i in range(xtraj.shape[0]):
            ##plot trajectory at one output location
            ax.plot(xtraj[i, :], ytraj[i, :], color=linecolor, linewidth=linewidth, zorder=4)

            ##add vector head if traj is long and straight enough
            dist = np.sqrt((xtraj[i,0]-xtraj[i,-1])**2 + (ytraj[i,0]-ytraj[i,-1])**2)
            if showhead and hl < leng[i] < 1.6*dist:
                ax.fill(*arrowhead_xy(xtraj[i,-1], xtraj[i,-2], ytraj[i,-1],ytraj[i,-2]), color=linecolor, zorder=5)

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

        self.set_xylim(ax)

    def plot_land(self, ax, color=None, linecolor='k', linewidth=1,
                  showriver=False, rivercolor='c',
                  showgrid=True, dlon=20, dlat=5):
        """
        Shows the map (coastline, rivers, lakes) and lon/lat grid for reference

        Parameters:
        - ax: matplotlib.pyplot.Axes object
          Handle for plotting

        - color: matplotlib color, optional
          Face color of the landmass polygon, default is None (transparent).

        - linecolor: matplotlib color, optional
          Line color of the coastline, default is 'k' (black).

        - linewidth: float, optional
          Line width of the coastline, default is 1.

        - showriver: bool, optional
          If True, show the rivers and lakes over the landmass. Default is False.

        - rivercolor: matplotlib color, optional
          Color of the rivers and lakes, default is 'c' (cyan).

        - showgrid: bool, optional
          If True (default), show the reference lat/lon grid.

        - dlon, dlat: float, optional
          The interval of lon,lat lines in the reference grid. Default is 20,5 degrees.
        """

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

        self.set_xylim(ax)

    def set_xylim(self, ax):
        ##set the correct extent of plot
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

