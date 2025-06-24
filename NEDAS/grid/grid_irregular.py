import numpy as np
import copy
from matplotlib.tri import Triangulation
from matplotlib import colormaps
from NEDAS.grid.grid_base import GridBase

class IrregularGrid(GridBase):
    """
    Class for handling irregular grids.

    Args:
        triangles (np.ndarray, optional):
            `None` (default), if the triangle indices `ind[grid.x.size, 3]` for the unstructured mesh are given here, the grid will skip computing triangulation.

    """
    def __init__(self, proj, x, y, bounds=None, cyclic_dim=None, distance_type='cartesian',
                 triangles=None, dst_grid=None):
        super().__init__(proj, x, y, bounds, cyclic_dim, distance_type, dst_grid)
        self.regular = False

        ##Generate triangulation, if tiangles are provided its very quick,
        ##otherwise Triangulation will generate one, but slower.
        self.x = self.x.flatten()
        self.y = self.y.flatten()
        self.npoints = self.x.size
        self.tri = Triangulation(self.x, self.y, triangles=triangles)
        self.tri.inds = np.arange(self.npoints)
        self._triangle_properties()
        dx = self._mesh_dx()
        self.dx = dx
        self.dy = dx
        self.x_elem = np.mean(self.tri.x[self.tri.triangles], axis=1)
        self.y_elem = np.mean(self.tri.y[self.tri.triangles], axis=1)
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        if self.cyclic_dim is not None:
            self._pad_cyclic_mesh_bounds()

        self.mask = np.full(self.x.shape, False)

        self.distance_type = distance_type

        self._dst_grid = None
        if dst_grid is not None:
            self.set_destination_grid(dst_grid)

    def change_resolution_level(self, nlevel):
        raise NotImplementedError("change_resolution only works for regular grid now")

    def _mesh_dx(self):
        """
        computes averaged edge length for irregular mesh, used in self.dx, dy
        """
        t = self.tri.triangles
        s1 = np.hypot(self.x[t][:,0]-self.x[t][:,1], self.y[t][:,0]-self.y[t][:,1])
        s2 = np.hypot(self.x[t][:,0]-self.x[t][:,2], self.y[t][:,0]-self.y[t][:,2])
        s3 = np.hypot(self.x[t][:,2]-self.x[t][:,1], self.y[t][:,2]-self.y[t][:,1])
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
        """
        computes triangle properties for the mesh triangles
        """
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
        x_ = np.array(x_).flatten()
        y_ = np.array(y_).flatten()

        ##lon: pyproj.Proj works only for lon=-180:180
        if self.proj_name == 'longlat':
            x_ = np.mod(x_ + 180., 360.) - 180.

        ###account for cyclic dim, when points drop "outside" then wrap around
        x_, y_ = self._wrap_cyclic_xy(x_, y_)

        ##for irregular mesh, use tri_finder to find index
        tri_finder = self.tri.get_trifinder()
        triangle_map = tri_finder(x_, y_)
        inside = (triangle_map >= 0)
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
        ##use barycentric coordinates as interp weights
        interp_weights = in_coords
        return interp_weights

    def interp(self, fld, x=None, y=None, method='linear'):
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
            fld = fld[self.tri.inds]
            if method == 'nearest':
                # find the node of the triangle with the maximum weight
                fld_interp[inside] = fld.flatten()[nearest]
            elif method == 'linear':
                # sum over the weights for each node of triangle
                fld_interp[inside] = np.einsum('nj,nj->n', np.take(fld.flatten(), vertices), weights)
            else:
                raise NotImplementedError(f"interp method {method} is not yet available")
        elif fld.shape == self.x_elem.shape:
            fld_interp[inside] = fld[indices]
        else:
            raise ValueError("field shape does not match grid shape, or number of triangle elements")
        return fld_interp.reshape(np.array(x).shape)

    def coarsen(self, fld):
        ##find which location x_,y_ falls in in dst_grid
        if fld.shape == self.x.shape:
            inside = self.coarsen_inside
            nearest = self.coarsen_nearest
        elif fld.shape == self.x_elem.shape:
            inside = self.coarsen_inside_elem
            nearest = self.coarsen_nearest_elem
        else:
            raise ValueError("field shape does not match grid shape, or number of triangle elements")

        fld_coarse = np.zeros(self.dst_grid.x.flatten().shape)
        count = np.zeros(self.dst_grid.x.flatten().shape)
        fld_inside = fld.flatten()[inside]
        valid = ~np.isnan(fld_inside)  ##filter out nan

        ##average the fld points inside each dst_grid element
        np.add.at(fld_coarse, nearest[valid], fld_inside[valid])
        np.add.at(count, nearest[valid], 1)

        valid = (count>1)  ##do not coarse grain if only one point near by
        fld_coarse[valid] /= count[valid]
        fld_coarse[~valid] = np.nan

        return fld_coarse.reshape(self.dst_grid.x.shape)

    def plot_field(self, ax, fld,  vmin=None, vmax=None, cmap='viridis', **kwargs):
        if vmin is None:
            vmin = np.nanmin(fld)
        if vmax is None:
            vmax = np.nanmax(fld)

        if isinstance(cmap, str):
            cmap = colormaps[cmap]

        if fld.shape == self.x.shape:
            fld = fld[self.tri.inds]
        im = ax.tripcolor(self.tri, fld, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

        self.set_xylim(ax)
        return im
