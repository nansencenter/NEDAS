import numpy as np
import copy
from functools import cached_property

class Grid1D(object):
    """
    Grid class to handle fields defined on a 1D grid
    """
    def __init__(self, x, bounds=None, regular=True, cyclic=False, dst_grid=None):

        ##coordinates and properties of the 2D grid
        self.x = x
        self.regular = regular
        self.cyclic = cyclic
        if bounds is not None:
            self.xmin, self.xmax = bounds
        else:
            self.xmin = np.min(self.x)
            self.xmax = np.max(self.x)

        self.nx = self.x.size
        if regular:
            self.dx = (self.xmax - self.xmin) / (self.nx - 1)
            self.Lx = self.nx * self.dx
        else:
            self.dx = np.mean(np.diff(np.sort(self.x)))
            self.Lx = self.xmax - self.xmin

        self._dst_grid = None
        if dst_grid is not None:
            self.set_destination_grid(dst_grid)

    def __eq__(self, other):
        if not isinstance(other, Grid1D):
            return False
        if self.x.size != other.x.size:
            return False
        if not np.allclose(self.x, other.x):
            return False
        return True

    @classmethod
    def regular_grid(cls, xstart, xend, dx, centered=False, **kwargs):
        """
        Create a regular grid within specified boundaries.
        """
        self = cls.__new__(cls)
        x = np.arange(xstart, xend, dx)
        if centered:
            x += 0.5*dx  ##move coords to center of grid box
        self.__init__(x, regular=True, **kwargs)
        return self

    @classmethod
    def random_grid(cls, xstart, xend, npoints, **kwargs):
        """
        Create a grid with randomly positioned points within specified boundaries.
        """
        self = cls.__new__(cls)
        x = np.random.uniform(0, 1, npoints) * (xend - xstart) + xstart
        self.__init__(x, bounds=[xstart, xend], regular=False, **kwargs)
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
            new_grid.nx = int(np.round(self.Lx / new_grid.dx))
            assert new_grid.nx > 1, "Grid.change_resolution_level: new resolution too low, try smaller nlevel"
            new_grid.x = self.xmin + np.arange(new_grid.nx) * new_grid.dx
            return new_grid

    @property
    def dst_grid(self):
        """
        Destination grid for convert, interp, rotate_vector methods
        once specified a dst_grid, the setter will compute corresponding rotation_matrix and interp_weights
        """
        return self._dst_grid

    @dst_grid.setter
    def dst_grid(self, grid):
        assert isinstance(grid, Grid1D), "dst_grid should be a Grid1D instance"
        if grid == self.dst_grid:  ##the same grid is set before
            return
        self._dst_grid = grid

        ##prepare indices and weights for interpolation
        ##when dst_grid is set, these info are prepared and stored to avoid recalculating
        ##too many times, when applying the same interp to a lot of flds
        inside, vertices, in_coords, nearest = self.find_index(grid.x)
        self.interp_inside = inside
        self.interp_vertices = vertices
        self.interp_nearest = nearest
        self.interp_weights = self._interp_weights(inside, vertices, in_coords)

        ##prepare indices for coarse-graining
        inside, _, _, nearest = self.dst_grid.find_index(self.x)
        self.coarsen_inside = inside
        self.coarsen_nearest = nearest

    def set_destination_grid(self, grid):
        """
        Set method for self.dst_grid the destination Grid object to convert to.
        """
        self.dst_grid = grid

    def wrap_cyclic(self, x_):
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
        if self.cyclic:
            x_ = np.mod(x_ - self.xmin, self.Lx) + self.xmin
        return x_

    def find_index(self, x_):
        x_ = np.array(x_).flatten()
        x_ = self.wrap_cyclic(x_)
        xi = self.x
        i_ = np.argsort(xi)
        xi_ = xi[i_]
        ##pad cyclic coordinates with additional grid point
        if self.cyclic:
            if xi_[0]+self.Lx not in xi_:
                xi_ = np.hstack((xi_, xi_[0] + self.Lx))
                i_ = np.hstack((i_, i_[0]))

        pi = np.array(np.searchsorted(xi_, x_, side='right'))
        inside = ~np.logical_or(pi==len(xi_), pi==0)
        pi = pi[inside]

        vertices = np.zeros(pi.shape+(2,), dtype=int)
        vertices[:, 0] = i_[pi-1]
        vertices[:, 1] = i_[pi]
        in_coords = (x_[inside] - xi_[pi-1]) / (xi_[pi] - xi_[pi-1])

        nearest = np.zeros(pi.shape, dtype=int)
        ind = np.where(in_coords<0.5)
        nearest[ind] = i_[pi-1][ind]
        ind = np.where(in_coords>=0.5)
        nearest[ind] = i_[pi][ind]

        return inside, vertices, in_coords, nearest

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
        ##compute bilinear interp weights
        interp_weights = np.zeros(vertices.shape)
        interp_weights[:, 0] =  1-in_coords
        interp_weights[:, 1] =  in_coords
        return interp_weights

    def interp(self, fld, x=None, method='linear'):
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
        if x is None:
            ##use precalculated weights for self.dst_grid
            inside = self.interp_inside
            vertices = self.interp_vertices
            nearest = self.interp_nearest
            weights = self.interp_weights
            x = self.dst_grid.x
        else:
            ##otherwise compute the weights for the given x,y
            inside, vertices, in_coords, nearest = self.find_index(x)
            weights = self._interp_weights(inside, vertices, in_coords)

        fld_interp = np.full(np.array(x).flatten().shape, np.nan)
        if fld.shape == self.x.shape:
            if method == 'nearest':
                # find the node of the triangle with the maximum weight
                fld_interp[inside] = fld.flatten()[nearest]
            elif method == 'linear':
                # sum over the weights for each node of triangle
                fld_interp[inside] = np.einsum('nj,nj->n', np.take(fld.flatten(), vertices), weights)
            else:
                raise ValueError("'method' should be 'nearest' or 'linear'")
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
        else:
            raise ValueError("field shape does not match grid shape")

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

    def convert(self, fld, is_vector=False, method='linear', coarse_grain=False, **kwargs):
        """
        Main method to convert from self to dst_grid coordinate systems:
        """
        if self.dst_grid != self:
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

    def distance(self, ref_x, x, p=1):
        """
        Compute distance for points x to the reference point
        """
        ##normal cartesian distances in x and y
        dist_x = np.abs(x-ref_x)
        ##if there are cyclic boundary condition, take care of the wrap around
        if self.cyclic:
            dist_x = np.minimum(dist_x, self.Lx - dist_x)
        return dist_x

