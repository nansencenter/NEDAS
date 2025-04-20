import numpy as np
import copy

class Grid1D:
    """
    Grid class to handle fields defined on a 1D grid

    Methods convert, interp and distance has the same input args as their counterparts in the Grid class.
    This introduces y coordinates in Grid1D class, although y is not defined for 1D grid, this makes the code that
    calles Grid/Grid1D methods to be easier to maintain.
    """
    def __init__(self, x, bounds=None, regular=True, cyclic=False, distance_type='cartesian', dst_grid=None):

        ##coordinates and properties of the 2D grid
        self.x = x
        self.y = np.zeros(x.size)  ##dummy y coords
        self.regular = regular
        self.cyclic = cyclic
        if bounds is not None:
            self.xmin, self.xmax = bounds
        else:
            self.xmin = np.min(self.x)
            self.xmax = np.max(self.x)
        self.ymin, self.ymax = 0, 0

        self.nx = self.x.size
        self.ny = 1
        if regular:
            self.dx = (self.xmax - self.xmin) / (self.nx - 1)
            self.Lx = self.nx * self.dx
        else:
            self.dx = np.mean(np.diff(np.sort(self.x)))
            self.Lx = self.xmax - self.xmin
        self.Ly = 0
        self.mfx, self.mfy = 1, 1

        if distance_type != 'cartesian':
            raise ValueError(f"distance_type {distance_type} not supported for 1D grid.")

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
        self = cls.__new__(cls)
        x = np.arange(xstart, xend, dx)
        if centered:
            x += 0.5*dx  ##move coords to center of grid box
        self.__init__(x, regular=True, **kwargs)
        return self

    @classmethod
    def random_grid(cls, xstart, xend, npoints, **kwargs):
        self = cls.__new__(cls)
        x = np.random.uniform(0, 1, npoints) * (xend - xstart) + xstart
        self.__init__(x, bounds=[xstart, xend], regular=False, **kwargs)
        return self

    def change_resolution_level(self, nlevel):
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
        self.dst_grid = grid

    def wrap_cyclic(self, x_):
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
        ##compute bilinear interp weights
        interp_weights = np.zeros(vertices.shape)
        interp_weights[:, 0] =  1-in_coords
        interp_weights[:, 1] =  in_coords
        return interp_weights

    def interp(self, fld, x=None, y=None, method='linear'):
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

    def distance(self, ref_x, x, ref_y=None, y=None, p=1):
        ##normal cartesian distances in x and y
        dist_x = np.abs(x - ref_x)
        ##if there are cyclic boundary condition, take care of the wrap around
        if self.cyclic:
            dist_x = np.minimum(dist_x, self.Lx - dist_x)
        return dist_x
