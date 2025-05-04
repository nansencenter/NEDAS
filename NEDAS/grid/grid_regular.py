import copy
import numpy as np
from matplotlib import colormaps
from NEDAS.grid.grid_base import GridBase

class RegularGrid(GridBase):
    """
    Regular 2D grid class

    Args:
        pole_dim (str, optional):
            `None` (default), if one of the dimension has poles, :code:`'x'` or :code:`'y'`
        pole_index (str, optional):
            `None` (default), tuple of the pole index(s) in `pole_dim`
        neighbors (np.ndarray, optional):
            `None` (default), for regular grid with special geometry (e.g. tripolar ocean grid),
            `neighbors` stores the j,i index of 4 neighors (east, north, west, and south) on each grid point.
            Since specifying neighbors already take care of cyclic boundary conditions, `cyclic_dim` will be discarded if `neighbors` is set.

    """
    def __init__(self, proj, x, y, bounds=None, cyclic_dim=None, distance_type='cartesian',
                 pole_dim=None, pole_index=None, neighbors=None, dst_grid=None,):
        super().__init__(proj, x, y, bounds, cyclic_dim, distance_type, dst_grid)
        self.regular = True
        self.pole_dim = pole_dim
        self.pole_index = pole_index
        self.neighbors = neighbors
        if self.neighbors is not None and self.cyclic_dim is not None:
            print('neighbors already implemented, discarding cyclic_dim')
            self.cyclic_dim = None

        self.nx = self.x.shape[1]
        self.ny = self.x.shape[0]
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.dy = (self.ymax - self.ymin) / (self.ny - 1)
        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy
        self.npoints = self.nx * self.ny

    def change_resolution_level(self, nlevel):
        """
        Generate a new grid with changed resolution.

        Args:
            nlevel (int):
                Positive number, downsample grid abs(nlevel) times, each time doubling the grid spacing;
                Negative number, upsample grid abs(nlevel) times, each time halving the grid spacing

        Returns:
            A new grid object with changed resolution.
        """
        if nlevel == 0:
            return self
        else:
            ##create a new grid object with x,y at new resolution level
            self._dst_grid = None
            new_grid = copy.deepcopy(self)
            fac = 2**nlevel
            new_grid.dx = self.dx * fac
            new_grid.dy = self.dy * fac
            new_grid.nx = int(np.round(self.Lx / new_grid.dx))
            new_grid.ny = int(np.round(self.Ly / new_grid.dy))
            assert min(new_grid.nx, new_grid.ny) > 1, "Grid.change_resolution_level: new resolution too low, try smaller nlevel"
            new_grid.x, new_grid.y = np.meshgrid(self.xmin + np.arange(new_grid.nx) * new_grid.dx, self.ymin + np.arange(new_grid.ny) * new_grid.dy)
            ##coarsen the mask
            self.set_destination_grid(new_grid)
            new_grid.mask = self.convert(self.mask, method='nearest').astype(bool)
            return new_grid

    def find_index(self, x_, y_):
        x_ = np.array(x_).flatten()
        y_ = np.array(y_).flatten()

        ##lon: pyproj.Proj works only for lon=-180:180
        if self.proj_name == 'longlat':
            x_ = np.mod(x_ + 180., 360.) - 180.

        ###account for cyclic dim, when points drop "outside" then wrap around
        x_, y_ = self._wrap_cyclic_xy(x_, y_)

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
        
        return inside, indices, vertices, in_coords, nearest

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
        vec_fld = super().rotate_vectors(vec_fld)
        for i in range(2):
            vec_fld[i,...] = self._fill_pole_void(vec_fld[i,...])
        return vec_fld

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

        Args:
            inside, vertices, in_coords: from the output of self.find_index

        Returns:
            interp_weights (np.array): interpolation weights
        """
        ##compute bilinear interp weights
        interp_weights = np.zeros(vertices.shape)
        interp_weights[:, 0] =  (1-in_coords[:, 0]) * (1-in_coords[:, 1])
        interp_weights[:, 1] =  in_coords[:, 0] * (1-in_coords[:, 1])
        interp_weights[:, 2] =  in_coords[:, 0] * in_coords[:, 1]
        interp_weights[:, 3] =  (1-in_coords[:, 0]) * in_coords[:, 1]
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
            if method == 'nearest':
                # find the node of the triangle with the maximum weight
                fld_interp[inside] = fld.flatten()[nearest]
            elif method == 'linear':
                # sum over the weights for each node of triangle
                fld_interp[inside] = np.einsum('nj,nj->n', np.take(fld.flatten(), vertices), weights)
            else:
                raise NotImplementedError(f"interp method {method} is not yet available")
        else:
            raise ValueError(f"field shape {fld.shape} does not match grid shape {self.x.shape}")
        return fld_interp.reshape(np.array(x).shape)

    ###utility function for coarse-graining (high->low resolution)
    def coarsen(self, fld):
        ##find which location x_,y_ falls in in dst_grid
        if fld.shape == self.x.shape:
            inside = self.coarsen_inside
            nearest = self.coarsen_nearest
        else:
            raise ValueError(f"field shape {fld.shape} does not match grid shape {self.x.shape}")

        fld_coarse = np.zeros(self.dst_grid.x.flatten().shape)
        count = np.zeros(self.dst_grid.x.flatten().shape)
        fld_inside = fld.flatten()[inside]
        valid = ~np.isnan(fld_inside)  ##filter out nan

        ##average the fld points inside each dst_grid box
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

        x = self.x
        y = self.y
        ##in case of lon convention 0:360, need to reorder so that x is monotonic
        if self.proj_name == 'longlat':
            ind = np.argsort(x[0,:])
            x = np.take(x, ind, axis=1)
            fld = np.take(fld, ind, axis=1)
        im = ax.pcolor(x, y, fld, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

        self.set_xylim(ax)
        return im
