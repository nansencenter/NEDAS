import numpy as np
from .grid import Grid

class Converter(object):
    def __init__(self, grid1, grid2):
        assert isinstance(grid1, Grid), "grid1 is not a Grid instance"
        assert isinstance(grid2, Grid), "grid2 is not a Grid instance"
        self.grid1 = grid1
        self.grid2 = grid2
        self.x1 = grid1.x
        self.y1 = grid1.y
        self.x2 = grid2.x
        self.y2 = grid2.y

        if grid1.cyclic_dim != None:
            self._pad_cyclic_dim_xy()

        ##rotate vector field from proj1 to proj2
        self._set_rotate_matrix()

        ##interpolate field from x1,y1 to x2,y2
        if grid1.regular:   ###use bilinear interp for regular grid
            self._set_interp_weights_regular()

        else:    ###use barycentric coordinate weights for unstruct mesh interp
            self.tri = self.grid1.tri
            self.tri_finder = self.tri.get_trifinder()
            self.num_triangles = len(self.tri.triangles)
            self._set_interp_weights_irregular()

    def _pad_cyclic_dim_xy(self):
        x_ = self.x1.copy()
        y_ = self.y1.copy()
        ny, nx = x_.shape
        for d in self.grid1.cyclic_dim:
            if d=='x':
                Lx = self.grid1.dx * nx
                if np.all(np.diff(x_[0, :]) > 0):
                    x_ = np.vstack((x_.T, x_[:, 0]+Lx)).T
                    y_ = np.vstack((y_.T, y_[:, 0])).T
                else:
                    x_ = np.vstack((x_[:, -1]+Lx, x_.T)).T
                    y_ = np.vstack((y_[:, -1], y_.T)).T
            elif d=='y':
                Ly = self.grid1.dy * ny
                if np.all(np.diff(y_[:, 0]) > 0):
                    x_ = np.vstack((x_, x_[0, :]))
                    y_ = np.vstack((y_, y_[0, :]+Ly))
                else:
                    x_ = np.vstack((x_[-1, :], x_))
                    y_ = np.vstack((y_[-1, :]+Ly, y_))
        self.x1 = x_
        self.y1 = y_

    def _pad_cyclic_dim(self, fld):
        for d in self.grid1.cyclic_dim:
            if d=='x':
                if np.all(np.diff(self.grid1.x[0, :]) > 0):
                    fld = np.vstack((fld.T, fld[:, 0])).T
                else:
                    fld = np.vstack((fld[:, -1], fld.T)).T
            elif d=='y':
                if np.all(np.diff(self.grid1.y[:, 0]) > 0):
                    fld = np.vstack((fld, fld[0, :]))
                else:
                    fld = np.vstack((fld[-1, :], fld))
        return fld

    def _proj(self, x, y, forward=True):
        if forward:
            lon, lat = self.grid1.proj(x, y, inverse=True)
            x_, y_ = self.grid2.proj(lon, lat)
            x_, y_ = self.grid2.wrap_cyclic_xy(x_, y_)
            return x_, y_
        else:
            lon, lat = self.grid2.proj(x, y, inverse=True)
            x_, y_ = self.grid1.proj(lon, lat)
            x_, y_ = self.grid1.wrap_cyclic_xy(x_, y_)
            return x_, y_

    def _set_rotate_matrix(self):
        self.rotate_matrix = np.zeros((4,)+self.x1.shape)
        if self.grid1.proj != self.grid2.proj:
            ##corresponding x1,y1 coordinates in proj2, call them x,y
            x, y = self._proj(self.x1, self.y1)

            ##find small increments in x,y due to small changes in x1,y1 in proj2
            eps = 0.1 * self.grid1.dx    ##grid spacing is specified in Grid object
            xu, yu = self._proj(self.x1 + eps, self.y1      )  ##move a bit in x dirn
            xv, yv = self._proj(self.x1      , self.y1 + eps)  ##move a bit in y dirn

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
        else:
            self.rotate_matrix[0, :] = 1.
            self.rotate_matrix[1, :] = 0.
            self.rotate_matrix[2, :] = 0.
            self.rotate_matrix[3, :] = 1.

    def _fill_pole_void(self, fld):
        if self.grid1.pole_dim == 'x':
            for i in self.grid1.pole_index:
                if i==0:
                    fld[:, 0] = np.mean(fld[:, 1])
                if i==-1:
                    fld[:, -1] = np.mean(fld[:, -2])
        if self.grid1.pole_dim == 'y':
            for i in self.grid1.pole_index:
                if i==0:
                    fld[0, :] = np.mean(fld[1, :])
                if i==-1:
                    fld[-1, :] = np.mean(fld[-2, :])
        return fld

    def _rotate_vectors(self, vec_fld):
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

    ###utility functions for interpolation
    def _set_interp_weights_regular(self):
        x, y = self._proj(self.x2, self.y2, forward=False)
        x_ = x.flatten()
        y_ = y.flatten()

        ###find indices id_x, id_y that x_,y_ falls in
        id_x, id_y = self.grid1.find_index(self.x1, self.y1, x_, y_)

        self.inside = ~np.logical_or(np.logical_or(id_y==self.x1.shape[0], id_y==0),
                                     np.logical_or(id_x==self.x1.shape[1], id_x==0))

        ###indices for the 4 vertices of each rectangular grid cell
        rectangles = np.zeros((x_.shape[0], 4), dtype=int)
        nx = self.x1.shape[1]
        rectangles[:, 0] =     id_y*nx +   id_x
        rectangles[:, 1] = (id_y-1)*nx +   id_x
        rectangles[:, 2] = (id_y-1)*nx + id_x-1
        rectangles[:, 3] =     id_y*nx + id_x-1
        self.vertices = rectangles[self.inside, :]

        ##compute bilinear interp weights
        x1_ = self.x1.flatten()
        y1_ = self.y1.flatten()
        x_o = x_[self.inside]
        y_o = y_[self.inside]
        x_1 = x1_[self.vertices][:,3]
        x_2 = x1_[self.vertices][:,0]
        y_1 = y1_[self.vertices][:,1]
        y_2 = y1_[self.vertices][:,0]
        s = (x_1 - x_2) * (y_1 - y_2)
        self.interp_weights = np.zeros(x_o.shape+(4,))
        self.interp_weights[:, 0] =  (x_o - x_1)*(y_o - y_1)/s
        self.interp_weights[:, 1] = -(x_o - x_1)*(y_o - y_2)/s
        self.interp_weights[:, 2] =  (x_o - x_2)*(y_o - y_2)/s
        self.interp_weights[:, 3] = -(x_o - x_2)*(y_o - y_1)/s

    def _set_interp_weights_irregular(self):
        x_, y_ = self._proj(self.x2, self.y2, forward=False)
        self.dst_points = np.array([x_.flatten(), y_.flatten()]).T
        self.triangle_map = self.tri_finder(x_, y_)
        self.dst_mask = (self.triangle_map < 0)
        self.triangle_map[self.dst_mask] = 0
        self.inside = ~self.dst_mask.flatten()

        ##transform matrix for barycentric coords computation
        x = self.tri.x[self.tri.triangles]
        y = self.tri.y[self.tri.triangles]
        a = x[:,0] - x[:,2]
        b = x[:,1] - x[:,2]
        c = y[:,0] - y[:,2]
        d = y[:,1] - y[:,2]
        det = a*d-b*c
        t_matrix = np.zeros((self.num_triangles, 3, 2))
        t_matrix[:,0,0] = d/det
        t_matrix[:,0,1] = -b/det
        t_matrix[:,1,0] = -c/det
        t_matrix[:,1,1] = a/det
        t_matrix[:,2,0] = x[:,2]
        t_matrix[:,2,1] = y[:,2]

        ##get barycentric coords, according to https://en.wikipedia.org/wiki/
        ##Barycentric_coordinate_system#Barycentric_coordinates_on_triangles,
        ##each row of bary is (lambda_1, lambda_2) for 1 destination point.
        inds = self.triangle_map.flatten()[self.inside]
        self.vertices = np.take(self.tri.triangles, inds, axis=0)
        temp = np.take(t_matrix, inds, axis=0)
        delta = self.dst_points[self.inside] - temp[:, 2]
        bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
        self.interp_weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def _interp_elements(self, fld):
        fld_interp = fld[self.triangle_map]
        fld_interp[self.dst_mask] = np.nan
        return fld_interp

    def _interp_nodes(self, fld, method):
        fld_interp = np.full(self.x2.flatten().shape, np.nan)

        w = self.interp_weights
        if method == 'linear':
            # sum over the weights for each node of triangle
            v = self.vertices
            fld_interp[self.inside] = np.einsum('nj,nj->n', np.take(fld.flatten(), v), w)

        elif method == 'nearest':
            # find the node of the triangle with the maximum weight
            v = np.array(self.vertices)
            v = v[np.arange(len(w), dtype=int), np.argmax(w, axis=1)]
            fld_interp[self.inside] = fld.flatten()[v]

        else:
            raise ValueError("'method' should be 'nearest' or 'linear'")

        return fld_interp.reshape(self.x2.shape)

    ### Method to convert from proj1, x1, y1 to proj2, x2, y2 coordinate systems:
    ###  Steps: 1. find corresponding x, y in proj2 from (proj1, x1, y1)
    ###         2. rotate vectors in proj1 if is_vector
    ###         3. interpolate fld in proj2 from (x, y) to (x2, y2)
    ###  To speed up, the rotate and interpolate weights are computed once and stored.
    ###  Some functions are adapted from nextsim-tools/pynextsim:
    ###  lib.py:transform_vectors, irregular_grid_interpolator.py
    def convert(self, fld, is_vector=False, method='linear'):
        if is_vector:
            ##vector field defined on proj1 needs to rotate to proj2 before interp
            assert fld.shape[0] == 2, "vector field should have first dim==2, for u,v component"
            if self.grid1.cyclic_dim != None:
                fld = np.array([self._pad_cyclic_dim(fld[0, :]), self._pad_cyclic_dim(fld[1, :])])
            assert fld.shape[1:] == self.x1.shape, "vector field shape does not match x1"

            fld = self._rotate_vectors(fld)

            fld_out = np.full((2,)+self.x2.shape, np.nan)
            ##now perform interpolation from (x,y) to (x2,y2)
            fld_out[0, :] = self._interp_nodes(fld[0, :], method)
            fld_out[1, :] = self._interp_nodes(fld[1, :], method)

        else:
            ##if fld is scalar, just interpolate
            if fld.shape == self.grid1.x.shape:
                if self.grid1.cyclic_dim != None:
                    fld = self._pad_cyclic_dim(fld)
                fld_out = self._interp_nodes(fld, method)

            elif not self.grid1.regular and len(fld.flatten()) == self.num_triangles:
                fld_out = self._interp_elements(fld.flatten())

            else:
                raise ValueError("field shape does not match x1.shape, or number of triangle elements in proj1")

        return fld_out

