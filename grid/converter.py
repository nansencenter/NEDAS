import numpy as np
from pyproj import Proj, Transformer
from matplotlib.tri import Triangulation

class Converter(object):
    def __init__(self, proj1, x1, y1, proj2, x2, y2):
        ##initialize and check
        self.proj1 = proj1
        self.x1 = x1
        self.y1 = y1
        self.proj2 = proj2
        self.x2 = x2
        self.y2 = y2
        assert isinstance(proj1, Proj), "proj1 is not pyproj.Proj instance"
        assert isinstance(proj2, Proj), "proj2 is not pyproj.Proj instance"
        assert y1.shape == self.x1.shape, "x1, y1 shape does not match"
        assert y2.shape == self.x2.shape, "x2, y2 shape does not match"

        ##pyproj.transformer to go forward/backward between proj1 and proj2
        self.pj_fwd = Transformer.from_proj(proj1, proj2).transform
        self.pj_bwd = Transformer.from_proj(proj2, proj1).transform

        ##find triangulation of grid in proj1
        self.tri = Triangulation(x1.flatten(), y1.flatten(), triangles=None)
        self.tri_finder = self.tri.get_trifinder()
        self.num_triangles = len(self.tri.triangles)

        ##get average size of triangles, i.e. grid spacing, used in finding rotate_matrix
        self._set_grid_spacing()

        ###convert (x1,y1) to corresponding coordinates in proj2, we call it (x,y)
        self.x, self.y = self.pj_fwd(self.x1, self.y1)

        ###prepare matrix to rotate vectors in proj1 to proj2:
        self._set_rotate_matrix()

        ###prepare weights for interpolation from (x,y) to (x2,y2) in proj2:
        self._set_t_matrix()
        self._set_interp_weights()

    ##utility functions for rotate_vectors
    def _set_grid_spacing(self):
        x = self.tri.x
        y = self.tri.y
        t = self.tri.triangles
        dx = np.min(np.sqrt((x[t[:, 0]] - x[t[:, 1]])**2 + (y[t[:, 0]] - y[t[:, 1]])**2))
        self.grid_spacing = dx

    def _set_rotate_matrix(self):
        x = self.x
        y = self.y

        ###find increments in x, y directions in proj2
        ##increment is set according to grid spacing of x2, y2 grid
        eps = 0.1 * self.grid_spacing
        xu, yu = self.pj_fwd(self.x1 + eps, self.y1      )  ##move a bit in x dirn
        xv, yv = self.pj_fwd(self.x1      , self.y1 + eps)  ##move a bit in y dirn

        self.rotate_matrix = np.zeros((4,)+self.x1.shape)
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

    def rotate_vectors(self, vec_fld):
        u = vec_fld[0, :]
        v = vec_fld[1, :]
        assert u.shape == self.x1.shape, "vector field shape does not match x1.shape"
        assert vec_fld.shape[0] == 2, "vector field should have first dim==2, containing u,v"

        vec_fld_rot = vec_fld.copy()
        rw = self.rotate_matrix
        u_rot = np.full(u.shape, np.nan)
        v_rot = np.full(u.shape, np.nan)
        u_rot = rw[0, :]*u + rw[1, :]*v
        v_rot = rw[2, :]*u + rw[3, :]*v
        vec_fld_rot[0, :] = u_rot
        vec_fld_rot[1, :] = v_rot

        return vec_fld_rot

    ###utility functions for interpolation
    def _set_t_matrix(self):
        ###Used for getting the barycentric coordinates on a triangle.
        ###For the i-th triangle,
        ###    self.t_matrix[i] = [[a', b'], [c', d'], [x_3, y3]]
        ###    where the first 2 rows are the inverse of the matrix T in the wikipedia link
        ###    and (x_3, y_3) are the coordinates of the 3rd vertex of the triangle
        x = self.tri.x[self.tri.triangles]
        y = self.tri.y[self.tri.triangles]
        a = x[:,0] - x[:,2]
        b = x[:,1] - x[:,2]
        c = y[:,0] - y[:,2]
        d = y[:,1] - y[:,2]
        det = a*d-b*c
        self.t_matrix = np.zeros((self.num_triangles, 3, 2))
        self.t_matrix[:,0,0] = d/det
        self.t_matrix[:,0,1] = -b/det
        self.t_matrix[:,1,0] = -c/det
        self.t_matrix[:,1,1] = a/det
        self.t_matrix[:,2,0] = x[:,2]
        self.t_matrix[:,2,1] = y[:,2]

    def _set_interp_weights(self):
        x_, y_ = self.pj_bwd(self.x2, self.y2)
        self.dst_points = np.array([x_.flatten(), y_.flatten()]).T
        self.triangle_map = self.tri_finder(x_, y_)
        self.dst_mask = (self.triangle_map < 0)
        self.triangle_map[self.dst_mask] = 0
        self.inside = ~self.dst_mask.flatten()

        ##get barycentric coords, according to https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_triangles, each row of bary is (lambda_1, lambda_2) for 1 destination point
        d = 2
        inds = self.triangle_map.flatten()[self.inside]
        self.vertices = np.take(self.tri.triangles, inds, axis=0)
        temp = np.take(self.t_matrix, inds, axis=0)
        delta = self.dst_points[self.inside] - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)

        # set interpolation weights
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
            assert fld.shape[1:] == self.x1.shape, "vector field shape does not match x1"
            fld_ = self.rotate_vectors(fld)

            ##now perform interpolation from (x,y) to (x2,y2)
            fld_out = np.full((2,)+self.x2.shape, np.nan)
            fld_out[0, :] = self._interp_nodes(fld_[0, :], method)
            fld_out[1, :] = self._interp_nodes(fld_[1, :], method)

        else:
            ##if fld is scalar, just interpolate
            fld_ = fld.flatten()
            if fld.shape == self.x1.shape:
                fld_out = self._interp_nodes(fld, method)

            elif len(fld_) == self.num_triangles:
                fld_out = self._interp_elements(fld_)

            else:
                raise ValueError("field shape does not match x1.shape, or number of triangle elements in proj1")

        return fld_out


