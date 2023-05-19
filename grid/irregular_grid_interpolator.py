###adapted from nextsim-tools/pynextsim

import numpy as np
from matplotlib.tri import Triangulation

class IrregularGridInterpolator(object):
    def __init__(self, x0, y0, x1, y1, triangles=None):
        '''
        Parameters:
        -----------
        x0 : np.ndarray(float)
            x-coords of source points
        y0 : np.ndarray(float)
            y-coords of source points
        x1 : np.ndarray(float)
            x-coords of destination points
        y1 : np.ndarray(float)
            y-coords of destination points
        triangles : np.ndarray(int)
            shape (num_triangles, 3)
            indices of nodes for each triangle

        Sets:
        -----
        self.inside: np.ndarray(bool)
            shape = (num_target_points,)
        self.vertices: np.ndarray(int)
            shape = (num_good_target_points, 3)
            good target points are those inside the source triangulation
        self.weights: np.ndarray(float)
            shape = (num_good_target_points, 3)
            good target points are those inside the source triangulation

        Follows this suggestion:
        https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
        x_target[i] = \sum_{j=0}^2 weights[i, j]*x_source[vertices[i, j]]
        y_target[i] = \sum_{j=0}^2 weights[i, j]*y_source[vertices[i, j]]
        We can do (linear) interpolation by replacing x_target, x_source with z_target, z_source
        where z_source is the field to be interpolated and z_target is the interpolated field
        '''

        # define and triangulate source points
        self.src_shape = x0.shape
        self.src_points = np.array([x0.flatten(), y0.flatten()]).T
        self.tri = Triangulation(x0.flatten(), y0.flatten(), triangles=triangles)
        self.tri_finder = self.tri.get_trifinder()
        self.num_triangles = len(self.tri.triangles)
        self._set_transform()

        # define target points
        self.dst_points = np.array([x1.flatten(), y1.flatten()]).T
        self.dst_shape = x1.shape
        self.triangle_map = self.tri_finder(x1, y1)
        self.dst_mask = (self.triangle_map < 0)
        self.triangle_map[self.dst_mask] = 0
        self.inside = ~self.dst_mask.flatten()

        """
        get barycentric coords
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_triangles
        each row of bary is (lambda_1, lambda_2) for 1 destination point
        """
        d = 2
        inds = self.triangle_map.flatten()[self.inside]
        self.vertices = np.take(self.tri.triangles, inds, axis=0)
        temp = np.take(self.transform, inds, axis=0)
        delta = self.dst_points[self.inside] - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)

        # set weights
        self.weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def _set_transform(self):
        """
        Used for getting the barycentric coordinates on a triangle.
        Follows:
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_triangles
        
        Sets:
        -----
        self.transform : numpy.ndarray
            For the i-th triangle,
                self.transform[i] = [[a', b'], [c', d'], [x_3, y3]]
            where the first 2 rows are the inverse of the matrix T in the wikipedia link
            and (x_3, y_3) are the coordinates of the 3rd vertex of the triangle
        """
        x = self.tri.x[self.tri.triangles]
        y = self.tri.y[self.tri.triangles]
        a = x[:,0] - x[:,2]
        b = x[:,1] - x[:,2]
        c = y[:,0] - y[:,2]
        d = y[:,1] - y[:,2]
        det = a*d-b*c

        self.transform = np.zeros((self.num_triangles, 3, 2))
        self.transform[:,0,0] = d/det
        self.transform[:,0,1] = -b/det
        self.transform[:,1,0] = -c/det
        self.transform[:,1,1] = a/det
        self.transform[:,2,0] = x[:,2]
        self.transform[:,2,1] = y[:,2]

    def interp_field(self, fld, method='linear'):
        """
        Interpolate field from elements elements or nodes of source triangulation
        to destination points

        Parameters:
        -----------
        fld: np.ndarray
            field to be interpolated
        method : str
            interpolation method if interpolating from nodes 
            - 'linear'  : linear interpolation
            - 'nearest' : nearest neighbour

        Returns:
        -----------
        fld_interp : np.ndarray
            field interpolated onto the destination points
        """
        if fld.shape == self.src_shape:
            return self._interp_nodes(fld, method=method)
        fld_ = fld.flatten()
        if len(fld_) == self.num_triangles:
            return self._interp_elements(fld_)
        msg = f"""Field to interpolate should have the same size as the source points
        i.e. {self.src_shape}, or be a vector with the same number of triangles
        as the source triangulation i.e. self.num_triangles"""
        raise ValueError(msg)

    def _interp_elements(self, fld):
        """
        Interpolate field from elements of source triangulation to destination points

        Parameters:
        -----------
        fld: np.ndarray
            field to be interpolated

        Returns:
        -----------
        fld_interp : np.ndarray
            field interpolated onto the destination points
        """
        fld_interp = fld[self.triangle_map]
        fld_interp[self.dst_mask] = np.nan
        return fld_interp

    def _interp_nodes(self, fld, method='linear'):
        """
        Interpolate field from nodes of source triangulation to destination points

        Parameters:
        -----------
        fld: np.ndarray
            field to be interpolated
        method : str
            interpolation method 
            - 'linear'  : linear interpolation
            - 'nearest' : nearest neighbour

        Returns:
        -----------
        fld_interp : np.ndarray
            field interpolated onto the destination points
        """
        ndst = self.dst_points.shape[0]
        fld_interp = np.full((ndst,), np.nan)
        w = self.weights
        if method == 'linear':
            # sum over the weights for each node of triangle
            v = self.vertices # shape = (ngood,3)
            fld_interp[self.inside] = np.einsum(
                    'nj,nj->n', np.take(fld.flatten(), v), w)

        elif method == 'nearest':
            # find the node of the triangle with the maximum weight
            v = np.array(self.vertices) # shape = (ngood,3)
            v = v[np.arange(len(w), dtype=int), np.argmax(w, axis=1)] # shape = (ngood,)
            fld_interp[self.inside] = fld.flatten()[v]

        else:
            raise ValueError("'method' should be 'nearest' or 'linear'")

        return fld_interp.reshape(self.dst_shape)
