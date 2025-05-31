import os
import inspect
from functools import cached_property
from abc import ABC, abstractmethod
import numpy as np
import shapefile
from pyproj import Proj, Geod
from matplotlib import colormaps
from NEDAS.utils.graphics import draw_line, draw_patch, arrowhead_xy, draw_reference_vector_legend

class GridBase(ABC):
    """
    Base class to handle 2D fields defined on regular grids or unstructured meshes.

    Args:
        proj (pyproj.Proj, custom func, None):
            Projection function mapping from longitude,latitude to x,y coordinates.
            If None, a default Mercator projection will be used.
        x (np.ndarray): X-coordinates for each grid point.
        y (np.ndarray): Y-coordinates for each grid point.
        bounds (list, optional):
            Grid boundary limits, [xmin, xmax, ymin, ymax], all float numbers.
            If not specified, will use min/max value of the coordinates.
        regular (bool, optional):
            Whether grid is regular or unstructured. Default is True (regular grid).
        cyclic_dim (str, optional):
            Cyclic dimension(s): ``'x'``, ``'y'``, ``'xy'``, or ``None`` if noncyclic.
        distance_type (str, optional): Type of distance functions: `cartesian` (default) or `spherical`.
        dst_grid (GridBase, optional): Destination grid object to convert to.

    Attributes:
        proj (pyproj.Proj or custom function): Projection function.
        proj_name (str): Name of the projection, empty if not available.
        bounds (list): Grid boundary limits [xmin, xmax, ymin, ymax].
        mask (np.ndarray):
            Mask (bool) for points that are not participating the analysis,same shape as :code:`x`, default is all False.
    """
    def __init__(self, proj, x, y, bounds=None, cyclic_dim=None, distance_type='cartesian', dst_grid=None):
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
        self.cyclic_dim = cyclic_dim

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

        self.distance_type = distance_type

        self.mask = np.full(self.x.shape, False)

        self._dst_grid = None
        if dst_grid is not None:
            self.set_destination_grid(dst_grid)

    def __eq__(self, other):
        if not isinstance(other, GridBase):
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

    @property
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

    @property
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
        assert isinstance(grid, GridBase), "dst_grid should be a Grid instance"
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

        Args:
            x_, y_ (np.array): x, y coordinates of a grid

        Returns:
            Same as input but x, y values are wrapped to be within the boundaries again.
        """
        if self.cyclic_dim is not None:
            for d in self.cyclic_dim:
                if d=='x':
                    x_ = np.mod(x_ - self.xmin, self.Lx) + self.xmin
                elif d=='y':
                    y_ = np.mod(y_ - self.ymin, self.Ly) + self.ymin
        return x_, y_

    @abstractmethod
    def find_index(self, x_, y_):
        """
        Find indices of `self.x`, `self.y` corresponding to the given `x_`, `y_`.

        Args:
            x_ (float or np.ndarray): x-coordinates of target point(s).
            y_ (float or np.ndarray): y-coordinates of target point(s).

        Outputs:
            inside (np.ndarray of bool): Boolean array of shape `(x_.size,)` indicating whether
                each `x_`, `y_` point lies inside the grid.

            indices (np.ndarray of int or None): Indices of grid elements containing the input points.
                - For regular grids, this is `None` since vertices suffice to locate the grid box.
                - For unstructured meshes, these are indices into `tri.triangles`, from `tri_finder`.

            vertices (np.ndarray of int): Array of shape `(inside_size, n)`, where
                `n = 4` for regular grid boxes or `n = 3` for mesh triangles.
                These are indices into `self.x`, `self.y` (flattened) for the vertices
                of the grid element that each point falls in.

            in_coords (np.ndarray of float): Array of shape `(inside_size, n)` giving internal coordinates
                of each point within the containing element. Used to compute interpolation weights.

            nearest (np.ndarray of int): Array of shape `(inside_size,)` with indices of the grid nodes
                closest to each point.

        Notes:
            - This function assumes `self.x`, `self.y` define either a regular or triangular grid.
            - Internal coordinates are used for interpolation and vary in dimension based on the grid type.
        """
        pass

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

    def rotate_vectors(self, vec_fld):
        """
        Apply the rotate_matrix to a vector field

        Args:
            vec_fld (np.array): The input vector field, shape (2, self.x.shape)

        Returns:
            The vector field rotated to the dst_grid.
        """
        u = vec_fld[0, :].copy()
        v = vec_fld[1, :].copy()
        rw = self.rotate_matrix
        u_rot = rw[0, :]*u + rw[1, :]*v
        v_rot = rw[2, :]*u + rw[3, :]*v
        return np.array([u_rot, v_rot])

    @abstractmethod
    def interp(self, fld, x=None, y=None, method='linear'):
        """
        Interpolation of 2D field data (fld) from one grid (self or given x,y) to another (dst_grid).
        This can be used for grid refining (low->high resolution) or grid thinning (high->low resolution).
        This also converts between different grid geometries.

        Args:
            fld (np.array): Input field defined on self, should have same shape as self.x
            x,y (float or np.array): Optional;
                If x,y are specified, the function computes the weights and apply them to fld
                If x,y are None, the self.dst_grid.x,y are used.
                Since their interp_weights are precalculated by dst_grid.setter it will be efficient
                to run interp for many different input flds quickly.
            method (str): Interpolation method, can be 'nearest' or 'linear'

        Returns:
            The interpolated field defined on the destination grid
        """
        pass

    @abstractmethod
    def coarsen(self, fld):
        """
        Coarse-graining is sometimes needed when the dst_grid is at lower resolution than self.
        Since many points of self.x,y falls in one dst_grid box/element, it is better to
        average them to represent the field on the low-res grid, instead of interpolating
        only from the nearest points that will cause representation errors.

        Args:
            fld (np.array): Input field to perform coarse-graining on, it is defined on self.

        Returns:
            The coarse-grained field defined on self.dst_grid.
        """
        pass

    def convert(self, fld, is_vector=False, method='linear', coarse_grain=False):
        """
        Main method to convert from self.proj, x, y to dst_grid coordinate systems:

        Notes:
            1. if projection changes and is_vector, rotate vectors from self.proj to dst_grid.proj
            2.1 interpolate fld components from self.x,y to dst_grid.x,y
            2.2 if dst_grid is low-res, coarse_grain=True will perform coarse-graining

        Args:
            fld (np.array): Input field to perform convertion on.
            is_vector (bool, optional):
                If False (default) the input fld is a scalar field,
                otherwise the input fld is a vector field.
            method (str, optional):
                Interpolation method, 'linear' (default) or 'nearest' 
            coarse_grain (bool, optional):
                If True, the coarse-graining will be applied using self.coarsen(). The default is False.

        Returns:
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

    def distance(self, ref_x, x, ref_y, y, p=2, type='cartesian'):
        """
        Compute distance for points (x,y) to the reference point
        
        Args:
            ref_x, ref_y (float):
                reference point x,y coordinates
            x, y (np.array):
                points whose distance to the reference points will be computed
            p (int, optional):
                Minkowski p-norm order, default is 2
            type (str, optional):
                distance type, 'cartesian' (default) or 'spherical'
        
        Returns:
            Distances between x,y and the reference point ref_x, ref_y.
        """
        if type == 'cartesian':
            ##normal cartesian distances in x and y
            dist_x = np.abs(x - ref_x)
            if self.cyclic_dim is not None and 'x' in self.cyclic_dim:
                dist_x = np.minimum(dist_x, self.Lx - dist_x)
            dist_y = np.abs(y - ref_y)
            if self.cyclic_dim is not None and 'y' in self.cyclic_dim:
                dist_y = np.minimum(dist_y, self.Ly - dist_y)
            if p == 1:
                dist = dist_x + dist_y  ##Manhattan distance, order 1
            elif p == 2:
                dist = np.hypot(dist_x, dist_y)   ##Euclidean distance, order 2
            else:
                raise NotImplementedError(f"grid.distance: p-norm order {p} is not implemented for 2D grid")
            return dist

        ##compute spherical distance on Earth instead
        elif type == 'spherical':
            reflon, reflat = self.proj(ref_x, ref_y, inverse=True)
            lon, lat = self.proj(x, y, inverse=True)
            RE = 6371000.0
            invrad = np.pi / 180.
            rlon1 = np.atleast_1d(reflon) * invrad
            rlat1 = np.atleast_1d(reflat) * invrad
            rlon2 = np.atleast_1d(lon) * invrad
            rlat2 = np.atleast_1d(lat) * invrad
            ##from m_spherdist.F90 in enkf-topaz:
            cos_d = np.sin(rlat1) * np.sin(rlat2) + np.cos(rlat1) * np.cos(rlat2) * np.cos(rlon1 - rlon2)
            dist = RE * np.acos(np.clip(cos_d, -1, 1))
            ##Haversine formula to avoid precision loss
            # a = np.sin((rlat2 - rlat1) / 2)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin((rlon1 - rlon2) / 2)**2
            # dist = 2 * RE * np.asin(np.sqrt(a))
            return dist

        else:
            raise ValueError(f"unknown distance type '{type}'")

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

    def llgrid_xy(self, dlon:float, dlat:float):
        """
        Prepare a lon/lat grid to plot as reference lines

        Args:
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

    @abstractmethod
    def plot_field(self, ax, fld, vmin=None, vmax=None, cmap='viridis', **kwargs):
        """
        Plot a scalar field using pcolor/tripcolor

        Args:
            ax (matplotlib.pyplot.Axes): Axes handle for plotting
            fld (np.array): The scalar field for plotting
            vmin, vmax (float, optional):
                The minimum and maximum value range for the colormap, if not specified (None)
                the np.min, np.max of the input fld will be used.
            cmap (matplotlib colormap, or str, optional):
                Colormap used in the plot, default is 'viridis'
        """
        pass

    def plot_vectors(self, ax, vec_fld, V=None, L=None, spacing=0.5, num_steps=10,
                     linecolor='k', linewidth=1,
                     showref=False, ref_xy=(0.9, 0.9), refcolor='w', ref_units='',
                     showhead=True, headwidth=0.1, headlength=0.3):
        """
        Plot vector fields (improved version of matplotlib quiver)

        Args:
            ax (matplotlib.pyplot.Axes): Axes handle for plotting
            vec_fld (np.array): The vector field for plotting
            V (float, optional):
                Velocity scale, typical velocity value in vec_fld units. If not specified (None)
                a typical value 0.33*max(abs(vec_fld[0,:])) will be used.
            L (float, optional):
                Length scale, how long in x,y units do vectors with velocity V show in the plot
                If not specified (None), a typical value 0.05*self.Lx will be used.
            spacing (float, optional):
                Distance between vectors in both directions is given by spacing*L. Default is 0.5.
                This controls the density of vectors in the plot.
                You can provide a tuple (float, float) for spacings in (x, y) if you want them
                to be set differently.
            num_steps (int, optional):
                Default is 10. If num_steps=1, straight vectors (as in quiver) will be displayed.
                num_steps>1 lets you display curved trajectories, at each sub-step the velocity is
                re-interpolated at the new position along the trajectories. As num_steps get larger
                the trajectories are more detailed.
            linecolor (str or matplotlib color, optional):
                Line color for the vector lines, default is 'k'
            linewidth (float, optional):
                Line width for the vector lines, default is 1.
            showref (bool, optional):
                If True, show a legend box with a reference vector (size L) inside. Default is False.
            ref_xy (tuple, optional):
                The x,y relative coordinates (0-1) for the reference vector box, default is upper right corner.
            ref_color (str or matplotlib color, optional):
                Background color for the reference vector box, default is 'w' (white).
            ref_units (str, optional):
                Units to be included in the reference vector box, default is ''.
            showhead (bool, optional):
                If True (default), show the arrow head of the vectors
            headwidth (float, optional):
                Width of arrow heads relative to L, default is 0.1.
            headlength (float, optional):
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
            leng = leng + np.hypot(ut, vt) * dt

        ##plot the vector lines
        hl = headlength * L
        hw = headwidth * L
        for i in range(xtraj.shape[0]):
            ##plot trajectory at one output location
            ax.plot(xtraj[i, :], ytraj[i, :], color=linecolor, linewidth=linewidth, zorder=4)
            ##add vector head if traj is long and straight enough
            dist = np.hypot(xtraj[i,0]-xtraj[i,-1], ytraj[i,0]-ytraj[i,-1])
            if showhead and hl < leng[i] < 1.6*dist:
                ax.fill(*arrowhead_xy(xtraj[i,-1], xtraj[i,-2], ytraj[i,-1],ytraj[i,-2], hw, hl), color=linecolor, zorder=5)

        ##add reference vector
        if showref:
            xr, yr = ref_xy[0]*self.Lx+self.xmin, ref_xy[1]*self.Ly+self.ymin
            draw_reference_vector_legend(ax, xr, yr, V, L, hw, hl, refcolor, linecolor, ref_units)

        self.set_xylim(ax)

    def plot_scatter(self, ax, fld, vmin=None, vmax=None, nlevels=20, cmap='viridis', markersize=10, x=None, y=None, is_vector=False, **kwargs):
        """
        Same as plot_field/vectors, but showing individual scattered points instead
        This is more suitable for plotting observations in space
        """
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if vmin is None:
            vmin = np.nanmin(fld)
        if vmax is None:
            vmax = np.nanmax(fld)
        dv = (vmax - vmin) / nlevels

        if is_vector:
            assert fld.shape[0] == 2
            assert fld.shape[1:] == x.shape
            V = vmax
            L = 0.05 * self.Lx
            hl, hw = 0.3 * L, 0.15 * L
            ref_x, ref_y = 0.9 * self.Lx + self.xmin, 0.9 * self.Ly + self.ymin
            refcolor = kwargs.get('refcolor', 'w')
            ref_units = kwargs.get('units', '')
            d = fld * L / V
            xtraj, ytraj = np.array([x, x + d[0,...]]), np.array([y, y + d[1,...]])
            linecolor = kwargs.get('linecolor', 'k')
            linewidth = kwargs.get('linewidth', 1)
            for i in np.ndindex(x.shape):
                ax.plot(xtraj[:,i], ytraj[:,i], color=linecolor, linewidth=linewidth, zorder=5)
                dist = np.hypot(xtraj[0,i]-xtraj[1,i], ytraj[0,i]-ytraj[1,i])
                if hl < 1.6*dist:
                    ax.fill(*arrowhead_xy(xtraj[1,i], xtraj[0,i], ytraj[1,i], ytraj[0,i], hw, hl), color=linecolor, zorder=5)
            draw_reference_vector_legend(ax, ref_x, ref_y, V, L, hw, hl, refcolor, linecolor, ref_units)

        else:
            assert fld.shape == x.shape
            msk = ~np.isnan(fld)
            v = np.array(fld[msk])
            vbound = np.maximum(np.minimum(v, vmax), vmin)

            if isinstance(cmap, str):
                cmap = colormaps[cmap]
            cmap = np.array([cmap(x)[0:3] for x in np.linspace(0, 1, nlevels+1)])

            cind = ((vbound - vmin) / dv).astype(int)
            ax.scatter(x[msk], y[msk], markersize, color=cmap[cind], **kwargs)

        self.set_xylim(ax)

    def plot_land(self, ax, color=None, linecolor='k', linewidth=1,
                  showriver=False, rivercolor='c',
                  showgrid=True, dlon=20, dlat=5):
        """
        Shows the map (coastline, rivers, lakes) and lon/lat grid for reference

        Args:
            ax (matplotlib.pyplot.Axes): Axes handle for plotting
            color (matplotlib color, optional):
                Face color of the landmass polygon, default is None (transparent).
            linecolor (matplotlib color, optional):
                Line color of the coastline, default is 'k' (black).
            linewidth (float, optional):
                Line width of the coastline, default is 1.
            showriver (bool, optional):
                If True, show the rivers and lakes over the landmass. Default is False.
            rivercolor (matplotlib color, optional):
                Color of the rivers and lakes, default is 'c' (cyan).
            showgrid (bool, optional):
                If True (default), show the reference lat/lon grid.
            dlon (float, optional):
                The interval of longitude lines in the reference grid. Default is 20 degrees.
            dlat (float, optional):
                The interval in latitude lines in the reference grid. Default is 5 degress.
        """
        ###plot the coastline to indicate land area
        if color is not None:
            draw_patch(ax, self.land_data, color=color, zorder=3)
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
