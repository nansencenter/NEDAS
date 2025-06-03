"""
This module contain classes to handle grid with various geometries and spatial structures.

Classes
-------
- :class:`Grid`: A factory class for 2D grids.
- :class:`GridBase`: Base class for handling 2D grids.
- :class:`RegularGrid`: Class for handling 2D regular grid.
- :class:`IrregularGrid`: Class for handling 2D irregular grid (triangular mesh).
- :class:`Grid1D`: Class for handling 1D grid.

Map projections
---------------
The projection is typically handled by a :class:`pyproj.Proj` object, see `list of supported projections <https://proj.org/en/9.2/operations/projections/index.html>`_.
However, if you are working with some unsupported grid geometries, you can also define a custom `proj` function.
What :class:`Grid` needs from :func:`proj` is the conversion between longitude,latitude and the x,y coordinates, and the inverse, i.e.

>>> x, y = proj(lon, lat)
>>> lon, lat = proj(x, y, inverse=True)

Coordinates
-----------
The coordinates `x,y` has the same dimensions as the 2D grid.

For a regular grid, when we have a scalar field indexed by `fld[j,i]`, its coordinate is `x[j,i], y[j,i]`. Note that for Python the rightmost dimension is the quickest (row index first), if you have data that is column index first `fld[i,j]` then you will need to transpose it before using `Grid`.

So far we only support regular grid x,y where all the `x[j,:]` and all the `y[:,i]` are identical, namely `x, y = np.meshgrid(xc, yc)` where `xc,yc` are the 1D coordinates arrays. If you have a rotated x,y plane, you need to let the projection take care of the rotation first.

Cyclic boundary condition is supported, by setting the `cyclic_dim` parameter you can tell `Grid` which dimension(s) should wrap around when the coordinates fall outside of the given range (done by `wrap_cyclic_xy()`)

The longitude,latitude (geodetic) grid is special since it is the only projection not on a flat surface. For this `+proj=longlat` projection we support longitude conventions both 0\~360 and -180\~180 as long as its unit is "degrees east", you shall set `cyclic_dim='x'` if x is your longitude dimension. The latitude has units "degrees north". We also support special treatment of the two poles (lat=-90, 90) at which rotation angle is not well defined. You shall specify which dimension contains the pole and which index in that dimension is the pole. For example, if your data is [lat=40\~90, long=0\~360], you can set `pole_dim='y'` and `pole_index=(-1)`.

For an unstructured mesh, the `x,y` coordinates are for the nodes (vertices of the triangles), the 2D field can be either defined on nodes or elements (triangle faces). If the later is true, `x_elem,y_elem` will be used in handling the field. Note that `x_elem` has different size with `x`.




Examples:
    Define a proj function, e.g. stereographic:

    >>> import pyproj
    >>> proj = pyproj.Proj('+proj=stere +lat_0=90 +lon_0=0')

    Create a regular mesh x, y coordinates, with 10 km resolution, in meter units:

    >>> import numpy as np
    >>> x, y = np.meshgrid(np.arange(-3e6, 3e6, 1e4), np.arange(-3e6, 3e6, 1e4))

    Create a grid instance:

    >>> from NEDAS.grid import Grid
    >>> grid = Grid(proj, x, y)

`self.nx`, `self.ny` are the grid dimension for regular grids.

`self.xmin`, `self.xmax`, `self.ymin`, `self.ymax` define the grid extent (bounding box) for regular grids.

`self.Lx`, `self.Ly` are the grid length in x and y directions.

`self.dx` is the grid spacing (resolution), for unstructured mesh it is the averaged edge length of the triangles. `self.dy` is also provided for regular grid, although it is typical that we use `dy`=`dx`.

`self.mfx`, `self.mfy` are the map factors in x and y directions. Due to map projection the actual length is distorted on the 2D grid plane (see [Tissot's indicatrix](https://en.wikipedia.org/wiki/Tissot%27s_indicatrix) for an illustration). These map factors are useful in computing actual distances on Earth, using `self.dx / self.mfx` and `self.dy / self.mfy`

`self.tri` for unstructured mesh this is the triangle indices, computed by `matplotlib.tri.Triangulation`

`self.x_elem`, `self.y_elem` are the center coordinates for each triangular element.

Typically we use the `Grid` class for conversion of a 2D field from one grid to another, which involves interpolation, rotation of vectors and coarse-graining if the two grid has different resolution. Additionally, we also provide basic tools for visualizing a field on the map. See `tutorials/grid_convert` for some illustrated examples.

Once you created a `grid1` object on which a 2D field `fld1` is defined, and a `grid2` object for the destination grid. You convert `fld1` by first setting

`grid1.set_destination_grid(grid2)`, then

`fld2 = grid1.convert(fld1)`

It is recommended that you initialize a `grid` object, set the destination grid by `set_destination_grid`, then apply the same `grid.convert` to many fields needing conversion. Since the interpolation and rotation matrices are computed by `set_destination_grid` only once, the following application of `grid.convert` can be very efficient.


### 2.1 Interpolation

Two methods are supported for now. `method=linear` does bi-linear interpolation for the regular grids and barycentric interpolation for the triangular meshes. `method=nearest` performs nearest neighbor interpolation.

If you just want to do interpolation, you can also call `grid.interp(fld, x, y, method)` directly, where `fld` is the original field, `x,y` are the target coordinates (can be just one number, or an array, multi-dimension arrays will be flattened), `method` is the interpolation method.

If you only want the indices of given `x,y` coordinates, you can also use `grid.find_index(x, y)`. But note that we return more than just the indices, also some other information. Five things will be returned by `find_index`:

* `inside`: True if the point x,y is inside the grid

* `indices`: `None` for regular grid; triangle index which the `x,y` position falls in.

* `vertices`: the indices of rectangle(4) or triangle(3) vertices the `x,y` position falls in.

* `in_coords`: the internal coordinates of the `x,y` point within the rectangle (`in_x`, `in_y`) or triangle (`in1`, `in2`, `in3`)

* `nearest`: index in `x.flatten()`, `y.flatten()` for the nearest grid point to the given `x,y` position.


### 2.2 Rotation of Vectors

We support both scalar and vector fields. For a vector field, its u- and v- components are stacked in the first dimension, so that `vec_fld` has dimension [2, `ny`, `nx`]. Parameter `is_vector` = `True` will let `grid.convert` know it's operating on a vector field and need to rotate vectors if projection changes.

The rotation matrix is computed by displacing a small amount in x,y directions on each grid point, and project both the original point and displaced point to the new grid, then measuring the displacement and figure out the rotation.

If you just want to rotate vectors, you can apply `grid.rotate_vectors(vec_fld)`.


### 2.3 Coarse-graining

Setting `coarse_grain=True` in `grid.convert` will perform an additional coarse graining (by `grid.coarsen`) during interpolation. When the destination grid has lower resolution than the source grid, more points will fall in the same destination grid element. The coarse-graining will average them to represent the field on the low-resolution grid, instead of interpolating only the nearest source grid points. For a field with small-scale variability on the source grid, the interpolation to low-resolution grid will be sensitive to those sub-grid-scale variability, causing the interpolated value to have representation errors. Taking the average in the low-resolution grid will remedy this issue.

However, if you are dealing with a sparse observing network where you want to interpolate in the source grid to find the observed values, you don't need coarse graining, because observation should pick up sub-grid scale variability in its measurement anyway. Remember to set `coarse_grain=False` if this is the case.


### 2.4 Visualization

To avoid intricate dependencies from installing the `cartopy` package, we instead provide some internal solutions for basic plotting.

`ax` is a `matplotlib.pyplot.Axes` object for plotting.

Scalar fields can be plotted using `grid.plot_field(ax, fld, vmin, vmax, cmap)`, we use `pyplot.pcolor` for regular grids and `pyplot.tripcolor` for unstructured meshes. `vmin`, `vmax` are minimum and maximum values shown for the field. `cmap` is the color map.

Vector fields can be plotted using `grid.plot_vectors(ax, vec_fld, V, L, spacing, num_steps, linecolor, linewidth, showred, ref_xy, refcolor, showhead, headwidth, headlength)`. This replaces the `pyplot.quiver` and provides more control over how a vector plot looks. `vec_fld` has a dimension [2, ...] for the two components. `V` is the velocity scale that is the typical velocity value in this field (same units as `vec_fld`), `L` is the length scale that is the length (in `x,y` units) for vectors with velocity `V`. If you don't provide `V` and `L`, a standard value will be used based on the data provided, which is often good enough. `spacing` controls the density of vectors in the plot, they are spaced with interval `spacing * L`. `num_steps`=1 displays straight vectors (as in `quiver`); >1 lets you display curved trajectories, at each sub-step the velocity is re-interpolated at the new position along the trajectories. `linecolor, linewith` controls the style of vector lines. `show_ref=True` displays the reference vector at position `ref_xy=(x,y)` and with background color `refcolor`. `showhead=True` displays the vector heads with `headwidth, headlength` controlling their styles.


To show the map overlay, you can use `grid.plot_land(ax, color, linecolor, linewidth, showriver, rivercolor, showgrid, dlon, dlat)`. There are several shape files downloaded from [www.naturalearthdata.com](https://www.naturalearthdata.com) are stored in `grid` directory, where the `plot_land` reads and gather `land_data`, `river_data` and `lake_data`. They data are cached after first time use, so making several plots in a role will be more efficient. `color` controls the face color of the landmass, `linecolor` and `linewidth` controls the style of the coastlines. `showriver=True` will plot the rivers and lakes with `rivercolor`. `showgrid=True` turns on the latitude longitude grid at `dlon, dlat` intervals (degrees). The x,y axes will show the coordinate tick marks in its units.

"""

from typing import Optional, Any
import numpy as np
from NEDAS.grid.grid_regular import RegularGrid
from NEDAS.grid.grid_irregular import IrregularGrid

class Grid:
    """
    Factory class for creating a 2D grid object.

    This class serves as a factory for creating either a RegularGrid or an IrregularGrid instance
    based provided parameters. It allows for easy instantiation of different grid types
    without needing to directly interact with the specific grid classes.

    Examples:
        The factory class provides two classmethods to help creating a grid instance more easily.
        To create a regular grid with coordinates x from 0 to 100 and y from 0 to 100, resolution dx=1:

        >>> from NEDAS.grid import Grid
        >>> grid = Grid.regular_grid(None, 0, 100, 0, 100, 1)

        To create a irregular grid with same range of coordinates, in total 2000 grid points randomly placed:

        >>> grid = Grid.random_grid(None, 0, 100, 0, 100, 2000)
    """
    def __new__(cls, proj, x, y, regular=True, bounds=None, cyclic_dim=None, pole_dim=None, pole_index=None,
                distance_type='cartesian', triangles=None, neighbors=None, dst_grid=None):
        if regular:
            return RegularGrid(proj, x, y, bounds, cyclic_dim, distance_type,
                               pole_dim, pole_index, neighbors, dst_grid)
        else:
            return IrregularGrid(proj, x, y, bounds, cyclic_dim, distance_type,
                                 triangles, dst_grid)

    @classmethod
    def regular_grid(cls, proj: Any,
                     xstart: float, xend: float, ystart: float, yend: float,
                     dx: float, centered: bool=False, **kwargs) -> RegularGrid:
        """
        Create a regular grid within specified boundaries.

        Parameters:
            proj (Any): Projection from lon,lat to x,y.
            xstart (float): Lower bound for X coordinates.
            xend (float): Upper bound for X coordinates.
            ystart (float): Lower bound for Y coordinates.
            yend (float): Upper bound for Y coordinates.
            dx (float): Grid spacing.
            centered (bool): Optional, toggle for grid points to be on vertices (False) or in the middle of each grid box (True).  Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            RegularGrid: A Grid object representing the regular grid.
        """
        dx = float(dx)
        xcoord = np.arange(xstart, xend, dx)
        ycoord = np.arange(ystart, yend, dx)
        x, y = np.meshgrid(xcoord, ycoord)
        if centered:
            x += 0.5*dx  ##move coords to center of grid box
            y += 0.5*dx
        return RegularGrid(proj, x, y, **kwargs)

    @classmethod
    def random_grid(cls, proj: Any,
                    xstart: float, xend: float, ystart: float, yend: float,
                    npoints: int, min_dist: Optional[float]=None, **kwargs) -> IrregularGrid:
        """
        Create a grid with randomly positioned points within specified boundaries.

        Parameters:
            proj (pyproj.Proj): Projection from lon,lat to x,y.
            xstart (float): Lower bound for X coordinates.
            xend (float): Upper bound for X coordinates.
            ystart (float): Lower bound for Y coordinates.
            yend (float): Upper bound for Y coordinates.
            npoints (int): Number of grid points.
            min_dist (float): Optional, minimal distance allowed between each pair of grid points.
            **kwargs: Additional keyword arguments.

        Returns:
            IrregularGrid: A Grid object representing the randomly positioned grid.
        """
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
        return IrregularGrid(proj, x, y, bounds=bounds, **kwargs)
