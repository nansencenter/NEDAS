import numpy as np
from NEDAS.grid.grid_regular import RegularGrid
from NEDAS.grid.grid_irregular import IrregularGrid

class Grid:
    """
    Factory class for creating grid objects.

    This class serves as a factory for creating either a RegularGrid or an IrregularGrid
    based on the provided parameters. It allows for easy instantiation of different grid types
    without needing to directly interact with the specific grid classes.

    Shared parameters:
        proj (pyproj.Proj): Projection from lon,lat to x,y.
        x (array-like): X-coordinates of the grid points.
        y (array-like): Y-coordinates of the grid points.
        regular (bool): If True, creates a RegularGrid; if False, creates an IrregularGrid.
        bounds (list): Optional, boundaries of the grid in the x and y directions.
        cyclic_dim (int): Optional, dimension of the grid that is cyclic.
        distance_type (str): Optional, type of distance calculation ('cartesian' or 'great_circle').
        dst_grid (Grid): Optional, destination grid for interpolation.

    Parameters for RegularGrid:
        pole_dim (str): Optional, dimension of the grid that is a pole ('x' or 'y').
        pole_index (list): Optional, indices of the poles.
        neighbors (array-like): Optional, neighbor indices for the grid.

    Parameters for IrregularGrid:
        triangles (array-like): Optional, triangle connectivity for the grid.
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
    def regular_grid(cls, proj, xstart, xend, ystart, yend, dx, centered=False, **kwargs):
        """
        Create a regular grid within specified boundaries.

        Parameters:
            proj (pyproj.Proj): Projection from lon,lat to x,y.
            xstart, xend, ystart, yend (float): Boundaries of the grid in the x and y directions.
            dx (float): Resolution of the grid.
            centered (bool): Optional, toggle for grid points to be on vertices (False) or in the middle of each grid box (True).  Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            A Grid object representing the regular grid.
        """
        dx = float(dx)
        xcoord = np.arange(xstart, xend, dx)
        ycoord = np.arange(ystart, yend, dx)
        x, y = np.meshgrid(xcoord, ycoord)
        if centered:
            x += 0.5*dx  ##move coords to center of grid box
            y += 0.5*dx
        return cls.__new__(cls, proj, x, y, regular=True, **kwargs)

    @classmethod
    def random_grid(cls, proj, xstart, xend, ystart, yend, npoints, min_dist=None, **kwargs):
        """
        Create a grid with randomly positioned points within specified boundaries.

        Parameters:
            proj (pyproj.Proj): Projection from lon,lat to x,y.
            xstart, xend, ystart, yend (float): Boundaries of the grid in the x and y directions.
            npoints (int): Number of grid points.
            min_dist (float): Optional, minimal distance allowed between each pair of grid points.
            **kwargs: Additional keyword arguments.

        Returns:
            A Grid object representing the randomly positioned grid.
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
        return cls.__new__(cls, proj, x, y, regular=False, bounds=bounds, **kwargs)
