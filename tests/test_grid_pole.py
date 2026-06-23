"""
Tests for _set_rotation_matrix pole singularity fix (issue #17).

At lat=±90 in a longlat grid, moving a small step in longitude leaves the
geographic position unchanged in any conformal projection, so the rotation
matrix is undefined (0/0 → NaN).  The fix replaces those entries with the
rotation computed at the reference point (lon=0, lat=±(90-delta)), following
the lon=0-meridian convention described in issue #17.
"""
import numpy as np
import unittest
from pyproj import Proj
from NEDAS.grid import Grid


def _longlat_to_polar_stere():
    """Shared grid pair: longlat source with pole row, polar-stere destination."""
    grid_ll = Grid.regular_grid(
        Proj('+proj=longlat'), -180, 181, 0, 91, 1, cyclic_dim='x'
    )
    grid_ps = Grid.regular_grid(
        Proj('+proj=stere +lat_0=90'), -1e6, 1e6, -1e6, 1e6, 1e4, centered=True
    )
    return grid_ll, grid_ps


class TestRotationMatrixAtPole(unittest.TestCase):

    def test_rotation_matrix_no_nan_at_north_pole(self):
        """rotate_matrix must be finite on the lat=90 row after fix."""
        grid_ll, grid_ps = _longlat_to_polar_stere()
        grid_ll.set_destination_grid(grid_ps)
        # last row of grid_ll corresponds to lat=90 (North Pole)
        pole_row = grid_ll.rotate_matrix[:, -1, :]
        self.assertFalse(
            np.isnan(pole_row).any(),
            "rotation matrix still has NaN at lat=90 — pole singularity not fixed"
        )

    def test_converted_vector_no_nan_without_pole_dim(self):
        """Vector conversion must not produce NaN even without pole_dim/pole_index."""
        grid_ll, grid_ps = _longlat_to_polar_stere()
        grid_ll.set_destination_grid(grid_ps)

        u = np.ones(grid_ll.x.shape)
        v = np.zeros(grid_ll.x.shape)
        vfld_out = grid_ll.convert(np.array([u, v]), is_vector=True)

        self.assertFalse(
            np.isnan(vfld_out).any(),
            "NaN in converted vector field — pole singularity not handled"
        )

    def test_pole_rotation_consistent_with_nearby_point(self):
        """Rotation at the pole should match the reference point (lon=0, lat=89.999)."""
        grid_ll, grid_ps = _longlat_to_polar_stere()
        grid_ll.set_destination_grid(grid_ps)

        # The fix copies rotation from lon=0, lat=90-delta for ALL pole points.
        # Every column in the lat=90 row should therefore have identical entries.
        pole_row = grid_ll.rotate_matrix[:, -1, :]   # shape (4, 361)
        self.assertTrue(
            np.allclose(pole_row, pole_row[:, :1], atol=1e-10),
            "rotate_matrix at the pole is not uniform across longitudes — "
            "the lon=0-meridian convention was not applied correctly"
        )

    def test_south_pole_no_nan(self):
        """Same fix must apply to the South Pole."""
        grid_ll = Grid.regular_grid(
            Proj('+proj=longlat'), -180, 181, -90, 1, 1, cyclic_dim='x'
        )
        grid_ps = Grid.regular_grid(
            Proj('+proj=stere +lat_0=-90'), -1e6, 1e6, -1e6, 1e6, 1e4, centered=True
        )
        grid_ll.set_destination_grid(grid_ps)
        # first row corresponds to lat=-90 (South Pole)
        pole_row = grid_ll.rotate_matrix[:, 0, :]
        self.assertFalse(
            np.isnan(pole_row).any(),
            "rotation matrix has NaN at lat=-90 — South Pole not handled"
        )


class TestPoleDimBackwardCompat(unittest.TestCase):
    """The pole_dim/_fill_pole_void path still works after the fix."""

    def test_pole_dim_still_produces_no_nan(self):
        grid_ll, grid_ps = _longlat_to_polar_stere()
        grid_ll.pole_dim = 'y'
        grid_ll.pole_index = [-1]
        grid_ll.set_destination_grid(grid_ps)

        u = np.ones(grid_ll.x.shape)
        v = np.zeros(grid_ll.x.shape)
        vfld_out = grid_ll.convert(np.array([u, v]), is_vector=True)
        self.assertFalse(np.isnan(vfld_out).any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
