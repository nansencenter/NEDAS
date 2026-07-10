import numpy as np
import unittest
from NEDAS.utils.multiscale import (
    lowpass_response,
    get_scale_component_spec_bandpass,
    get_remaining_scale_component_spec_bandpass,
)


class TestLowpassResponse(unittest.TestCase):

    def test_below_k1_is_one(self):
        k2d = np.array([0.0, 0.5, 1.0])
        r = lowpass_response(k2d, k1=2.0, k2=4.0)
        np.testing.assert_array_equal(r, 1.0)

    def test_above_k2_is_zero(self):
        k2d = np.array([5.0, 6.0, 10.0])
        r = lowpass_response(k2d, k1=2.0, k2=4.0)
        np.testing.assert_array_equal(r, 0.0)

    def test_transition_between_zero_and_one(self):
        k2d = np.linspace(2.0, 4.0, 50)
        r = lowpass_response(k2d, k1=2.0, k2=4.0)
        self.assertTrue(np.all(r >= 0.0))
        self.assertTrue(np.all(r <= 1.0))

    def test_transition_is_monotone_decreasing(self):
        k2d = np.linspace(2.0, 4.0, 20)
        r = lowpass_response(k2d, k1=2.0, k2=4.0)
        self.assertTrue(np.all(np.diff(r) <= 0))

    def test_at_k1_response_is_one(self):
        k2d = np.array([2.0])
        r = lowpass_response(k2d, k1=2.0, k2=4.0)
        np.testing.assert_allclose(r, 1.0)

    def test_at_k2_response_is_zero(self):
        k2d = np.array([4.0])
        r = lowpass_response(k2d, k1=2.0, k2=4.0)
        np.testing.assert_allclose(r, 0.0, atol=1e-15)

    def test_2d_input(self):
        k2d = np.array([[0.0, 3.0], [6.0, 1.0]])
        r = lowpass_response(k2d, k1=2.0, k2=4.0)
        self.assertEqual(r.shape, (2, 2))
        self.assertEqual(r[0, 0], 1.0)  # k=0 < k1
        self.assertEqual(r[1, 0], 0.0)  # k=6 > k2


class TestGetScaleComponentSpecBandpass(unittest.TestCase):

    def _make_grid(self, ny=32, nx=32, L=1000e3):
        """Minimal duck-typed grid for get_scale_component_spec_bandpass."""
        class FakeGrid:
            regular = True
            Lx = L
            Ly = L
        return FakeGrid()

    def test_nscale_one_returns_original(self):
        grid = self._make_grid()
        fld = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
        result = get_scale_component_spec_bandpass(grid, fld, [500e3], s=0)
        np.testing.assert_array_equal(result, fld)

    def test_two_scale_components_sum_to_original(self):
        grid = self._make_grid(ny=32, nx=32, L=1000e3)
        rng = np.random.default_rng(7)
        fld = rng.standard_normal((32, 32)).astype(np.float32)
        character_length = [800e3, 200e3]
        comp0 = get_scale_component_spec_bandpass(grid, fld, character_length, s=0)
        comp1 = get_scale_component_spec_bandpass(grid, fld, character_length, s=1)
        np.testing.assert_allclose(comp0 + comp1, fld, atol=1e-4)

    def test_three_scale_components_sum_to_original(self):
        grid = self._make_grid()
        rng = np.random.default_rng(3)
        fld = rng.standard_normal((32, 32)).astype(np.float32)
        cl = [800e3, 400e3, 100e3]
        total = sum(
            get_scale_component_spec_bandpass(grid, fld, cl, s=s)
            for s in range(3)
        )
        np.testing.assert_allclose(total, fld, atol=1e-4)

    def test_output_shape_matches_input(self):
        grid = self._make_grid()
        fld = np.zeros((32, 32), dtype=np.float32)
        comp = get_scale_component_spec_bandpass(grid, fld, [800e3, 200e3], s=0)
        self.assertEqual(comp.shape, fld.shape)


class TestGetRemainingScaleComponentSpecBandpass(unittest.TestCase):

    def _make_grid(self, ny=32, nx=32, L=1000e3):
        class FakeGrid:
            regular = True
            Lx = L
            Ly = L
        return FakeGrid()

    def test_last_scale_has_no_remainder(self):
        grid = self._make_grid()
        rng = np.random.default_rng(1)
        fld = rng.standard_normal((32, 32)).astype(np.float32)
        cl = [800e3, 400e3, 100e3]
        remainder = get_remaining_scale_component_spec_bandpass(grid, fld, cl, s=2)
        np.testing.assert_array_equal(remainder, np.zeros_like(fld))

    def test_remainder_plus_processed_bands_equals_original(self):
        """bands 0..s plus the remainder after s must reconstruct the full field,
        matching the frozen/remaining split used in AlignmentUpdator.update_files."""
        grid = self._make_grid()
        rng = np.random.default_rng(11)
        fld = rng.standard_normal((32, 32)).astype(np.float32)
        cl = [800e3, 400e3, 100e3]
        for s in range(len(cl)):
            processed = sum(
                get_scale_component_spec_bandpass(grid, fld, cl, s=j)
                for j in range(s + 1)
            )
            remainder = get_remaining_scale_component_spec_bandpass(grid, fld, cl, s=s)
            np.testing.assert_allclose(processed + remainder, fld, atol=1e-4)

    def test_remainder_equals_sum_of_later_bands(self):
        grid = self._make_grid()
        rng = np.random.default_rng(23)
        fld = rng.standard_normal((32, 32)).astype(np.float32)
        cl = [800e3, 400e3, 100e3]
        s = 0
        later_bands = sum(
            get_scale_component_spec_bandpass(grid, fld, cl, s=j)
            for j in range(s + 1, len(cl))
        )
        remainder = get_remaining_scale_component_spec_bandpass(grid, fld, cl, s=s)
        np.testing.assert_allclose(remainder, later_bands, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
