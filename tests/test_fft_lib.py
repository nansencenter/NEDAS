import numpy as np
import unittest
from NEDAS.utils.fft_lib import fftwn, get_wn, fft2, ifft2


class TestFftwn(unittest.TestCase):

    def test_first_element_is_zero(self):
        for n in [4, 5, 8, 16]:
            self.assertEqual(fftwn(n)[0], 0)

    def test_length_equals_n(self):
        for n in [4, 5, 8, 16]:
            self.assertEqual(len(fftwn(n)), n)

    def test_even_n_symmetric_negative(self):
        wn = fftwn(8)
        # For n=8: [0,1,2,3,4,-3,-2,-1]  (Nyquist at index 4 is +n/2)
        self.assertEqual(wn[1], 1)
        self.assertEqual(wn[-1], -1)

    def test_odd_n(self):
        wn = fftwn(5)
        # [0,1,2,-2,-1]
        self.assertEqual(wn[1], 1)
        self.assertEqual(wn[-1], -1)

    def test_covers_positive_and_negative(self):
        for n in [6, 7, 8, 9]:
            wn = fftwn(n)
            self.assertTrue(any(w > 0 for w in wn))
            self.assertTrue(any(w < 0 for w in wn))


class TestGetWn(unittest.TestCase):

    def test_output_shapes_match_input(self):
        fld = np.zeros((8, 12))
        wnx, wny = get_wn(fld)
        self.assertEqual(wnx.shape, fld.shape)
        self.assertEqual(wny.shape, fld.shape)

    def test_zero_wavenumber_at_origin(self):
        fld = np.zeros((8, 8))
        wnx, wny = get_wn(fld)
        self.assertEqual(wnx[0, 0], 0.0)
        self.assertEqual(wny[0, 0], 0.0)

    def test_wnx_constant_along_rows(self):
        # same column → same wnx value
        fld = np.zeros((8, 8))
        wnx, _ = get_wn(fld)
        for col in range(fld.shape[1]):
            vals = wnx[:, col]
            np.testing.assert_array_equal(vals, vals[0])

    def test_wny_constant_along_columns(self):
        fld = np.zeros((8, 8))
        _, wny = get_wn(fld)
        for row in range(fld.shape[0]):
            vals = wny[row, :]
            np.testing.assert_array_equal(vals, vals[0])

    def test_3d_input_shape(self):
        fld = np.zeros((3, 8, 8))
        wnx, wny = get_wn(fld)
        self.assertEqual(wnx.shape, fld.shape)


class TestFft2Ifft2Roundtrip(unittest.TestCase):

    def _random_field(self, ny=16, nx=16, seed=42):
        return np.random.default_rng(seed).standard_normal((ny, nx)).astype(np.float32)

    def test_roundtrip_recovers_field(self):
        fld = self._random_field()
        fld_rec = ifft2(fft2(fld))
        np.testing.assert_allclose(fld_rec, fld, atol=1e-4)

    def test_roundtrip_3d(self):
        fld = self._random_field(ny=8, nx=8).reshape(1, 8, 8)
        fld_rec = ifft2(fft2(fld))
        np.testing.assert_allclose(fld_rec, fld, atol=1e-4)

    def test_fft2_output_shape_matches_input(self):
        fld = self._random_field()
        fh = fft2(fld)
        self.assertEqual(fh.shape, fld.shape)

    def test_constant_field_spectrum_peak_at_zero(self):
        # A constant field's energy should be concentrated at wavenumber 0
        fld = np.ones((8, 8), dtype=np.float32)
        fh = fft2(fld)
        mag = np.abs(fh)
        self.assertGreater(mag[0, 0], mag[1, 1])


if __name__ == '__main__':
    unittest.main()
