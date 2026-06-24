import numpy as np
import unittest
from NEDAS.utils.random_perturb import random_field_gaussian, random_field_powerlaw


class TestRandomFieldGaussian(unittest.TestCase):

    def test_output_shape(self):
        fld = random_field_gaussian(nx=16, ny=12, amp=1.0, hcorr=4)
        self.assertEqual(fld.shape, (12, 16))

    def test_amplitude_approx_correct(self):
        # With enough repetitions, ensemble std should be close to amp
        amp = 2.0
        samples = [random_field_gaussian(nx=32, ny=32, amp=amp, hcorr=8)
                   for _ in range(20)]
        std_all = np.std(samples)
        self.assertAlmostEqual(std_all, amp, delta=0.5)

    def test_zero_mean(self):
        # Mean over many realisations should be near zero
        samples = np.array([random_field_gaussian(nx=16, ny=16, amp=1.0, hcorr=4)
                            for _ in range(50)])
        np.testing.assert_allclose(samples.mean(), 0.0, atol=0.2)

    def test_no_nan_or_inf(self):
        fld = random_field_gaussian(nx=32, ny=32, amp=1.0, hcorr=8)
        self.assertFalse(np.any(np.isnan(fld)))
        self.assertFalse(np.any(np.isinf(fld)))

    def test_short_hcorr_vs_long_hcorr_spectrum(self):
        # A field with longer hcorr should have more energy at large scales
        nx, ny = 32, 32
        fld_short = random_field_gaussian(nx, ny, amp=1.0, hcorr=2)
        fld_long  = random_field_gaussian(nx, ny, amp=1.0, hcorr=12)

        from NEDAS.utils.fft_lib import fft2
        import numpy as np
        spec_short = np.abs(fft2(fld_short.astype(np.float32)))**2
        spec_long  = np.abs(fft2(fld_long.astype(np.float32)))**2

        # energy at wavenumber 1 (large scale) relative to total energy
        ratio_short = spec_short[0, 1] / spec_short.sum()
        ratio_long  = spec_long[0, 1]  / spec_long.sum()
        self.assertGreater(ratio_long, ratio_short)


class TestRandomFieldPowerlaw(unittest.TestCase):

    def test_output_shape(self):
        fld = random_field_powerlaw(nx=16, ny=12, amp=1.0, pwrlaw=-3)
        self.assertEqual(fld.shape, (12, 16))

    def test_no_nan_or_inf(self):
        fld = random_field_powerlaw(nx=32, ny=32, amp=1.0, pwrlaw=-5)
        self.assertFalse(np.any(np.isnan(fld)))
        self.assertFalse(np.any(np.isinf(fld)))

    def test_zero_mean_component(self):
        # wavenumber-0 amplitude is set to zero, so mean should be 0
        fld = random_field_powerlaw(nx=32, ny=32, amp=1.0, pwrlaw=-3)
        np.testing.assert_allclose(fld.mean(), 0.0, atol=0.1)


if __name__ == '__main__':
    unittest.main()
