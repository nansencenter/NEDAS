import unittest
import numpy as np
from NEDAS.core.types import ErrorModel
from NEDAS.utils.obs_error import perturb_obs, assimilation_std


def model(type, std, **kw):
    return ErrorModel(type=type, std=std, hcorr=0., vcorr=0., tcorr=0., cross_corr=(), **kw)


class TestNormal(unittest.TestCase):
    def test_matches_the_formula_it_replaced(self):
        """Seeded, the normal model must give exactly truth + std*N(0,1), as before."""
        truth = np.linspace(-3, 3, 50)
        np.random.seed(7); expected = truth + np.random.normal(0, 1, truth.shape) * 0.4
        np.random.seed(7); got = perturb_obs(truth, model('normal', 0.4))
        np.testing.assert_array_equal(got, expected)

    def test_assimilation_std_is_constant_and_inflated(self):
        s = assimilation_std(np.arange(5.0), model('normal', 0.4, infl=2.0))
        np.testing.assert_allclose(s, 0.8)

    def test_assimilation_std_is_per_location_for_vector_obs(self):
        s = assimilation_std(np.ones((2, 5)), model('normal', 0.4))
        self.assertEqual(s.shape, (5,))


class TestLognormal(unittest.TestCase):
    def test_always_positive_even_where_the_truth_is_zero(self):
        truth = np.zeros(2000)
        obs = perturb_obs(truth, model('lognormal', 0.5, floor=1e-3))
        self.assertGreater(obs.min(), 0.0)

    def test_log_error_has_the_requested_std_and_no_bias_in_log_space(self):
        np.random.seed(0)
        truth = np.full(200000, 2e-3)
        logerr = np.log(perturb_obs(truth, model('lognormal', 0.3)) / truth)
        self.assertAlmostEqual(logerr.mean(), 0.0, places=2)
        self.assertAlmostEqual(logerr.std(), 0.3, places=2)

    def test_floor_applies_only_below_it(self):
        np.random.seed(1)
        truth = np.array([0.0, 1e-4, 5e-3])
        obs = perturb_obs(np.repeat(truth, 50000), model('lognormal', 0.2, floor=1e-3)).reshape(3, -1)
        medians = np.median(obs, axis=1)
        np.testing.assert_allclose(medians, [1e-3, 1e-3, 5e-3], rtol=0.02)

    def test_assimilation_std_is_relative_to_each_observation(self):
        obs = np.array([1e-3, 2e-3, 4e-3])
        s = assimilation_std(obs, model('lognormal', 0.25, infl=1.5))
        np.testing.assert_allclose(s, 0.25 * obs * 1.5)
        self.assertTrue(np.all(s > 0))

    def test_negative_floor_is_refused(self):
        with self.assertRaises(ValueError):
            perturb_obs(np.ones(3), model('lognormal', 0.2, floor=-1.0))


class TestUnknownType(unittest.TestCase):
    def test_refused_rather_than_silently_treated_as_normal(self):
        with self.assertRaisesRegex(ValueError, 'unsupported observation error type'):
            perturb_obs(np.ones(3), model('student-t', 1.0))
        with self.assertRaisesRegex(ValueError, 'unsupported observation error type'):
            assimilation_std(np.ones(3), model('student-t', 1.0))


if __name__ == '__main__':
    unittest.main()
