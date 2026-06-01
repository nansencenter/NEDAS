import numpy as np
import unittest
from NEDAS.assim_tools.localization.distance_based import (
    gaspari_cohn_func, step_func, exponential_func,
)
from NEDAS.assim_tools.localization import get_localization_funcs, get_localization_func_component
from NEDAS.core import Context


class TestGaspariCohnFunc(unittest.TestCase):
    def test_zero_distance_is_one(self):
        result = gaspari_cohn_func(np.array([0.0]), roi=1000.0)
        self.assertAlmostEqual(float(result[0]), 1.0, places=5)

    def test_at_roi_is_zero(self):
        roi = 500.0
        result = gaspari_cohn_func(np.array([roi]), roi=roi)
        self.assertAlmostEqual(float(result[0]), 0.0, places=10)

    def test_beyond_roi_is_zero(self):
        roi = 500.0
        result = gaspari_cohn_func(np.array([600.0, 1000.0, 1e6]), roi=roi)
        np.testing.assert_array_equal(result, 0.0)

    def test_monotone_decrease(self):
        roi = 1000.0
        dist = np.linspace(0, roi, 50)
        result = gaspari_cohn_func(dist, roi=roi)
        self.assertTrue(np.all(np.diff(result) <= 1e-12))

    def test_between_zero_and_one(self):
        roi = 1000.0
        dist = np.linspace(0, roi, 100)
        result = gaspari_cohn_func(dist, roi=roi)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0 + 1e-12))

    def test_output_shape_preserved(self):
        result = gaspari_cohn_func(np.ones((3, 4)) * 100.0, roi=1000.0)
        self.assertEqual(result.shape, (3, 4))


class TestStepFunc(unittest.TestCase):
    def test_inside_roi_is_one(self):
        result = step_func(np.array([0.0, 50.0, 99.9]), roi=100.0)
        np.testing.assert_array_equal(result, 1.0)

    def test_at_roi_boundary_is_one(self):
        result = step_func(np.array([100.0]), roi=100.0)
        self.assertEqual(float(result[0]), 1.0)

    def test_outside_roi_is_zero(self):
        result = step_func(np.array([100.1, 200.0, 1e9]), roi=100.0)
        np.testing.assert_array_equal(result, 0.0)

    def test_output_shape_preserved(self):
        result = step_func(np.ones((5, 6)) * 50.0, roi=100.0)
        self.assertEqual(result.shape, (5, 6))


class TestExponentialFunc(unittest.TestCase):
    def test_zero_distance_is_one(self):
        result = exponential_func(np.array([0.0]), roi=100.0)
        self.assertAlmostEqual(float(result[0]), 1.0)

    def test_at_roi_is_exp_minus_one(self):
        roi = 200.0
        result = exponential_func(np.array([roi]), roi=roi)
        self.assertAlmostEqual(float(result[0]), np.exp(-1.0), places=10)

    def test_decay_formula(self):
        roi = 300.0
        dist = np.array([0.0, 100.0, 300.0, 600.0])
        np.testing.assert_allclose(exponential_func(dist, roi=roi), np.exp(-dist / roi), rtol=1e-12)

    def test_always_non_negative(self):
        # exp(-d/roi) >= 0; underflows to 0.0 for very large d in float64
        dist = np.linspace(0, 1e6, 100)
        self.assertTrue(np.all(exponential_func(dist, roi=1000.0) >= 0))

    def test_output_shape_preserved(self):
        result = exponential_func(np.ones((2, 3, 4)) * 10.0, roi=100.0)
        self.assertEqual(result.shape, (2, 3, 4))


class TestGetLocalizationFuncs(unittest.TestCase):
    def _context_with(self, htype=None, vtype=None, ttype=None):
        c = Context()
        c.config.localization_def = {
            'horizontal': {'type': htype} if htype else None,
            'vertical':   {'type': vtype} if vtype else None,
            'temporal':   {'type': ttype} if ttype else None,
        }
        return c

    def test_returns_dict_with_three_keys(self):
        funcs = get_localization_funcs(self._context_with(htype='gaspari_cohn'))
        for key in ('horizontal', 'vertical', 'temporal'):
            self.assertIn(key, funcs)

    def test_none_entry_returns_none(self):
        funcs = get_localization_funcs(self._context_with(htype='step'))
        self.assertIsNone(funcs['vertical'])
        self.assertIsNone(funcs['temporal'])

    def test_callable_returned_for_valid_type(self):
        for ltype in ('gaspari_cohn', 'step', 'exponential'):
            with self.subTest(ltype=ltype):
                func = get_localization_func_component(ltype)
                self.assertTrue(callable(func))

    def test_missing_type_key_raises(self):
        c = Context()
        c.config.localization_def = {'horizontal': {'roi': 100}, 'vertical': None, 'temporal': None}
        with self.assertRaises(KeyError):
            get_localization_funcs(c)

    def test_unknown_type_raises(self):
        with self.assertRaises((ValueError, UnboundLocalError)):
            get_localization_func_component('no_such_method')


if __name__ == '__main__':
    unittest.main()
