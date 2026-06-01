import unittest
from NEDAS.core import Context
from NEDAS.assim_tools.inflation import get_inflation_func
from NEDAS.assim_tools.inflation.multiplicative import MultiplicativeInflation
from NEDAS.assim_tools.inflation.RTPP import RTPPInflation


class TestGetInflationFunc(unittest.TestCase):

    def _context_with(self, inflation_def):
        c = Context()
        c.config.inflation_def = inflation_def
        return c

    def test_multiplicative_returns_correct_type(self):
        c = self._context_with({'type': 'multiplicative,prior', 'coef': 1.2})
        self.assertIsInstance(get_inflation_func(c), MultiplicativeInflation)

    def test_RTPP_returns_correct_type(self):
        c = self._context_with({'type': 'RTPP,post'})
        self.assertIsInstance(get_inflation_func(c), RTPPInflation)

    def test_coef_stored_on_instance(self):
        c = self._context_with({'type': 'multiplicative,prior', 'coef': 1.5})
        infl = get_inflation_func(c)
        self.assertAlmostEqual(infl.coef, 1.5)

    def test_adaptive_flag_propagated(self):
        c = self._context_with({'type': 'multiplicative,prior', 'adaptive': True})
        self.assertTrue(get_inflation_func(c).adaptive)

    def test_prior_flag_set(self):
        c = self._context_with({'type': 'multiplicative,prior'})
        infl = get_inflation_func(c)
        self.assertTrue(infl.prior)
        self.assertFalse(infl.post)

    def test_post_flag_set(self):
        c = self._context_with({'type': 'multiplicative,post'})
        infl = get_inflation_func(c)
        self.assertFalse(infl.prior)
        self.assertTrue(infl.post)

    def test_missing_type_key_raises_key_error(self):
        c = self._context_with({'coef': 1.0})
        with self.assertRaises(KeyError):
            get_inflation_func(c)

    def test_missing_inflation_def_raises_attribute_error(self):
        c = Context()
        del c.config.inflation_def
        with self.assertRaises(AttributeError):
            get_inflation_func(c)

    def test_unknown_type_raises_runtime_error(self):
        c = self._context_with({'type': 'nonexistent_inflator'})
        with self.assertRaises(RuntimeError):
            get_inflation_func(c)


if __name__ == '__main__':
    unittest.main()
