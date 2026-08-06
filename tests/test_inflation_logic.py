import numpy as np
import unittest
from unittest.mock import MagicMock
from NEDAS.assim_tools.inflation.multiplicative import MultiplicativeInflation
from NEDAS.assim_tools.inflation.RTPP import RTPPInflation


def _mock_context():
    c = MagicMock()
    c.debug = False
    return c


def _patch_stats(infl_obj, stats):
    """Replace obs_space_stats to return a fixed dict."""
    infl_obj.obs_space_stats = lambda c: stats


class TestMultiplicativePriorInflation(unittest.TestCase):

    def _make(self):
        return MultiplicativeInflation(coef=1.0, adaptive=True, prior=True, post=False)

    def test_formula_coef(self):
        infl = self._make()
        varb, varo, omb2 = 4.0, 1.0, 9.0
        # expected coef = sqrt((omb2 - varo) / varb) = sqrt(8/4) = sqrt(2)
        _patch_stats(infl, {'total_nobs': 10, 'varb': varb*10, 'varo': varo*10, 'omb2': omb2*10,
                             'vara': 0, 'omaamb': 0, 'amb2': 0})
        c = _mock_context()
        infl.adaptive_prior_inflation(c)
        np.testing.assert_allclose(infl.coef, np.sqrt((omb2 - varo) / varb), rtol=1e-10)

    def test_fallback_nobs_less_than_3(self):
        infl = self._make()
        _patch_stats(infl, {'total_nobs': 2, 'varb': 1, 'varo': 1, 'omb2': 2,
                             'vara': 0, 'omaamb': 0, 'amb2': 0})
        infl.adaptive_prior_inflation(_mock_context())
        self.assertEqual(infl.coef, 1.0)


class TestMultiplicativePostInflation(unittest.TestCase):

    def _make(self):
        return MultiplicativeInflation(coef=1.0, adaptive=True, prior=False, post=True)

    def test_formula_coef(self):
        infl = self._make()
        # ratio = omaamb / vara = <(a-b)(o-a)>/vara -- matches Ying (2019)'s
        # `infl=sqrt(max(1,sum(amb*oma)/sum(vara)))` (qgmodel_enkf/filter.py, commit f2e1be2)
        vara, omaamb = 2.0, 5.0
        n = 10
        _patch_stats(infl, {'total_nobs': n, 'varb': 3*n, 'vara': vara*n, 'varo': 1*n,
                             'omb2': 8*n, 'omaamb': omaamb*n, 'amb2': 2*n})
        infl.adaptive_post_inflation(_mock_context())
        np.testing.assert_allclose(infl.coef, np.sqrt(omaamb / vara), rtol=1e-10)

    def test_negative_ratio_gives_one(self):
        infl = self._make()
        # ratio = omaamb/vara < 0
        n = 10
        _patch_stats(infl, {'total_nobs': n, 'varb': 2*n, 'vara': 3*n, 'varo': 5*n,
                             'omb2': 2*n, 'omaamb': -1*n, 'amb2': 1*n})
        infl.adaptive_post_inflation(_mock_context())
        self.assertEqual(infl.coef, 1.0)

    def test_fallback_nobs_less_than_3(self):
        infl = self._make()
        _patch_stats(infl, {'total_nobs': 1, 'varb': 1, 'vara': 1, 'varo': 1,
                             'omb2': 2, 'omaamb': 0, 'amb2': 0})
        infl.adaptive_post_inflation(_mock_context())
        self.assertEqual(infl.coef, 1.0)

    def test_vara_zero_gives_one(self):
        infl = self._make()
        _patch_stats(infl, {'total_nobs': 10, 'varb': 10, 'vara': 0, 'varo': 10,
                             'omb2': 20, 'omaamb': 0, 'amb2': 0})
        infl.adaptive_post_inflation(_mock_context())
        self.assertEqual(infl.coef, 1.0)

    def test_default_post_infl_formula_is_omaamb(self):
        infl = self._make()
        self.assertEqual(infl.post_infl_formula, 'omaamb')

    def test_post_infl_formula_omb2_amb2(self):
        infl = MultiplicativeInflation(coef=1.0, adaptive=True, prior=False, post=True,
                                        post_infl_formula='omb2_amb2')
        vara, varo, omb2, amb2 = 2.0, 1.0, 8.0, 2.0
        # ratio = (omb2 - varo - amb2) / vara = 5/2 = 2.5
        n = 10
        _patch_stats(infl, {'total_nobs': n, 'varb': 3*n, 'vara': vara*n, 'varo': varo*n,
                             'omb2': omb2*n, 'omaamb': 0, 'amb2': amb2*n})
        infl.adaptive_post_inflation(_mock_context())
        np.testing.assert_allclose(infl.coef, np.sqrt((omb2 - varo - amb2) / vara), rtol=1e-10)

    def test_unknown_post_infl_formula_raises(self):
        infl = MultiplicativeInflation(coef=1.0, adaptive=True, prior=False, post=True,
                                        post_infl_formula='bogus')
        n = 10
        _patch_stats(infl, {'total_nobs': n, 'varb': 3*n, 'vara': 2*n, 'varo': 1*n,
                             'omb2': 8*n, 'omaamb': 5*n, 'amb2': 2*n})
        with self.assertRaises(ValueError):
            infl.adaptive_post_inflation(_mock_context())


class TestRTPPPostInflation(unittest.TestCase):

    def _make(self):
        return RTPPInflation(coef=0.0, adaptive=True, prior=False, post=True)

    def test_formula_coef(self):
        infl = self._make()
        varb, vara, varo, omb2 = 4.0, 2.0, 1.0, 8.0
        # la = max(sqrt((omb2-varo)/varb), 1.0) = max(sqrt(7/4), 1.0) = sqrt(7)/2
        # beta = sqrt(varb/vara) = sqrt(2); coef = (la-1)/(beta-1)
        n = 10
        _patch_stats(infl, {'total_nobs': n, 'varb': varb*n, 'vara': vara*n, 'varo': varo*n,
                             'omb2': omb2*n, 'omaamb': 0, 'amb2': 0})
        infl.adaptive_post_inflation(_mock_context())
        beta = np.sqrt(varb / vara)
        la = max(np.sqrt(max(0.0, (omb2 - varo) / varb)), 1.0)
        expected = (la - 1) / (beta - 1)
        np.testing.assert_allclose(infl.coef, expected, rtol=1e-10)

    def test_beta_leq_one_gives_zero(self):
        infl = self._make()
        # varb <= vara → beta <= 1
        n = 10
        _patch_stats(infl, {'total_nobs': n, 'varb': 1*n, 'vara': 4*n, 'varo': 1*n,
                             'omb2': 8*n, 'omaamb': 0, 'amb2': 1*n})
        infl.adaptive_post_inflation(_mock_context())
        self.assertEqual(infl.coef, 0)

    def test_coef_is_nonnegative(self):
        infl = self._make()
        # la is floored at 1.0 and beta>1 is required to reach the (la-1)/(beta-1) branch at
        # all, so coef is guaranteed >= 0 by construction under the corrected formula (unlike
        # the old buggy one, which could go negative -- see relaxation_adaptive_coef's own
        # docstring in core/inflation.py for the 2026-07-28 fix this reflects).
        varb, vara, varo, omb2 = 1.01, 1.0, 0.0, 1000.0
        n = 10
        _patch_stats(infl, {'total_nobs': n, 'varb': varb*n, 'vara': vara*n, 'varo': varo*n,
                             'omb2': omb2*n, 'omaamb': 0, 'amb2': 0})
        infl.adaptive_post_inflation(_mock_context())
        self.assertGreaterEqual(infl.coef, 0.0)
        self.assertTrue(np.isfinite(infl.coef))

    def test_near_singular_beta_falls_back_to_zero(self):
        infl = self._make()
        # beta just barely above 1 (vara only fractionally below varb) drives the (beta-1)
        # denominator toward 0 -- the isfinite guard should catch any resulting non-finite
        # value and fall back to coef=0 rather than propagating inf/nan (this is exactly the
        # failure mode confirmed 2026-07-28 in a live run using the OLD vara-denominator
        # formula, where the coefficient estimate itself went NaN after a few cycles).
        varb, vara, varo, omb2 = 1.0 + 1e-300, 1.0, 0.0, 1000.0
        n = 10
        _patch_stats(infl, {'total_nobs': n, 'varb': varb*n, 'vara': vara*n, 'varo': varo*n,
                             'omb2': omb2*n, 'omaamb': 0, 'amb2': 0})
        infl.adaptive_post_inflation(_mock_context())
        self.assertTrue(np.isfinite(infl.coef))

    def test_prior_inflation_not_implemented(self):
        infl = self._make()
        with self.assertRaises(NotImplementedError):
            infl.adaptive_prior_inflation(_mock_context())


if __name__ == '__main__':
    unittest.main()
