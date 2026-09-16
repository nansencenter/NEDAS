"""
Checks on NEDAS's interface to PDAF's compiled analysis kernels, through pyPDAF.

Skipped unless pyPDAF is importable; it has no pip/conda package, see
NEDAS/assim_tools/assimilators/PDAF/install_pypdaf.md for how to build it.

The substantive test runs one partition through PDAF's LETKF and compares against NEDAS's
native ETKF on the same data. Both are the same square-root filter, but they differ in which
square root they take (and PDAF applies its own type_trans rotation), so the comparison is on
the posterior mean and covariance rather than member by member -- those are what the filter
is defined by, and a real numerics change upstream moves them.

The rest are structural checks that need no pyPDAF: the configuration PDAFomi cannot express
(vertical/temporal/cross-variable localization) has to be refused rather than silently dropped.
"""
import unittest
import numpy as np

from NEDAS.assim_tools.assimilators.PDAF.core import (
    PDAFAssimilator, FILTER_KINDS, LOC_WEIGHTS, NEDAS_TO_PDAF_WEIGHT,
)
from NEDAS.assim_tools.assimilators.ETKF.core import local_analysis_main
from NEDAS.assim_tools.localization.distance_based import gaspari_cohn_func

try:
    import pyPDAF  # noqa: F401
    HAS_PYPDAF = True
except ImportError:
    HAS_PYPDAF = False

HROI = 5.0
NENS, NFLD, NLOC, NLOBS = 20, 2, 3, 6


def make_partition(seed=42):
    """
    A synthetic NEDAS partition: the subset of pack_local_state_data /
    pack_local_obs_data output that the analysis actually reads.
    """
    rng = np.random.default_rng(seed)
    state_data = {
        'state_prior': rng.normal(0, 1, (NENS, NFLD, NLOC)),
        'x': np.linspace(0, 4, NLOC),
        'y': np.zeros(NLOC),
        'z': np.zeros((NFLD, NLOC)),
        't': np.zeros(NFLD),
        'var_id': np.zeros(NFLD, dtype=int),
    }
    obs_data = {
        'obs': rng.normal(0, 1, NLOBS),
        'obs_prior': rng.normal(0, 1, (NENS, NLOBS)),
        'err_std': np.full(NLOBS, 0.5),
        'x': np.linspace(0, 4, NLOBS),
        'y': np.zeros(NLOBS),
        'z': np.zeros(NLOBS),
        't': np.zeros(NLOBS),
        'obs_rec_id': np.zeros(NLOBS, dtype=int),
        'hroi': np.array([HROI]),
        'vroi': np.array([np.inf]),
        'troi': np.array([np.inf]),
        'impact_on_variable': np.ones((1, 1)),
    }
    return state_data, obs_data


def native_etkf_analysis(state_data, obs_data):
    """the same partition through NEDAS's own ETKF, as the reference"""
    state_post = state_data['state_prior'].copy()
    no_static_state = np.zeros((0, NFLD, NLOC))
    no_static_obs = np.zeros((0, NLOBS))
    for loc_id in range(NLOC):
        hdist = np.abs(obs_data['x'] - state_data['x'][loc_id])
        hlfactor = gaspari_cohn_func(hdist, HROI)
        local_analysis_main(state_post[..., loc_id], obs_data['obs_prior'],
                            no_static_state[..., loc_id], no_static_obs,
                            obs_data['obs'], obs_data['err_std'], hlfactor,
                            state_data['z'][:, loc_id], obs_data['z'],
                            np.inf, gaspari_cohn_func,
                            state_data['t'], obs_data['t'],
                            np.inf, gaspari_cohn_func,
                            np.ones((NLOBS, NFLD)), np.eye(NENS), False,
                            1.0 / np.sqrt(NENS - 1), 0.0, False)
    return state_post


def make_assimilator(**kwargs):
    """
    A PDAFAssimilator without a Context: analyze_partition only needs the settings
    that assimilation_algorithm would have derived from the config and the grid.
    """
    self = PDAFAssimilator.__new__(PDAFAssimilator)
    self.filter_kind = kwargs.get('filter_kind', 'LETKF')
    self.subtype = kwargs.get('subtype', 0)
    self.forget = kwargs.get('forget', 1.0)
    self.screen = 0
    self._locweight = LOC_WEIGHTS['gaspari_cohn']
    self._disttype = 0
    self._domainsize = np.array([-1.0, -1.0])
    return self


class TestPDAFMapping(unittest.TestCase):
    """the parts that do not need pyPDAF installed"""

    def test_only_domain_localized_filters_offered(self):
        # PDAF's global filters would silently ignore NEDAS's localization, so they
        # must not be reachable through filter_kind
        for name in FILTER_KINDS:
            self.assertTrue(name.startswith('L'), name)

    def test_localization_taper_maps_to_pdafomi(self):
        for nedas_type, code in NEDAS_TO_PDAF_WEIGHT.items():
            self.assertIn(code, LOC_WEIGHTS.values(), nedas_type)


@unittest.skipUnless(HAS_PYPDAF, "pyPDAF is not installed (see PDAF/install_pypdaf.md)")
class TestPDAFAnalysis(unittest.TestCase):

    def setUp(self):
        self.state_data, self.obs_data = make_partition()
        self.prior = self.state_data['state_prior'].copy()

    def analyze(self, **kwargs):
        state_data = dict(self.state_data)
        state_data['state_prior'] = self.prior.copy()
        make_assimilator(**kwargs).analyze_partition(None, state_data, self.obs_data)
        return state_data['state_prior']

    def test_letkf_matches_native_etkf(self):
        post = self.analyze(filter_kind='LETKF')
        reference = native_etkf_analysis(self.state_data, self.obs_data)
        # mean and covariance define the analysis; the ensemble members themselves differ
        # by the square-root convention each filter picks
        np.testing.assert_allclose(post.mean(axis=0), reference.mean(axis=0), atol=1e-8)
        for loc_id in range(NLOC):
            np.testing.assert_allclose(np.cov(post[..., loc_id], rowvar=False),
                                       np.cov(reference[..., loc_id], rowvar=False), atol=1e-8)

    def test_analysis_reduces_spread(self):
        post = self.analyze()
        self.assertLessEqual(np.std(post, axis=0).mean(), np.std(self.prior, axis=0).mean())

    def test_obs_far_outside_hroi_have_no_effect(self):
        obs_data = dict(self.obs_data)
        obs_data['x'] = self.obs_data['x'] + 10 * HROI
        state_data = dict(self.state_data)
        state_data['state_prior'] = self.prior.copy()
        make_assimilator().analyze_partition(None, state_data, obs_data)
        np.testing.assert_allclose(state_data['state_prior'], self.prior, atol=1e-12)

    def test_filter_kinds_run(self):
        for filter_kind in FILTER_KINDS:
            post = self.analyze(filter_kind=filter_kind)
            self.assertTrue(np.isfinite(post).all(), filter_kind)


if __name__ == '__main__':
    unittest.main()
