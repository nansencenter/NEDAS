import numpy as np
import unittest
from NEDAS.utils.njit import njit
from NEDAS.assim_tools.assimilators.QCEF.core import (
    epanechnikov_kernel, epanechnikov_cdf,
    gauss_quad, get_kde_bandwidths, get_kde_params,
    kde_pdf, kde_cdf, obs_increment_qcef,
)

# gauss_quad is @njit, so test callables must also be @njit
@njit
def _const3(x): return np.ones_like(x) * 3.0

@njit
def _linear(x): return x

@njit
def _quadratic(x): return x**2

@njit
def _zero_width(x): return x**2


class TestEpanechnikovKernel(unittest.TestCase):

    def test_max_at_zero(self):
        self.assertAlmostEqual(float(epanechnikov_kernel(np.array([0.0]))[0]), 0.75)

    def test_zero_at_boundaries(self):
        self.assertAlmostEqual(float(epanechnikov_kernel(np.array([1.0]))[0]), 0.0)
        self.assertAlmostEqual(float(epanechnikov_kernel(np.array([-1.0]))[0]), 0.0)

    def test_zero_outside_support(self):
        x = np.array([1.5, -2.0, 10.0])
        np.testing.assert_array_equal(epanechnikov_kernel(x), 0.0)

    def test_non_negative(self):
        x = np.linspace(-1, 1, 200)
        self.assertTrue(np.all(epanechnikov_kernel(x) >= 0.0))

    def test_symmetric(self):
        x = np.linspace(0, 1, 50)
        np.testing.assert_allclose(epanechnikov_kernel(x), epanechnikov_kernel(-x), rtol=1e-12)


class TestEpanechnikovCDF(unittest.TestCase):

    def test_zero_at_minus_one(self):
        self.assertAlmostEqual(float(epanechnikov_cdf(np.array([-1.0]))[0]), 0.0)

    def test_one_at_plus_one(self):
        self.assertAlmostEqual(float(epanechnikov_cdf(np.array([1.0]))[0]), 1.0)

    def test_half_at_zero(self):
        self.assertAlmostEqual(float(epanechnikov_cdf(np.array([0.0]))[0]), 0.5)

    def test_monotone_increasing(self):
        x = np.linspace(-1.5, 1.5, 100)
        vals = epanechnikov_cdf(x)
        self.assertTrue(np.all(np.diff(vals) >= -1e-12))

    def test_clamps_outside_support(self):
        self.assertAlmostEqual(float(epanechnikov_cdf(np.array([-5.0]))[0]), 0.0)
        self.assertAlmostEqual(float(epanechnikov_cdf(np.array([5.0]))[0]), 1.0)


class TestGaussQuad(unittest.TestCase):
    # gauss_quad is @njit — callables must be @njit too (defined at module level above)

    def test_constant_function(self):
        result = gauss_quad(0.0, 2.0, _const3)
        self.assertAlmostEqual(result, 6.0, places=10)

    def test_linear_function(self):
        # integral of x from 0 to 1 = 0.5
        result = gauss_quad(0.0, 1.0, _linear)
        self.assertAlmostEqual(result, 0.5, places=10)

    def test_quadratic_function(self):
        # integral of x^2 from 0 to 1 = 1/3, exact for 5th-order Gauss-Legendre
        result = gauss_quad(0.0, 1.0, _quadratic)
        self.assertAlmostEqual(result, 1.0 / 3.0, places=12)

    def test_zero_width_interval(self):
        result = gauss_quad(1.0, 1.0, _zero_width)
        self.assertAlmostEqual(result, 0.0, places=12)


class TestGetKDEBandwidths(unittest.TestCase):

    def test_returns_array_of_correct_length(self):
        obs_prior = np.random.default_rng(0).normal(0, 1, 20).astype(np.float64)
        bw = get_kde_bandwidths(obs_prior)
        self.assertEqual(len(bw), 20)

    def test_all_positive(self):
        obs_prior = np.random.default_rng(1).normal(0, 2, 30).astype(np.float64)
        bw = get_kde_bandwidths(obs_prior)
        self.assertTrue(np.all(bw > 0))

    def test_wider_prior_gives_larger_bandwidths(self):
        rng = np.random.default_rng(2)
        narrow = np.sort(rng.normal(0, 0.1, 20)).astype(np.float64)
        wide   = np.sort(rng.normal(0, 10.0, 20)).astype(np.float64)
        self.assertGreater(np.mean(get_kde_bandwidths(wide)),
                           np.mean(get_kde_bandwidths(narrow)))


# BUG: get_kde_params, kde_pdf, kde_cdf are marked @njit but use Python dicts
# and lambdas that capture non-constant variables — numba rejects this.
# These tests are skipped until the @njit decorators are removed from those functions.
class TestKDEPrior(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.obs_prior = rng.normal(0, 1, 20).astype(np.float64)
        self.params = get_kde_params(self.obs_prior, 0.0, np.inf)

    def test_pdf_non_negative(self):
        xs = np.linspace(-5, 5, 100)
        vals = np.array([float(kde_pdf(x, self.params)) for x in xs])
        self.assertTrue(np.all(vals >= -1e-12))

    def test_cdf_far_left_near_zero(self):
        self.assertAlmostEqual(float(kde_cdf(-100.0, self.params)), 0.0, places=4)

    def test_cdf_far_right_near_one(self):
        self.assertAlmostEqual(float(kde_cdf(100.0, self.params)), 1.0, places=4)

    def test_cdf_monotone(self):
        xs = np.linspace(-5, 5, 50)
        vals = np.array([float(kde_cdf(x, self.params)) for x in xs])
        self.assertTrue(np.all(np.diff(vals) >= -1e-8))

class TestObsIncrementQCEF(unittest.TestCase):

    def test_zero_increment_when_all_members_equal(self):
        obs_prior = np.ones(20) * 3.0
        incr = obs_increment_qcef(obs_prior, obs=3.0, obs_err=1.0)
        np.testing.assert_array_equal(incr, 0.0)

    def test_output_length_matches_ensemble_size(self):
        obs_prior = np.random.default_rng(0).normal(0, 1, 20).astype(np.float64)
        incr = obs_increment_qcef(obs_prior, obs=0.5, obs_err=1.0)
        self.assertEqual(len(incr), 20)

    def test_quantile_mapping_property(self):
        """Posterior CDF at each updated member equals prior CDF at the original member."""
        rng = np.random.default_rng(5)
        nens = 20
        obs_prior = np.sort(rng.normal(0, 1, nens)).astype(np.float64)
        obs, obs_err = 1.0, 0.5
        incr = obs_increment_qcef(obs_prior, obs=obs, obs_err=obs_err)
        obs_post = obs_prior + incr

        params_prior = get_kde_params(obs_prior, 0.0, np.inf)
        params_post  = get_kde_params(obs_prior, obs, obs_err)
        for i in range(nens):
            u_prior = kde_cdf(obs_prior[i], params_prior)
            u_post  = kde_cdf(obs_post[i],  params_post)
            self.assertAlmostEqual(float(u_prior), float(u_post), places=3)


# ---------------------------------------------------------------- the QCEFF framework (DART filter_assim in Python)

import copy
from types import SimpleNamespace
from scipy.special import gammainc
from NEDAS.core import Context
from NEDAS.utils.parallel import DummyComm
from NEDAS.assim_tools.localization.distance_based import gaspari_cohn_func
from NEDAS.assim_tools.assimilators.QCEF import probit
from NEDAS.assim_tools.assimilators.QCEF.core import (
    QCEFAssimilator, obs_increment_eakf, obs_increment_enkf, obs_increment_gamma,
    obs_increment_rank_histogram, obs_increment_bounded_norm_rhf, inv_bnrh_cdf_like)
from NEDAS.assim_tools.assimilators.EAKF.core import EAKFAssimilator
import NEDAS.assim_tools.assimilators.EAKF.core as eakf

BOUNDS = {
    'NORMAL_DISTRIBUTION': (False, False, 0.0, 0.0),
    'LOG_NORMAL_DISTRIBUTION': (False, False, 0.0, 0.0),
    'UNIFORM_DISTRIBUTION': (True, True, -1.0, 20.0),
    'GAMMA_DISTRIBUTION': (True, False, 0.0, 0.0),
    'BETA_DISTRIBUTION': (True, True, 0.0, 1.0),
    'BOUNDED_NORMAL_RH_DISTRIBUTION': (True, False, 0.0, 0.0),
    'KDE_DISTRIBUTION': (True, False, 0.0, 0.0),
}


class TestProbitTransforms(unittest.TestCase):

    def ensemble(self, dist):
        rng = np.random.default_rng(3)
        if dist == 'BETA_DISTRIBUTION':
            return rng.beta(2, 3, (6, 20))
        ens = rng.gamma(2.0, 1.0, (6, 20)) + (0.1 if dist == 'LOG_NORMAL_DISTRIBUTION' else 0.0)
        if dist in ('BOUNDED_NORMAL_RH_DISTRIBUTION', 'KDE_DISTRIBUTION'):
            ens[1, :4] = 0.0        # members on the bound
            ens[2, 3:6] = ens[2, 2]  # tied members
        return ens

    def test_round_trip(self):
        for dist, bounds in BOUNDS.items():
            with self.subTest(dist=dist):
                ens = self.ensemble(dist)
                y, p = probit.to_probit(ens, dist, *bounds)
                self.assertTrue(np.all(np.isfinite(y)))
                np.testing.assert_allclose(probit.from_probit(y, p), ens, atol=1e-10)
                # the fitted transform applied again gives the same probit values
                np.testing.assert_allclose(probit.probit_given(ens, p), y, atol=1e-10)

    def test_no_spread_is_not_transformed(self):
        for dist in ('BOUNDED_NORMAL_RH_DISTRIBUTION', 'KDE_DISTRIBUTION', 'GAMMA_DISTRIBUTION'):
            y, p = probit.to_probit(np.full((1, 10), 2.0), dist, *BOUNDS[dist])
            self.assertFalse(p['ok'][0], dist)
            np.testing.assert_array_equal(y, 2.0)

    def test_bnrh_quantiles(self):
        # n distinct members sit at the quantiles 1/(n+1) ... n/(n+1); tied ones share the middle of their run
        ens = np.array([[3.0, 1.0, 2.0, 5.0, 4.0]])
        y, _ = probit.to_probit(ens, 'BOUNDED_NORMAL_RH_DISTRIBUTION')
        np.testing.assert_allclose(probit.ndtr(y[0]), np.array([3, 1, 2, 5, 4]) / 6)
        ens = np.array([[1.0, 2.0, 2.0, 3.0]])
        y, _ = probit.to_probit(ens, 'BOUNDED_NORMAL_RH_DISTRIBUTION')
        np.testing.assert_allclose(probit.ndtr(y[0]), [0.2, 0.5, 0.5, 0.8])

    def test_kde_cdf_is_a_cdf(self):
        rng = np.random.default_rng(4)
        for bounded in (False, True):
            ens = np.abs(rng.normal(0.3, 1.0, 30))
            k = probit.kde_fit(ens, bounded, False, 0.0, 0.0)
            x = np.linspace(k['edges'][0] - 1, k['edges'][-1] + 1, 400)
            q = probit.kde_cdf(x, k)
            self.assertAlmostEqual(q[0], 0.0)
            self.assertAlmostEqual(q[-1], 1.0, places=10)
            self.assertTrue(np.all(np.diff(q) >= -1e-12))
            u = np.linspace(0.01, 0.99, 50)
            np.testing.assert_allclose(probit.kde_cdf(probit.inv_kde_cdf(u, k), k), u, atol=1e-12)


class TestObsIncrements(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(5)
        self.ens = self.rng.gamma(2.0, 1.0, 40)
        self.mean, self.var = self.ens.mean(), self.ens.var(ddof=1)

    def test_eakf_is_the_native_eakf(self):
        incr = obs_increment_eakf(self.ens, self.mean, self.var, 1.5, 0.3)
        native = eakf.obs_increment_eakf(self.ens, np.zeros(0), 1.5, np.sqrt(0.3), 1.0, 0.0, False)
        np.testing.assert_allclose(incr, native, atol=1e-12)

    def test_enkf_posterior_mean(self):
        incr = obs_increment_enkf(self.ens, self.var, 1.5, 0.3, np.random.default_rng(0))
        eakf_mean = (self.ens + obs_increment_eakf(self.ens, self.mean, self.var, 1.5, 0.3)).mean()
        self.assertAlmostEqual((self.ens + incr).mean(), eakf_mean, places=12)

    def test_enkf_sorted_increments_keep_the_ranks(self):
        incr = obs_increment_enkf(self.ens, self.var, 1.5, 0.3, np.random.default_rng(0), sort_obs_inc=True)
        np.testing.assert_array_equal(np.argsort(self.ens + incr), np.argsort(self.ens))

    def test_gamma_conserves_quantiles(self):
        obs, obs_var = 1.5, 0.3
        post = self.ens + obs_increment_gamma(self.ens, self.mean, self.var, obs, obs_var)
        ps, pc = self.mean**2 / self.var, self.var / self.mean
        ls, lc = obs**2 / obs_var, obs_var / obs
        qs, qc = ps + ls - 1, pc * lc / (pc + lc)
        np.testing.assert_allclose(gammainc(qs, post / qc), gammainc(ps, self.ens / pc), atol=1e-12)

    def test_rank_histogram_flat_likelihood_changes_nothing(self):
        for tails in (False, True):
            incr = obs_increment_rank_histogram(self.ens, self.var, 1.5, 1e12, gaussian_likelihood_tails=tails)
            np.testing.assert_allclose(incr, 0.0, atol=1e-6)

    def test_rank_histogram_moves_toward_the_obs(self):
        incr = obs_increment_rank_histogram(self.ens, self.var, 5.0, 0.5)
        self.assertGreater((self.ens + incr).mean(), self.mean)
        self.assertLess((self.ens + incr).var(ddof=1), self.var)

    def test_bnrh_flat_likelihood_changes_nothing(self):
        # DART's null test of inv_bnrh_cdf_like
        p = probit.bnrh_fit(self.ens[None], True, False, 0.0, 0.0)
        q = probit.bnrh_ens_quantiles(self.ens[None], p['sort'], True, False, 0.0, 0.0)[0]
        post = inv_bnrh_cdf_like(q, probit.rows(p, 0), np.ones(self.ens.size))
        np.testing.assert_allclose(post, self.ens, atol=1e-10)

    def test_bnrh_stays_within_the_bound(self):
        incr = obs_increment_bounded_norm_rhf(self.ens, 0.0, 0.5, True, False, 0.0, 0.0)
        self.assertTrue(np.all(self.ens + incr >= 0.0))
        self.assertLess((self.ens + incr).mean(), self.mean)


def fake_context(ens, obs_points, obs, err_std, hroi=8.0):
    """The parts of a Context the serial loop reaches for: one field on a line of points, identity obs"""
    nens, nloc = ens.shape
    x = np.arange(nloc, dtype=float)
    state_data = {
        'x': x, 'y': np.zeros(nloc), 'z': np.zeros((1, nloc)), 't': np.zeros(1),
        'var_id': np.zeros(1, dtype=int), 'field_ids': [(0, None)],
        'state_prior': ens[:, None, :].copy(), 'state_static': np.zeros((0, 1, nloc)),
    }
    nobs = len(obs_points)
    obs_data = {
        'obs': np.asarray(obs, dtype=float), 'x': x[obs_points], 'y': np.zeros(nobs), 'z': np.zeros(nobs),
        't': np.zeros(nobs), 'err_std': np.full(nobs, err_std),
        'obs_prior': ens[:, obs_points].copy(), 'obs_prior_static': np.zeros((0, nobs)),
        'used': np.zeros(nobs, dtype=bool), 'hroi': np.array([hroi]), 'vroi': np.array([1.0]),
        'troi': np.array([1.0]), 'impact_on_variable': np.ones((1, 1)), 'obs_impact': np.ones(nobs),
        'obs_rec_id': np.zeros(nobs, dtype=int), 'obs_comp': np.full(nobs, -1),
    }
    out = {}
    record = SimpleNamespace(name='u', is_vector=False)
    state = SimpleNamespace(state_prior={}, state_z=None, state_static=None, info=SimpleNamespace(fields={0: record}),
                            pack_local_state_data=lambda *a: copy.deepcopy(state_data),
                            unpack_local_state_data=lambda c, par_id, post, data: out.update(state=data['state_prior'][:, 0].copy()))
    obsp = SimpleNamespace(lobs={}, lobs_prior={}, lobs_prior_static={}, valid={},
                           info=SimpleNamespace(records={0: record}),
                           pack_local_obs_data=lambda *a: copy.deepcopy(obs_data),
                           global_obs_list=lambda c, valid: [(0, None, 0, i) for i in range(nobs)],
                           unpack_local_obs_data=lambda c, par_id, lobs, post, data: out.update(obs=data['obs_prior'].copy()))
    grid = SimpleNamespace(distance=lambda x0, x, y0, y, p=2: np.hypot(np.asarray(x) - x0, np.asarray(y) - y0))
    c = SimpleNamespace(state=state, obs=obsp, grid=grid, pid_mem=0, comm=DummyComm(), comm_mem=DummyComm(),
                        localization_funcs={'horizontal': gaspari_cohn_func, 'vertical': gaspari_cohn_func,
                                            'temporal': gaspari_cohn_func, 'correlation': None}, message='', debug_message='',
                        current_task=0, total_tasks=0,
                        covariance=SimpleNamespace(beta=0.0, static_var_scaling=1.0, hybrid_perturbation=False))
    return c, out


def analysis(cls, assimilator_def, *args, **kwargs):
    ctx = Context()
    ctx.config.assimilator_def = dict(assimilator_def)
    a = cls(ctx)
    c, out = fake_context(*args, **kwargs)
    a.assimilation_algorithm(c)
    return out


class TestQCEFAnalysis(unittest.TestCase):
    """The serial loop with QCEF, on a line of points observed at a few of them"""

    def setUp(self):
        rng = np.random.default_rng(6)
        nloc = 30
        base = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(nloc) / nloc)
        self.ens = base * rng.gamma(2.0, 0.5, (20, 1)) + 0.1 * rng.normal(size=(20, nloc))**2
        self.points = [5, 12, 20]

    def test_defaults_are_the_eakf(self):
        args = (self.ens, self.points, [0.3, 0.8, 1.2], 0.3)
        q = analysis(QCEFAssimilator, {'type': 'QCEF'}, *args)
        e = analysis(EAKFAssimilator, {'type': 'EAKF'}, *args)
        np.testing.assert_allclose(q['state'], e['state'], atol=1e-12)
        np.testing.assert_allclose(q['obs'], e['obs'], atol=1e-12)

    def test_bounded_variable_stays_within_its_bound(self):
        # obs near the bound pull the EAKF below 0; the BNRH filter and transform keep the members above
        args = (self.ens, self.points, [0.0, 0.0, 0.0], 0.05)
        e = analysis(EAKFAssimilator, {'type': 'EAKF'}, *args)
        self.assertLess(e['state'].min(), 0.0)
        opt = {'filter_kind': 'BOUNDED_NORMAL_RHF', 'dist': 'BOUNDED_NORMAL_RH_DISTRIBUTION',
               'bounded_below': True, 'lower_bound': 0.0}
        q = analysis(QCEFAssimilator, {'type': 'QCEF', 'qceff': {'u': opt}}, *args)
        self.assertGreaterEqual(q['state'].min(), 0.0)
        self.assertLess(q['state'][:, self.points].mean(), self.ens[:, self.points].mean())

    def test_kde_runs_and_reduces_spread_at_the_obs(self):
        args = (self.ens, self.points, [0.3, 0.8, 1.2], 0.3)
        opt = {'filter_kind': 'KDE_FILTER', 'dist': 'KDE_DISTRIBUTION', 'bounded_below': True, 'lower_bound': 0.0}
        q = analysis(QCEFAssimilator, {'type': 'QCEF', 'qceff': {'u': opt}}, *args)
        self.assertTrue(np.all(np.isfinite(q['state'])))
        self.assertTrue(np.all(q['state'][:, self.points].std(axis=0) < self.ens[:, self.points].std(axis=0)))


if __name__ == '__main__':
    unittest.main()
