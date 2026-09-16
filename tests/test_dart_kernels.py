"""
Checks on NEDAS's interface to DART's compiled filter kernels.

Skipped unless libdartkernels.so has been built (see
NEDAS/assim_tools/assimilators/DART/build_dart_kernels.sh); point NEDAS_DART_LIB at
it if it is not next to the DART assimilator code.

Three kinds of test live here:

* EAKF is checked against NEDAS's own native EAKF. The two implement the same published
  algorithm by different routes, so they should agree to roundoff; a failure means
  upstream's numerics moved (or ours did), which is what this interface exists to catch.
* The other filter kinds have no native NEDAS counterpart, so they only get smoke tests --
  the increments are finite, they respond to the observation, and bounded kernels respect
  their bounds. These would not catch a subtle numerical change upstream.
* The seeding tests pin the behaviour the stochastic kernels depend on: seeding explicitly
  keeps them away from DART's own my_task_id()-based fallback, which would initialize DART's
  utilities and stop the run when no input.nml is present.

DART's kernels take the dynamic ensemble alone, so everything here compares against the
plain EAKF path with no static members (covariance_def.nens_static = 0).
"""
import os
import unittest
import numpy as np

from NEDAS.assim_tools.assimilators.DART.core import (
    DARTAssimilator, load_dart_kernels, default_lib_path, FILTER_KINDS, seed_from_time,
    REQUIRED_NML_SECTIONS, dart_error_message,
)
from NEDAS.assim_tools.assimilators.EAKF.core import (
    obs_increment_eakf, update_ensemble,
    update_local_state_linear, update_local_obs_linear,
)
from NEDAS.assim_tools.localization.distance_based import gaspari_cohn_func

LIB_PATH = os.environ.get('NEDAS_DART_LIB', default_lib_path())


def _no_static(ens):
    """an empty batch of static members matching the shape of ens (nens, ...)"""
    return np.zeros((0,) + ens.shape[1:])


def _plain_obs_increment(obs_prior, obs, obs_err):
    """plain EAKF: no static members, the dynamic ensemble covariance alone"""
    return obs_increment_eakf(obs_prior, np.zeros(0), obs, obs_err, 1.0, 0.0, False)


def _plain_update_ensemble(ens_prior, obs_prior, obs_incr, local_factor):
    """plain EAKF regression: no static members"""
    return update_ensemble(ens_prior, _no_static(ens_prior), obs_prior, np.zeros(0),
                           obs_incr, local_factor, 1.0, 0.0, False)


@unittest.skipUnless(os.path.exists(LIB_PATH), f"DART kernel library not built: {LIB_PATH}")
class DARTKernelTestCase(unittest.TestCase):
    """shared helpers for calling the raw C entry points"""

    @classmethod
    def setUpClass(cls):
        cls.lib = load_dart_kernels(LIB_PATH)

    def seed(self, value):
        self.lib.dart_set_random_seed(int(value))

    def dart_obs_increment(self, obs_prior, obs, obs_err, kind='EAKF',
                           bounded_below=False, bounded_above=False,
                           lower_bound=0.0, upper_bound=1.0):
        obs_prior = np.ascontiguousarray(obs_prior, dtype=np.float64)
        obs_inc = np.empty_like(obs_prior)
        net_a = np.zeros(1)
        status = self.lib.dart_obs_increment(FILTER_KINDS[kind], obs_prior.size, obs_prior,
                                             float(obs), float(obs_err)**2,
                                             int(bounded_below), int(bounded_above),
                                             float(lower_bound), float(upper_bound),
                                             obs_inc, net_a)
        return status, obs_inc, net_a[0]

    def dart_regress(self, ens, obs_prior, obs_inc, lfactor, net_a=0.0):
        ens = np.ascontiguousarray(ens, dtype=np.float64)
        flat = ens.reshape(ens.shape[0], -1)
        self.lib.dart_update_from_obs_inc(
            ens.shape[0], flat.shape[1],
            np.ascontiguousarray(obs_prior, dtype=np.float64),
            np.ascontiguousarray(obs_inc, dtype=np.float64),
            float(net_a), flat,
            np.ascontiguousarray(lfactor, dtype=np.float64).reshape(-1))
        return ens


class TestEAKFAgainstNative(DARTKernelTestCase):
    """the real regression check: DART's EAKF vs NEDAS's own"""

    def test_obs_increment_matches_native(self):
        rng = np.random.default_rng(42)
        for nens, obs, obs_err in ((20, 3.0, 1.0), (50, -2.5, 0.5), (100, 0.0, 2.0)):
            prior = rng.normal(1.0, 2.0, nens)
            status, dart_inc, _ = self.dart_obs_increment(prior, obs, obs_err)
            self.assertEqual(status, 0)
            np.testing.assert_allclose(dart_inc, _plain_obs_increment(prior, obs, obs_err),
                                       rtol=1e-13, atol=1e-13)

    def test_regression_matches_native(self):
        rng = np.random.default_rng(7)
        nens, nfld, nloc = 30, 2, 5
        obs_prior = rng.normal(0, 1, nens)
        obs_inc = _plain_obs_increment(obs_prior, 1.5, 0.8)
        ens = rng.normal(0, 1, (nens, nfld, nloc))
        lfactor = rng.uniform(0, 1, (nfld, nloc))

        native = _plain_update_ensemble(ens.copy(), obs_prior, obs_inc, lfactor)
        dart = self.dart_regress(ens.copy(), obs_prior, obs_inc, lfactor)
        np.testing.assert_allclose(dart, native, rtol=1e-12, atol=1e-12)


class TestKernelEdgeCases(DARTKernelTestCase):

    def test_obs_increment_zero_prior_spread(self):
        status, dart_inc, _ = self.dart_obs_increment(np.full(10, 5.0), 5.0, 1.0)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(dart_inc, 0.0)

    def test_obs_increment_degenerate_returns_status(self):
        # zero obs error variance and zero prior spread: DART calls error_handler here,
        # the wrapper must report instead of aborting the process
        status, _, _ = self.dart_obs_increment(np.full(10, 5.0), 5.0, 0.0)
        self.assertEqual(status, 1)

    def test_unknown_filter_kind_returns_status(self):
        prior = np.random.default_rng(0).normal(0, 1, 10)
        obs_inc = np.empty_like(prior)
        net_a = np.zeros(1)
        status = self.lib.dart_obs_increment(999, prior.size, prior, 1.0, 1.0,
                                             0, 0, 0.0, 1.0, obs_inc, net_a)
        self.assertEqual(status, 2)

    def test_regression_skips_zero_localization(self):
        rng = np.random.default_rng(11)
        nens, nloc = 20, 4
        obs_prior = rng.normal(0, 1, nens)
        obs_inc = _plain_obs_increment(obs_prior, 1.0, 1.0)
        ens = rng.normal(0, 1, (nens, nloc))
        dart = self.dart_regress(ens.copy(), obs_prior, obs_inc, np.zeros(nloc))
        np.testing.assert_array_equal(dart, ens)

    def test_regression_no_update_when_obs_prior_var_zero(self):
        ens = np.random.default_rng(3).normal(0, 1, (10, 3))
        dart = self.dart_regress(ens.copy(), np.full(10, 2.0), np.ones(10), np.ones(3))
        np.testing.assert_array_equal(dart, ens)


class TestRandomSeeding(DARTKernelTestCase):
    """
    The stochastic kernels draw from DART's module-level random sequence.

    Left to itself DART seeds that sequence with my_task_id() + 1 on first use, which
    initializes DART's utilities and reads input.nml -- stopping the run when the file is
    absent -- and yields the same stream in every process. Seeding explicitly avoids both.
    """

    def test_stochastic_kinds_run_after_seeding(self):
        # deliberately checks that no input.nml is needed: if one happens to be in the
        # working directory the test proves nothing, so skip rather than pass vacuously
        if os.path.exists('input.nml'):
            self.skipTest("an input.nml in the working directory makes this test vacuous")
        prior = np.random.default_rng(5).normal(5.0, 1.0, 80)
        for kind in ('ENKF', 'KERNEL'):
            with self.subTest(kind=kind):
                self.seed(12345)
                status, inc, _ = self.dart_obs_increment(prior, 0.0, 1.0, kind=kind)
                self.assertEqual(status, 0)
                self.assertTrue(np.all(np.isfinite(inc)))

    def test_same_seed_reproduces(self):
        prior = np.random.default_rng(6).normal(5.0, 1.0, 60)
        self.seed(4242)
        _, first, _ = self.dart_obs_increment(prior, 0.0, 1.0, kind='ENKF')
        self.seed(4242)
        _, second, _ = self.dart_obs_increment(prior, 0.0, 1.0, kind='ENKF')
        np.testing.assert_array_equal(first, second)

    def test_different_seed_differs(self):
        prior = np.random.default_rng(6).normal(5.0, 1.0, 60)
        self.seed(1)
        _, first, _ = self.dart_obs_increment(prior, 0.0, 1.0, kind='ENKF')
        self.seed(999)
        _, second, _ = self.dart_obs_increment(prior, 0.0, 1.0, kind='ENKF')
        self.assertFalse(np.allclose(first, second),
                         "a different seed must give different perturbed observations")

    def test_seed_from_time_varies_by_cycle(self):
        from datetime import datetime
        a = seed_from_time(datetime(2026, 1, 1, 0))
        b = seed_from_time(datetime(2026, 1, 1, 6))
        self.assertNotEqual(a, b)
        for s in (a, b):
            self.assertGreater(s, 0)
            self.assertLess(s, 2**31)
        # identical inputs must give identical seeds: every rank derives it independently
        self.assertEqual(a, seed_from_time(datetime(2026, 1, 1, 0)))


class TestOtherFilterKinds(DARTKernelTestCase):
    """
    Smoke tests only -- NEDAS has no native implementation of these to diff against, so
    these check that the kernels run and behave sanely, not that they are numerically right.
    """

    def test_unbounded_kinds_pull_posterior_toward_obs(self):
        rng = np.random.default_rng(5)
        prior = rng.normal(0.0, 1.0, 80) + 5.0     # prior mean ~5, obs at 0
        self.seed(20260915)                        # the stochastic kinds need a seed
        for kind in ('EAKF', 'ENKF', 'KERNEL', 'PARTICLE', 'RHF'):
            with self.subTest(kind=kind):
                status, inc, _ = self.dart_obs_increment(prior, 0.0, 1.0, kind=kind)
                self.assertEqual(status, 0)
                self.assertTrue(np.all(np.isfinite(inc)))
                post_mean = np.mean(prior + inc)
                self.assertLess(post_mean, np.mean(prior),
                                f"{kind}: posterior mean should move toward the obs")

    def test_gamma_kind_runs_on_positive_prior(self):
        # the gamma filter assumes a positive-definite quantity
        prior = np.random.default_rng(6).gamma(shape=4.0, scale=1.0, size=60)
        status, inc, _ = self.dart_obs_increment(prior, 2.0, 0.5, kind='GAMMA')
        self.assertEqual(status, 0)
        self.assertTrue(np.all(np.isfinite(inc)))

    def test_bounded_kind_respects_lower_bound(self):
        # KDE is deliberately excluded: it reads kde_nml, which needs DART's utilities
        # initialized, and aborts the process otherwise (see UNSUPPORTED_KINDS).
        rng = np.random.default_rng(8)
        prior = np.abs(rng.normal(0.5, 0.3, 60))     # non-negative quantity, e.g. concentration
        status, inc, _ = self.dart_obs_increment(
            prior, 0.05, 0.1, kind='BNRHF',
            bounded_below=True, bounded_above=False, lower_bound=0.0)
        self.assertEqual(status, 0)
        post = prior + inc
        self.assertTrue(np.all(np.isfinite(post)))
        self.assertGreaterEqual(post.min(), -1e-12, "BNRHF: posterior crossed the lower bound")


@unittest.skipUnless(os.path.exists(LIB_PATH), f"DART kernel library not built: {LIB_PATH}")
class TestDARTAssimilatorMethods(unittest.TestCase):
    """
    Exercise the assimilator methods rather than the raw entry points, so that the
    localization-factor assembly and the in-place update are covered too.

    Each case drives obs_increment() first, as the serial loop does, so net_a carries
    over to the regression exactly the way it does in a real analysis.
    """

    def setUp(self):
        # bypass Assimilator.__init__, which wants a full Context just to read config
        self.assim = DARTAssimilator.__new__(DARTAssimilator)
        self.assim.dart_lib = LIB_PATH
        self.assim.filter_kind = 'EAKF'

    def test_unknown_filter_kind_raises(self):
        self.assim.filter_kind = 'NOT_A_FILTER'
        with self.assertRaises(ValueError):
            _ = self.assim.filter_kind_code

    def test_kde_initializes_dart_and_runs(self):
        """
        KDE needs DART's utilities up before it can read kde_nml.

        The assimilator must arrange that itself: write an input.nml if none is there and
        call dart_initialize. Done in a temporary working directory, since the namelist
        lookup is cwd-relative and DART drops its log files alongside it.
        """
        import tempfile
        self.assim.filter_kind = 'KDE'
        self.assim.bounded_below = True
        self.assim.lower_bound = 0.0
        self.assim.write_input_nml = True

        prior = np.abs(np.random.default_rng(4).normal(0.5, 0.3, 60))
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                inc = self.assim.obs_increment(prior, np.zeros(0), 0.05, 0.1)
                self.assertTrue(os.path.exists('input.nml'), 'a minimal input.nml should be written')
            finally:
                os.chdir(cwd)

        post = prior + inc
        self.assertTrue(np.all(np.isfinite(post)))
        self.assertGreaterEqual(post.min(), -1e-12, 'KDE: posterior crossed the lower bound')

    def test_kde_refuses_without_input_nml_when_not_allowed_to_write(self):
        import tempfile
        self.assim.filter_kind = 'KDE'
        self.assim.write_input_nml = False
        prior = np.abs(np.random.default_rng(4).normal(0.5, 0.3, 60))
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with self.assertRaises(FileNotFoundError):
                    self.assim.obs_increment(prior, np.zeros(0), 0.05, 0.1)
            finally:
                os.chdir(cwd)

    def test_stochastic_kind_is_seeded_automatically(self):
        # without _ensure_seeded the kernel would fall back to DART's my_task_id() seeding
        # and stop the run; this must work straight out of obs_increment.
        # ENKF also reads sort_obs_inc, so the assimilator initializes DART and writes an
        # input.nml -- run it in a temporary directory rather than the repo.
        import tempfile
        self.assim.filter_kind = 'ENKF'
        self.assim.random_seed = 777
        prior = np.random.default_rng(2).normal(5.0, 1.0, 50)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                inc = self.assim.obs_increment(prior, np.zeros(0), 0.0, 1.0)
                self.assertTrue(os.path.exists('input.nml'))
            finally:
                os.chdir(cwd)
        self.assertTrue(self.assim._seeded)
        self.assertTrue(np.all(np.isfinite(inc)))

    def test_written_namelist_carries_the_configured_options(self):
        self.assim.filter_kind = 'RHF'
        self.assim.sort_obs_inc = False
        self.assim.gaussian_likelihood_tails = True
        self.assim.quadrature_order = 5
        text = self.assim._input_nml_text()
        self.assertIn('sort_obs_inc = .false.', text)
        self.assertIn('gaussian_likelihood_tails = .true.', text)
        self.assertIn('quadrature_order = 5', text)
        # the gated option is always written off, whatever else is configured
        self.assertIn('sampling_error_correction = .false.', text)
        # every section DART demands, plus kde_nml for quadrature_order
        for section in REQUIRED_NML_SECTIONS + ('kde_nml',):
            self.assertIn('&' + section, text)

    def test_incomplete_user_namelist_is_refused_in_python(self):
        """
        A missing section makes DART stop the process, so it has to be caught beforehand.
        """
        import tempfile
        self.assim.filter_kind = 'RHF'
        self.assim.write_input_nml = False      # use the file as supplied
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with open('input.nml', 'w') as f:
                    f.write('&utilities_nml\n/\n')     # no assim_tools_nml, no obs_kind_nml
                with self.assertRaises(ValueError) as err:
                    self.assim._ensure_initialized()
                self.assertIn('obs_kind_nml', str(err.exception))
            finally:
                os.chdir(cwd)

    def test_dart_error_message_is_extracted(self):
        sample = ("  ERROR FROM:\n  source : utilities_mod.f90\n"
                  "  routine: find_namelist_in_file\n"
                  "  message:  Namelist entry &obs_kind_nml must exist in file input.nml\n")
        self.assertEqual(dart_error_message(sample),
                         'Namelist entry &obs_kind_nml must exist in file input.nml')
        # unrecognised output still yields something rather than an empty message
        self.assertTrue(dart_error_message('segmentation fault'))

    def test_sampling_error_correction_is_refused(self):
        self.assim.filter_kind = 'RHF'
        self.assim.sampling_error_correction = True
        with self.assertRaises(NotImplementedError):
            self.assim._check_gated_options()

    def test_foreign_input_nml_is_not_overwritten(self):
        import tempfile
        self.assim.filter_kind = 'RHF'
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with open('input.nml', 'w') as f:
                    f.write('&utilities_nml\n/\n')      # someone else's namelist
                with self.assertRaises(RuntimeError):
                    self.assim._ensure_initialized()
                with open('input.nml') as f:
                    self.assertNotIn('NEDAS', f.read())
            finally:
                os.chdir(cwd)

    def test_update_local_state_matches_native(self):
        rng = np.random.default_rng(5)
        nens, nfld, nloc = 20, 3, 8
        obs, obs_err = 1.2, 0.7
        hroi, vroi, troi = 5.0, 2.0, 3.0
        obs_prior = rng.normal(0, 1, nens)
        h_dist = rng.uniform(0, 8, nloc)
        v_dist = rng.uniform(0, 3, (nfld, nloc))
        t_dist = rng.uniform(0, 4, nfld)
        impact = rng.uniform(0.5, 1.0, nfld)
        state = rng.normal(0, 1, (nens, nfld, nloc))

        native = state.copy()
        update_local_state_linear(native, _no_static(native), obs_prior, np.zeros(0),
                                  _plain_obs_increment(obs_prior, obs, obs_err),
                                  h_dist, v_dist, t_dist, hroi, vroi, troi,
                                  gaspari_cohn_func, gaspari_cohn_func, gaspari_cohn_func,
                                  impact, 1.0, 0.0, False)

        dart = state.copy()
        obs_incr = self.assim.obs_increment(obs_prior, np.zeros(0), obs, obs_err)
        self.assim.update_local_state(dart, _no_static(dart), obs_prior, np.zeros(0), obs_incr,
                                      h_dist, v_dist, t_dist, hroi, vroi, troi,
                                      gaspari_cohn_func, gaspari_cohn_func, gaspari_cohn_func, impact)

        self.assertFalse(np.allclose(dart, state), "test is vacuous if nothing was updated")
        np.testing.assert_allclose(dart, native, rtol=1e-12, atol=1e-12)

    def test_update_local_obs_matches_native(self):
        rng = np.random.default_rng(9)
        nens, nlobs = 20, 10
        obs, obs_err = -0.8, 1.1
        hroi, vroi, troi = 6.0, 2.5, 3.5
        obs_prior = rng.normal(0, 1, nens)
        h_dist = rng.uniform(0, 9, nlobs)
        v_dist = rng.uniform(0, 3, nlobs)
        t_dist = rng.uniform(0, 4, nlobs)
        impact = rng.uniform(0.5, 1.0, nlobs)
        used = np.zeros(nlobs, dtype=bool)
        used[:3] = True          # already-assimilated obs must be left alone
        obs_data = rng.normal(0, 1, (nens, nlobs))

        native = obs_data.copy()
        update_local_obs_linear(native, _no_static(native), used, obs_prior, np.zeros(0),
                                _plain_obs_increment(obs_prior, obs, obs_err),
                                h_dist, v_dist, t_dist, hroi, vroi, troi,
                                gaspari_cohn_func, gaspari_cohn_func, gaspari_cohn_func,
                                impact, 1.0, 0.0, False)

        dart = obs_data.copy()
        obs_incr = self.assim.obs_increment(obs_prior, np.zeros(0), obs, obs_err)
        self.assim.update_local_obs(dart, _no_static(dart), used, obs_prior, np.zeros(0), obs_incr,
                                    h_dist, v_dist, t_dist, hroi, vroi, troi,
                                    gaspari_cohn_func, gaspari_cohn_func, gaspari_cohn_func, impact)

        self.assertFalse(np.allclose(dart, obs_data), "test is vacuous if nothing was updated")
        np.testing.assert_array_equal(dart[:, used], obs_data[:, used])
        np.testing.assert_allclose(dart, native, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
