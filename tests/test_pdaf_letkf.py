"""
Checks on NEDAS's interface to PDAF's compiled analysis kernels, through pyPDAF.

Skipped unless pyPDAF is importable; it has no pip/conda package, see
NEDAS/assim_tools/assimilators/PDAF/install_pypdaf.md for how to build it.

The substantive test runs one partition through PDAF's LETKF and compares against NEDAS's
native ETKF on the same data. Both are the same square-root filter, but they differ in which
square root they take (and PDAF applies its own type_trans rotation), so the comparison is on
the posterior mean and covariance rather than member by member -- those are what the filter
is defined by, and a real numerics change upstream moves them.

Each assimilator reproduces the formulation of the code it comes from, differences included --
that is the point of having several: comparing them measures what an implementation's own
choices do to a method the literature calls the same. One such difference shows up here, in
how the localization taper enters the analysis (linearly for PDAF, squared for NEDAS's ETKF),
and the tests below both measure it and hold everything else to roundoff, so that a real
numerics change upstream cannot hide behind it. Neither side is adjusted to match the other.

PDAF can only be initialized once per process (a second PDAF_init crashes) and the filter kind
is fixed at init, so no single process can run two filters. Everything that sweeps filter kinds
here therefore runs one subprocess per kind (see _run_kind); the in-process tests never switch.

The rest are structural checks that need no pyPDAF: the configuration PDAFomi cannot express
(vertical/temporal/cross-variable localization) has to be refused rather than silently dropped.
"""
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from NEDAS.assim_tools.assimilators.PDAF.core import (
    PDAFAssimilator, FILTER_KINDS, LOC_WEIGHTS, NEDAS_TO_PDAF_WEIGHT, import_pypdaf,
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


def native_etkf_analysis(state_data, obs_data, taper_power=0.5):
    """
    The same partition through NEDAS's own ETKF, as the reference.

    ``taper_power`` selects which taper the reference runs with, and exists to separate the two
    things that could make the codes disagree:

    * ``1.0`` -- each code as its own source formulates it. PDAF does textbook R-localization
      (Hunt et al. 2007), scaling the inverse observation error variance by the weight w, while
      NEDAS's ETKF multiplies the whitened obs anomalies AND the innovation by w
      (ensemble_transform_weights, whitening_factor = local_factor / obs_err_std), so its taper
      enters the analysis Hessian as w^2. The same hroi therefore localizes more tightly in
      NEDAS's ETKF than in PDAF's LETKF. That gap is a property of the two implementations, and
      test_taper_convention_differs measures it rather than removing it.
    * ``0.5`` -- the reference is handed sqrt(w), so its w^2 equals PDAF's w and the taper drops
      out of the comparison. Everything else in the two analyses then has to agree to roundoff,
      which is what makes test_letkf_matches_native_etkf a usable regression test on upstream's
      numerics: with the known difference held fixed, anything that moves is a new one.
    """
    state_post = state_data['state_prior'].copy()
    no_static_state = np.zeros((0, NFLD, NLOC))
    no_static_obs = np.zeros((0, NLOBS))
    for loc_id in range(NLOC):
        hdist = np.abs(obs_data['x'] - state_data['x'][loc_id])
        hlfactor = gaspari_cohn_func(hdist, HROI) ** taper_power
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


class FakeContext:
    """the bits of a Context these tests reach for"""
    def __init__(self, nens):
        self.nens = nens


class Grid1DStub:
    """NEDAS's Grid1D spells periodicity as a bool and has no y (lorenz96)"""
    def __init__(self, cyclic, Lx):
        self.cyclic, self.Lx, self.Ly = cyclic, Lx, 0


class Grid2DStub:
    """NEDAS's Grid spells it as a string: 'x', 'y', 'xy' or None (vort2d, qg, ...)"""
    def __init__(self, cyclic_dim, Lx, Ly):
        self.cyclic_dim, self.Lx, self.Ly = cyclic_dim, Lx, Ly


def make_assimilator(**kwargs):
    """
    A PDAFAssimilator without a Context: analyze_partition only needs the settings
    that assimilation_algorithm would have derived from the config and the grid, plus
    the process-wide PDAF_init that ensure_initialized() does.
    """
    self = PDAFAssimilator.__new__(PDAFAssimilator)
    self.filter_kind = kwargs.get('filter_kind', 'LETKF')
    self.subtype = kwargs.get('subtype', 0)
    self.forget = kwargs.get('forget', 1.0)
    self.screen = 0
    self._locweight = LOC_WEIGHTS['gaspari_cohn']
    self._disttype = 0
    self._domainsize = np.array([-1.0, -1.0])
    self.ensure_initialized(FakeContext(NENS), NFLD * NLOC)
    return self


class TestPDAFMapping(unittest.TestCase):
    """the parts that do not need pyPDAF installed"""

    def test_only_domain_localized_filters_offered(self):
        # The codes have to be PDAF's own (PDAF_da.F90), because they are handed straight to
        # PDAF_init; a version bump that renumbers them would otherwise be silent. PDAF's
        # global filters would silently ignore NEDAS's localization (they'd see only the
        # obs already cut down to the partition, a hard edge rather than a taper), so they
        # must not be reachable through filter_kind.
        self.assertEqual(FILTER_KINDS,
                         {'LSEIK': 3, 'LETKF': 5, 'LESTKF': 7, 'LNETF': 10, 'LKNETF': 11})
        for name in FILTER_KINDS:
            self.assertTrue(name.startswith('L'), name)

    def test_localization_taper_maps_to_pdafomi(self):
        for nedas_type, code in NEDAS_TO_PDAF_WEIGHT.items():
            self.assertIn(code, LOC_WEIGHTS.values(), nedas_type)

    def test_cyclic_grids_get_a_periodic_disttype(self):
        # NEDAS's two grid classes spell periodicity differently, and getting this wrong is
        # invisible except near the domain edge: Lorenz-96 is a ring, and reading only
        # cyclic_dim left PDAF localizing it as if it had two open ends.
        a = PDAFAssimilator.__new__(PDAFAssimilator)
        a.disttype = -1
        for grid, expect_disttype, expect_size in [
                (Grid1DStub(cyclic=True, Lx=40.0), 1, [40.0, -1.0]),     # lorenz96
                (Grid1DStub(cyclic=False, Lx=40.0), 0, [-1.0, -1.0]),
                (Grid2DStub(cyclic_dim='x', Lx=10.0, Ly=5.0), 1, [10.0, -1.0]),
                (Grid2DStub(cyclic_dim='xy', Lx=10.0, Ly=5.0), 1, [10.0, 5.0]),  # vort2d
                (Grid2DStub(cyclic_dim=None, Lx=10.0, Ly=5.0), 0, [-1.0, -1.0])]:
            c = FakeContext(NENS)
            c.grid = grid
            self.assertEqual(a.disttype_code(c), expect_disttype, grid)
            np.testing.assert_array_equal(a.domainsize(c), expect_size)


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
        # with the taper difference held fixed (taper_power 0.5, see native_etkf_analysis),
        # nothing else may differ: this is the check that catches upstream numerics moving.
        post = self.analyze(filter_kind='LETKF')
        reference = native_etkf_analysis(self.state_data, self.obs_data, taper_power=0.5)
        # mean and covariance define the analysis; the ensemble members themselves differ
        # by the square-root convention each filter picks
        np.testing.assert_allclose(post.mean(axis=0), reference.mean(axis=0), atol=1e-12)
        for loc_id in range(NLOC):
            np.testing.assert_allclose(np.cov(post[..., loc_id], rowvar=False),
                                       np.cov(reference[..., loc_id], rowvar=False), atol=1e-12)

    def test_taper_convention_differs(self):
        # run as each code formulates it: same method, same hroi, same observations, and a
        # posterior that differs by ~5e-2 here purely because the taper enters linearly in one
        # and squared in the other. Pinned so that a change of formulation on either side is
        # visible as a test failure rather than as a quiet shift in everyone's tuned hroi.
        post = self.analyze(filter_kind='LETKF')
        as_formulated = native_etkf_analysis(self.state_data, self.obs_data, taper_power=1.0)
        difference = np.abs(post.mean(axis=0) - as_formulated.mean(axis=0)).max()
        self.assertGreater(difference, 1e-3)
        # ... and PDAF's wider taper keeps more of the observations, so it moves the mean further
        self.assertGreater(np.abs(post.mean(axis=0) - self.prior.mean(axis=0)).max(),
                           np.abs(as_formulated.mean(axis=0) - self.prior.mean(axis=0)).max())

    def test_integer_grid_coordinates(self):
        # a grid built from integer spacing (vort3d: np.arange(nx)*dx) hands the assimilator
        # integer coordinates; pyPDAF's typed memoryviews only take float64, and this went
        # undetected until the first 3-D run because lorenz96's coordinates are floats.
        state_data = dict(self.state_data)
        state_data['state_prior'] = self.prior.copy()
        state_data['x'] = (self.state_data['x'] * 1000).astype(np.int64)
        state_data['y'] = np.zeros(NLOC, dtype=np.int64)
        obs_data = dict(self.obs_data)
        obs_data['x'] = (self.obs_data['x'] * 1000).astype(np.int64)
        obs_data['y'] = np.zeros(NLOBS, dtype=np.int64)
        obs_data['hroi'] = np.array([HROI * 1000])
        make_assimilator().analyze_partition(None, state_data, obs_data)
        self.assertTrue(np.isfinite(state_data['state_prior']).all())
        self.assertFalse(np.allclose(state_data['state_prior'], self.prior))

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

    def test_second_filter_kind_is_refused(self):
        # PDAF cannot be initialized twice in a process, so a run cannot switch filters
        # halfway (an assimilator_def.type per-iteration dict, say). That has to be an
        # error rather than the segfault a second PDAF_init would give.
        self.analyze(filter_kind='LETKF')
        with self.assertRaises(RuntimeError) as err:
            make_assimilator(filter_kind='LESTKF')
        self.assertIn('already initialized', str(err.exception))


# Runs one filter kind over the test partition, in its own interpreter, and saves the
# posterior. Inherits this module so the partition/assimilator helpers are the same ones.
_KIND_DRIVER = '''
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.environ["NEDAS_TEST_DIR"])
import test_pdaf_letkf as T
kind, hroi, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
state_data, obs_data = T.make_partition(seed=42)
obs_data = dict(obs_data)
obs_data["hroi"] = np.array([hroi])
assim = T.make_assimilator(filter_kind=kind)
sd = dict(state_data)
sd["state_prior"] = state_data["state_prior"].copy()
assim.analyze_partition(None, sd, obs_data)
np.save(out, sd["state_prior"])
'''


def _run_kind(kind, hroi=HROI):
    """Analyse the test partition with one PDAF filter kind, in a fresh process."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'post.npy')
        env = dict(os.environ,
                   NEDAS_TEST_DIR=os.path.dirname(os.path.abspath(__file__)))
        run = subprocess.run([sys.executable, '-c', _KIND_DRIVER, kind, repr(hroi), out],
                             capture_output=True, text=True, env=env, timeout=600)
        if not os.path.exists(out):
            raise AssertionError(f"{kind} produced no analysis (exit {run.returncode}):\n"
                                 f"{run.stdout[-1500:]}\n{run.stderr[-1500:]}")
        return np.load(out)


@unittest.skipUnless(HAS_PYPDAF, 'pyPDAF is not importable')
class TestPDAFFilterKinds(unittest.TestCase):
    """Every offered filter kind, through this interface.

    Each of these runs in its own process, so this class costs one interpreter per kind.
    """

    def test_every_offered_kind_analyses_a_partition(self):
        prior = make_partition(seed=42)[0]['state_prior']
        for kind in sorted(FILTER_KINDS):
            with self.subTest(filter_kind=kind):
                post = _run_kind(kind)
                self.assertTrue(np.isfinite(post).all(), f'{kind} produced non-finite values')
                self.assertEqual(post.shape, prior.shape)
                # it has to have done something to every field
                self.assertGreater(np.abs(post - prior).max(), 1e-8, kind)


if __name__ == '__main__':
    unittest.main()
