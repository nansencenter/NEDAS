"""
What the Python/compiled boundary costs in the DART and PDAF assimilators.

NEDAS runs its own Python assimilators against compiled ones, and the comparison only reads
if the price of the binding is known. Neither binding is a thin wrapper: DART crosses the
boundary three times per observation, and PDAF calls back into Python once per local analysis
domain, so the crossing count is a function of the problem size, not a constant.

The instrument is off unless NEDAS_CALL_COST is set, since it wraps regions only a few
hundred nanoseconds long; these tests turn it on explicitly, and one of them checks that it
really does cost nothing when left off.

These are tests of the instrument (NEDAS/utils/call_cost.py), not the measurement. The
numbers that go in a paper come from a realistic partition on a cluster -- the vort3d
scaling baseline, where pyPDAF is built -- so what is checked here is that the counts come
out equal to the problem's own arity, which is the thing that makes the cost scale, and that
a measured kernel region never exceeds the Python region around it.
"""
import os
import unittest
from unittest import mock
import numpy as np

from NEDAS.utils.call_cost import CallCost

from test_pdaf_letkf import (
    HAS_PYPDAF, NENS, NFLD, NLOC, make_partition, make_assimilator,
)
from test_dart_kernels import LIB_PATH
from NEDAS.assim_tools.assimilators.DART.core import DARTAssimilator


class TestCallCost(unittest.TestCase):
    """the instrument itself, which needs neither library"""

    def test_counts_and_accumulates(self):
        cost = CallCost(enabled=True)
        for _ in range(3):
            with cost.measure('region'):
                pass
        self.assertEqual(cost.count('region'), 3)
        self.assertGreater(cost.seconds('region'), 0.0)

    def test_wrap_preserves_return_and_counts_calls(self):
        cost = CallCost(enabled=True)
        wrapped = cost.wrap(lambda a, b: a + b, label='add')
        self.assertEqual([wrapped(1, 2), wrapped(3, 4)], [3, 7])
        self.assertEqual(cost.count('add'), 2)

    def test_counts_a_raising_call(self):
        """a callback that raises still crossed the boundary, so it still counts"""
        cost = CallCost(enabled=True)

        def boom():
            raise ValueError('boom')

        with self.assertRaises(ValueError):
            cost.wrap(boom)()
        self.assertEqual(cost.count('boom'), 1)

    def test_nested_region_is_a_share_of_the_enclosing_one(self):
        cost = CallCost(enabled=True)
        with cost.measure('outer'):
            for _ in range(5):
                with cost.measure('inner'):
                    sum(range(10000))
        self.assertEqual(cost.count('inner'), 5)
        self.assertLessEqual(cost.seconds('inner'), cost.seconds('outer'))
        # the enclosing region is not itself a crossing
        self.assertEqual(cost.crossings('outer'), 5)
        self.assertIn('share', cost.report('outer'))

    def test_disabled_records_nothing_and_adds_no_wrapper(self):
        """
        The default. An analysis that is not being measured must not pay for the instrument,
        so wrap has to hand back the very same function object, not a wrapper around it.
        """
        def payload():
            return 1

        cost = CallCost(enabled=False)
        self.assertIs(cost.wrap(payload), payload)
        with cost.measure('region'):
            payload()
        self.assertEqual(cost.calls, {})
        self.assertIn('NEDAS_CALL_COST', cost.report())

    def test_enabled_follows_the_environment(self):
        with mock.patch.dict(os.environ, {'NEDAS_CALL_COST': '1'}):
            self.assertTrue(CallCost().enabled)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(CallCost().enabled)

    def test_report_without_an_enclosing_region(self):
        cost = CallCost(enabled=True)
        with cost.measure('solo'):
            pass
        self.assertIn('solo', cost.report())


@unittest.skipUnless(os.path.exists(LIB_PATH), f"DART kernel library not built: {LIB_PATH}")
class TestDARTBindingCost(unittest.TestCase):
    """
    Three crossings per observation: one obs_increment, then one regression onto the state
    and one onto the remaining obs priors. That is the arity the serial loop pays per
    observation, so it is what the cost scales with.
    """

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {'NEDAS_CALL_COST': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assim = DARTAssimilator.__new__(DARTAssimilator)
        self.assim.dart_lib = LIB_PATH
        self.assim.filter_kind = 'EAKF'
        self.assim.bounded_below = self.assim.bounded_above = False
        self.assim.lower_bound = self.assim.upper_bound = 0.0

    def test_crossings_scale_with_observation_count(self):
        rng = np.random.default_rng(0)
        nens, nobs, nloc = 20, 4, 8
        state = np.ascontiguousarray(rng.normal(0, 1, (nens, 1, nloc)))
        obs_prior = np.ascontiguousarray(rng.normal(0, 1, (nens, nobs)))

        for j in range(nobs):
            incr = self.assim.obs_increment(obs_prior[:, j], None, float(rng.normal()), 1.0)
            self.assim._regress(state, obs_prior[:, j], incr, np.ones(nloc))

        cost = self.assim.call_cost
        self.assertEqual(cost.count('obs_increment'), nobs)
        self.assertEqual(cost.count('dart_obs_increment'), nobs)
        self.assertEqual(cost.count('_regress'), nobs)
        self.assertEqual(cost.count('dart_update_from_obs_inc'), nobs)
        # the kernel is inside the method that prepares its arrays, so it cannot cost more
        self.assertLessEqual(cost.seconds('dart_obs_increment'), cost.seconds('obs_increment'))
        self.assertLessEqual(cost.seconds('dart_update_from_obs_inc'), cost.seconds('_regress'))
        print('\nDART binding cost\n' + cost.report())


@unittest.skipUnless(HAS_PYPDAF, 'pyPDAF not installed')
class TestPDAFBindingCost(unittest.TestCase):
    """
    PDAF drives the analysis and calls Python back, so the crossing count follows the
    problem: the per-domain callbacks run once per local analysis domain, and the obs
    operator once per member plus once for the ensemble mean.

    This runs in whatever process the suite gives it, and PDAF can only be initialized once
    per process -- so it measures the partition the shared fixture builds rather than a
    realistic one. The point here is the arity, not the seconds.
    """

    def test_crossing_counts_follow_the_partition(self):
        patcher = mock.patch.dict(os.environ, {'NEDAS_CALL_COST': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        state_data, obs_data = make_partition()
        assim = make_assimilator()
        assim.analyze_partition(None, state_data, obs_data)

        cost = assim.call_cost
        self.assertEqual(cost.count('init_dim_l_pdaf'), NLOC)
        self.assertEqual(cost.count('init_dim_obs_l_pdafomi'), NLOC)
        self.assertEqual(cost.count('prepoststep_pdaf'), 2)
        # H(x) is served per member, and PDAF also asks for the ensemble mean
        self.assertGreaterEqual(cost.count('obs_op_pdafomi'), NENS)
        # every callback ran inside the analysis, so none can exceed it
        for label in cost.calls:
            if label != 'assim_offline':
                self.assertLessEqual(cost.seconds(label), cost.seconds('assim_offline'))
        self.assertEqual(cost.crossings('assim_offline'), sum(
            cost.count(label) for label in cost.calls if label != 'assim_offline'))
        print('\nPDAF binding cost (NENS=%d, NFLD=%d, NLOC=%d)\n%s'
              % (NENS, NFLD, NLOC, cost.report('assim_offline')))


if __name__ == '__main__':
    unittest.main()
