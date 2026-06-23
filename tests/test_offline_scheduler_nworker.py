"""
Tests for _run_ensemble_tasks_offline_scheduler nworker calculation.

Four cases:
  A. HPC offline, nproc > 1          -> nworker = nens
  B. HPC offline, max_concurrent set -> nworker = max_concurrent
  C. Local, nproc=4, total_nproc=16  -> nworker = 4
  D. Local, nproc not in opts        -> nworker = total_nproc (defaults nproc=1)
  E. Local, nproc > total_nproc      -> AssertionError
"""
import types
import unittest
from unittest.mock import MagicMock, patch, call

from NEDAS.core.scheme import Scheme
from NEDAS.job_submitters.hpc import HPCJobSubmitter


def make_self(*, nens=10, nproc_config=32, jsub_is_hpc=True, in_alloc=False):
    """Build a minimal stand-in for a Scheme instance."""
    s = types.SimpleNamespace()

    # config
    s.config = types.SimpleNamespace(nproc=nproc_config, debug=False)

    # context
    jsub = MagicMock(spec=HPCJobSubmitter if jsub_is_hpc else object)
    jsub.in_job_allocation = in_alloc
    comm = MagicMock()
    comm.Barrier = MagicMock()
    s.c = types.SimpleNamespace(
        nens=nens,
        jsub=jsub,
        comm=comm,
        debug_message='',
        io=MagicMock(),
    )
    s.scheduler = None
    return s


def run_offline(self_ns, opts):
    """Call the real _run_ensemble_tasks_offline_scheduler with a mocked self."""
    Scheme._run_ensemble_tasks_offline_scheduler(self_ns, tag='current', task_name='test', func=lambda: None, **opts)


class TestOfflineSchedulerNworker(unittest.TestCase):

    def _captured_nworker(self, self_ns, opts):
        """Run the method with OfflineScheduler patched; return the nworker it received."""
        captured = {}

        class FakeScheduler:
            def __init__(self_, c, nworker, *a, **kw):
                captured['nworker'] = nworker
                self_.error_jobs = {}
            def submit_job(self_, *a, **kw): pass
            def start_queue(self_): pass
            def shutdown(self_): pass

        with patch('NEDAS.core.scheme.OfflineScheduler', FakeScheduler):
            run_offline(self_ns, opts)

        return captured['nworker']

    # ── Case A: HPC offline, default concurrency ──────────────────────────
    def test_hpc_offline_uses_nens(self):
        s = make_self(nens=10, jsub_is_hpc=True, in_alloc=False)
        nw = self._captured_nworker(s, {'nproc': 256, 'total_nproc': 8})
        self.assertEqual(nw, 10)

    # ── Case B: HPC offline, max_concurrent throttle ──────────────────────
    def test_hpc_offline_max_concurrent_override(self):
        s = make_self(nens=50, jsub_is_hpc=True, in_alloc=False)
        nw = self._captured_nworker(s, {'nproc': 256, 'total_nproc': 8, 'max_concurrent': 5})
        self.assertEqual(nw, 5)

    # ── Case C: local mode (nproc fits in total_nproc) ───────────────────
    def test_local_nworker_from_total_nproc(self):
        s = make_self(nens=10, jsub_is_hpc=False, in_alloc=False)
        nw = self._captured_nworker(s, {'nproc': 4, 'total_nproc': 16})
        self.assertEqual(nw, 4)

    # ── Case D: nproc not in opts → defaults to 1 ─────────────────────────
    def test_local_nproc_defaults_to_1(self):
        s = make_self(nens=10, nproc_config=8, jsub_is_hpc=False)
        # total_nproc comes from config.nproc=8, nproc defaults to 1 → nworker=8
        nw = self._captured_nworker(s, {})
        self.assertEqual(nw, 8)

    # ── Case E: HPC but IN an allocation → falls through to local branch ──
    def test_hpc_in_allocation_uses_local_formula(self):
        s = make_self(nens=10, jsub_is_hpc=True, in_alloc=True)
        nw = self._captured_nworker(s, {'nproc': 4, 'total_nproc': 16})
        self.assertEqual(nw, 4)

    # ── Case F: local, nproc > total_nproc → AssertionError ──────────────
    def test_local_nproc_exceeds_total_raises(self):
        s = make_self(nens=10, jsub_is_hpc=False)
        with self.assertRaises(AssertionError):
            with patch('NEDAS.core.scheme.OfflineScheduler', MagicMock()):
                run_offline(s, {'nproc': 64, 'total_nproc': 8})

    # ── Case G: nproc=1 → local branch even with HPC submitter ───────────
    def test_nproc_1_stays_local_even_with_hpc_submitter(self):
        s = make_self(nens=10, jsub_is_hpc=True, in_alloc=False)
        nw = self._captured_nworker(s, {'nproc': 1, 'total_nproc': 16})
        self.assertEqual(nw, 16)


if __name__ == '__main__':
    unittest.main(verbosity=2)
