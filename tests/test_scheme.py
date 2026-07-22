import unittest
from types import SimpleNamespace
from datetime import datetime
from NEDAS.schemes import get_scheme
from NEDAS.schemes.filter import FilterAnalysisScheme

class TestAnalysisScheme(unittest.TestCase):
    def test_raise_exception_when_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            get_scheme(scheme='foo')

class TestGetTaskOpts(unittest.TestCase):
    """
    prepare_init_ensemble and ensemble_forecast size their worker pool at
    nproc_per_run, the same pairing ensemble_forecast has always used, not the
    nproc_per_util pairing preprocess/postprocess use -- so their total_nproc
    must come from config.nproc, not the get_task_opts default fallback to
    nproc_util for steps that don't themselves run under mpi (issue #26).
    """
    def _make_scheme(self, nproc, nproc_util):
        fake = SimpleNamespace()
        fake.config = SimpleNamespace(
            nproc=nproc, nproc_util=nproc_util, cycle_period=6, job_submit=None,
        )
        fake.c = SimpleNamespace(time=datetime(2026, 1, 1))
        fake.steps_need_mpi = FilterAnalysisScheme.steps_need_mpi
        return fake

    def test_default_falls_back_to_nproc_util(self):
        # unaffected steps (preprocess/postprocess) keep the nproc_util default
        fake = self._make_scheme(nproc=128, nproc_util=4)
        opts = FilterAnalysisScheme.get_task_opts(fake, 'preprocess', 'mymodel')
        self.assertEqual(opts['total_nproc'], 4)

    def test_prepare_init_ensemble_call_site_uses_full_nproc(self):
        # mirrors the total_nproc override filter.py's prepare_init_ensemble
        # and ensemble_forecast now pass explicitly
        fake = self._make_scheme(nproc=128, nproc_util=4)
        opts = FilterAnalysisScheme.get_task_opts(
            fake, 'prepare_init_ensemble', 'mymodel', nproc=8, total_nproc=fake.config.nproc,
        )
        self.assertEqual(opts['total_nproc'], 128)

if __name__ == '__main__':
    unittest.main()
