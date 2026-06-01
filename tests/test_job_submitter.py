import unittest

from NEDAS.core import Context, JobSubmitter
from NEDAS.job_submitters import registry_by_host, registry_by_scheduler, get_job_submitter

class TestJobSubmitter(unittest.TestCase):

    def test_init_job_submitter_by_host(self):
        for host in registry_by_host.keys():
            jsub = get_job_submitter(host=host)
            self.assertIsInstance(jsub, JobSubmitter)

    def test_init_job_submitter_by_scheduler(self):
        for scheduler in registry_by_scheduler.keys():
            jsub = get_job_submitter(scheduler=scheduler)
            self.assertIsInstance(jsub, JobSubmitter)

class TestLocalJobSubmitter(unittest.TestCase):

    def setUp(self):
        job_submit = {'host':'local', 'parallel_mode':'mpi'}
        self.c = Context(job_submit=job_submit)

    def test_nproc_setter(self):
        self.c.jsub.nproc = 2
        self.assertEqual(self.c.jsub.nproc, 2)

    def test_nproc_invalid_value(self):
        with self.assertRaises(ValueError):
            self.c.jsub.nproc = 0
        with self.assertRaises(ValueError):
            self.c.jsub.nproc = -2
        with self.assertRaises(ValueError):
            self.c.jsub.nproc = '1'

    def test_nproc_serial_mode_constraint(self):
        self.c.jsub.parallel_mode = 'serial'
        with self.assertRaises(ValueError):
            self.c.jsub.nproc = 2

    def test_nproc_edge_cases(self):
        with self.assertRaises(ValueError):
            self.c.jsub.offset = 0
            self.c.jsub.nproc = self.c.jsub.nproc_avail+1
        with self.assertRaises(ValueError):
            self.c.jsub.offset = self.c.jsub.nproc_avail
            self.c.jsub.nproc = 1

    def test_offset_setter(self):
        self.c.jsub.offset = 2
        self.assertEqual(self.c.jsub.offset, 2)

    def test_offset_invalid_value(self):
        with self.assertRaises(ValueError):
            self.c.jsub.offset = -2
        with self.assertRaises(ValueError):
            self.c.jsub.offset = '1'

    def test_offset_edge_cases(self):
        # 1 + nproc_avail > nproc_avail, exceeds available resources
        with self.assertRaises(ValueError):
            self.c.jsub.nproc = 1
            self.c.jsub.offset = self.c.jsub.nproc_avail
        # 0 + nproc_avail = nproc_avail, should be ok
        self.c.jsub.offset = 0
        self.c.jsub.nproc = self.c.jsub.nproc_avail
        self.assertEqual(self.c.jsub.offset, 0)

    def test_parse_commands(self):
        commands = "JOB_EXECUTE echo 1"
        self.c.jsub.parallel_mode = 'serial'
        self.assertEqual(self.c.jsub.parse_commands(commands), ' echo 1')
        self.c.jsub.parallel_mode = 'mpi'
        self.c.jsub.nproc = 2
        self.assertEqual(self.c.jsub.parse_commands(commands), 'mpirun -np 2 echo 1')
        self.c.jsub.parallel_mode = 'openmp'
        self.assertEqual(self.c.jsub.parse_commands(commands), 'export OMP_NUM_THREADS=2; echo 1')

if __name__ == '__main__':
    unittest.main()
