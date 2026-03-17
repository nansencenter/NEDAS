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

if __name__ == '__main__':
    unittest.main()
