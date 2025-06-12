import subprocess
from .slurm import SLURMJobSubmitter

class BetzyJobSubmitter(SLURMJobSubmitter):
    """
    JobSubmitter subclass for Norwegian betzy.sigma2.no supercomputers
    """

    def check_resources(self):
        super().check_resources()

        ##don't allow more than 6 processors on betzy login node
        p = subprocess.run("hostname", capture_output=True, text=True)
        if p.stdout.strip()[:5] == 'login':
            assert self.nproc+self.offset < 6, "Unsafe to run more than 6 processors on Betzy login node, aborting"
