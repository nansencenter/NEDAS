import subprocess
from NEDAS.job_submitters.oar import OARJobSubmitter

class GricadJobSubmitter(OARJobSubmitter):
    """
    Job submitter configured specifically for Gricad machines
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ##host specific settings
        p = subprocess.run("hostname", capture_output=True, text=True)
        if p.stdout.replace('\n', '').replace(' ', '') in ['dahu-oar3', 'f-dahu']:
            self.job_submit_node = None  ##don't need ssh for oarsub on compute nodes
        else:
            ##job submit node based on queue type
            if self.queue == 'devel':
                self.job_submit_node = 'dahu-oar3'
            else:
                self.job_submit_node = 'f-dahu'

    def job_submit_cmd(self):
        if self.job_submit_node:
            return ['ssh', self.job_submit_node, f"oarsub -S {self.job_script}"]
        else:
            return ["oarsub", "-S", f"{self.job_script}"]

    def check_job_status(self):
        assert self.job_submit_node is not None

        if self.use_job_array:
            cmd = f'oarstat -f --array {self.job_id} | grep "state = "'
        else:
            cmd = f'oarstat -f --job {self.job_id} | grep "state = "'

        p = subprocess.run(['ssh', self.job_submit_node, cmd], capture_output=True, text=True)
        return p