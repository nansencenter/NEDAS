import subprocess
from NEDAS.job_submitters.oar import OARJobSubmitter

class GricadJobSubmitter(OARJobSubmitter):
    """
    Gricad
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

