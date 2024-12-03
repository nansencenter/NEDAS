import os
import sys
import subprocess
import tempfile

class JobSubmitter(object):

    def __init__(self, **kwargs):
        """
        Input kwargs:
        -job_name: str, name of the job
        -nproc: int, number of processors requested
        -offset: int, number of processors to offset from the start
        -ppn: int, number of processors per compute node
        -host: str, host machine name
        -project: str or None, project id for allocation
        -queue: str or None, which job queue to use
        -walltime: int, maximum run time allowed (in seconds)
        -run_separate_jobs: bool, if true, submit each job separately to the scheduler, otherwise run in the same allocation as job steps.
        -use_job_array: bool, if true, use job array for ensemble members
        -nens: int, ensemble size
        """
        ## setting default parameters if not specified in kwargs
        self.job_name = kwargs.get('job_name', 'run')
        self._nproc = kwargs.get('nproc', 1)
        self._ppn = kwargs.get('ppn', 1)
        self._offset = kwargs.get('offset', 0)
        self.host = kwargs.get('host', 'localhost').lower()
        self.project = kwargs.get('project')
        self.queue = kwargs.get('queue')
        self.walltime = kwargs.get('walltime', 3600)
        self.run_separate_jobs = kwargs.get('run_separate_jobs', False)
        self.use_job_array = kwargs.get('use_job_array', False)
        self.nens = 1

        ##some host specific settings here:
        if self.host == 'betzy':
            ##get the node name
            r = subprocess.run("hostname", capture_output=True, text=True)
            if r.stdout.strip()[:5] == 'login':
                if self.nproc > 16:
                    print("WARNING: you are running job on Betzy login node, reducing to nproc=16...")
                    self.nproc = 16

    @property
    def nproc(self):
        """
        Number of requested processors for the job
        """
        return self._nproc

    @nproc.setter
    def nproc(self, value):
        self._nproc = value

    @property
    def ppn(self):
        """
        Number of processors per compute node requested for the job
        """
        return self._ppn

    @ppn.setter
    def ppn(self, value):
        self._ppn = value

    @property
    def offset(self):
        """
        Number of processors to skip from the beginning for the job
        This allows different jobs to spawm the total available nproc in the allocation
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    @property
    def nnode(self):
        """
        Number of compute nodes for the job
        """
        return self._nproc // self._ppn

    @property
    def offset_node(self):
        """
        Number of compute nodes to skip from the beginning
        """
        return self._offset // self._ppn

    @property
    def nproc_avail(self):
        """
        Number of available processors on a host machine
        Vanila JobSubmitter assumes that requested nproc is always available
        This should be redefined in subclasses to machine specific behavior
        """
        return self.nproc

    @property
    def nnode_avail(self):
        """
        Number of available compute nodes on a host machine
        """
        return self.nnode

    @property
    def ppn_avail(self):
        """
        Number of available processors per compute node
        """
        return self.ppn

    @property
    def execute_command(self):
        """
        Execute command for running the job on the host machine, replacing 'JOB_EXECUTE' in 'commands'
        Vanila JobSubmitter will just run "mpirun -np nproc ...", and discard the ppn and offset settings
        """
        return f"mpirun -np {self.nproc}"

    @property
    def job_array_index_name(self):
        """
        Job array index variable name for the host machine, replacing 'JOB_ARRAY_INDEX' in 'commands'
        """
        return 'JOB_ARRAY_INDEX'

    def parse_commands(self, commands):
        commands = commands.replace('JOB_EXECUTE', self.execute_command)
        commands = commands.replace('JOB_ARRAY_INDEX', self.job_array_index_name)
        return commands

    def check_resources(self):
        assert self.nproc+self.offset <= self.nproc_avail, f"Requested nproc={self.nproc} and offset={self.offset} exceeds nproc_avail={self.nproc_avail}"
        assert self.nnode+self.offset_ndoe <= self.nnode_avail, f"Requested nnode={self.nnode} and offset_node={self.offset_node} exceeds nnode_avail={self.nnode_avail}"

    def run_job_as_step(self, commands):
        """
        Run 'commands' from within a job allocation
        Use nproc processors starting from the offset+1 processor of the allocation
        """
        commands = self.parse_commands(commands)

        p = subprocess.Popen(commands, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
        p.wait()

        ##handle error
        if p.returncode != 0:
            print(f"JobSubmitter: job '{self.job_name}' exited with error")
            sys.exit(1)

    def submit_job_and_monitor(self, commands):
        """
        Build a job script with commands and submit it to the queue
        Monitor job status in the queue and wait until it finishes
        """
        self.check_resources()

        ##create a temporary job script to be submitted to the queue
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.sh') as job_script:
            job_script.write("#!/bin/bash\n")

            ##add scheduler headers here

            ##add the commands
            commands = self.parse_commands(commands)
            job_script.write(commands)
            job_script.write('\n')

            self.job_script = job_script.name

        ##submit the job script
        self.job_id = subprocess.Popen(f"bash {self.job_script}", shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)

        ##wait for job to complete
        ##with schedulers this should be replaced by a query of the queue status
        self.job_id.wait()

        ##clean up temp job script
        os.remove(self.job_script)

        ##handle error
        if self.job_id.returncode != 0:
            print(f"JobSubmitter: job '{self.job_name}' exited with error")
            sys.exit(1)

    def run(self, commands):
        """
        Top level run method
        Input commands: str, shell commands to be run by the job submitter
        In the commands string, 'JOB_EXECUTE' will be replaced by the correct execute_command,
        and 'JOB_ARRAY_INDEX' will be replaced by the scheduler's index variable name to perform array jobs.
        """
        if self.run_separate_jobs:
            self.submit_job_and_monitor(commands)

        else:
            self.run_job_as_step(commands)

