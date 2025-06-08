import sys
import subprocess

class JobSubmitter:
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
        -check_dt: int, time interval (in seconds) to check job status
        -run_separate_jobs: bool, if true, submit each job separately to the scheduler, otherwise run in the same allocation as job steps.
        -use_job_array: bool, if true, use job array for ensemble members
        -array_size: int, size of the job array
        -parallel_mode
        -debug
        """
        ## setting default parameters if not specified in kwargs
        self.job_name = kwargs.get('job_name', 'run')
        self.run_dir = kwargs.get('run_dir', '.')
        self._nproc = kwargs.get('nproc', 1)
        self._ppn = kwargs.get('ppn', 1)
        self._offset = kwargs.get('offset', 0)
        self.host = kwargs.get('host', 'localhost').lower()
        self.project = kwargs.get('project')
        self.queue = kwargs.get('queue')
        self.walltime = kwargs.get('walltime', 3600)
        self.check_dt = kwargs.get('check_dt', 20)
        self.run_separate_jobs = kwargs.get('run_separate_jobs', False)
        self.use_job_array = kwargs.get('use_job_array', False)
        self.array_size = kwargs.get('array_size', 1)
        self.parallel_mode = kwargs.get('parallel_mode', 'mpi')
        self.debug = kwargs.get('debug', False)

    @property
    def nproc(self) -> int:
        """
        Number of requested processors for the job
        """
        return self._nproc

    @nproc.setter
    def nproc(self, value):
        self._nproc = value

    @property
    def ppn(self) -> int:
        """
        Number of processors per compute node requested for the job
        """
        return self._ppn

    @ppn.setter
    def ppn(self, value):
        self._ppn = value

    @property
    def offset(self) -> int:
        """
        Number of processors to skip from the beginning for the job
        This allows different jobs to spawm the total available nproc in the allocation
        Discarded if run_separate_jobs
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    @property
    def nnode(self) -> int:
        """
        Number of compute nodes for the job
        """
        return (self._nproc + self._ppn - 1) // self._ppn

    @property
    def offset_node(self) -> int:
        """
        Number of compute nodes to skip from the beginning
        """
        return self._offset // self._ppn

    @property
    def nproc_avail(self) -> int:
        """
        Number of available processors on a host machine (only used when not run_separate_jobs)
        Vanila JobSubmitter assumes that requested nproc is always available
        This should be redefined in subclasses to machine specific behavior
        """
        return self.nproc + self.offset

    @property
    def nnode_avail(self) -> int:
        """
        Number of available compute nodes on a host machine (only used when not run_separate_jobs)
        """
        return self.nnode + self.offset_node

    @property
    def ppn_avail(self) -> int:
        """
        Number of available processors per compute node (only used when not run_separate_jobs)
        """
        return self.ppn

    @property
    def execute_command(self) -> str:
        """
        Execute command for running the job on the host machine, replacing 'JOB_EXECUTE' in 'commands'
        Vanila JobSubmitter will just run "mpirun -np nproc ...", and discard the ppn and offset settings
        """
        if self.parallel_mode == 'mpi':
            return f"mpirun -np {self.nproc}"
        elif self.parallel_mode == 'openmp':
            return f"export OMP_NUM_THREADS={self.nproc};"
        else:
            raise ValueError(f"unknown parallel_mode '{self.parallel_mode}'")

    @property
    def job_array_index_name(self) -> str:
        """
        Job array index variable name for the host machine, replacing 'JOB_ARRAY_INDEX' in 'commands'
        """
        return 'JOB_ARRAY_INDEX'

    def parse_commands(self, commands) -> str:
        commands = commands.replace('JOB_EXECUTE', self.execute_command)
        commands = commands.replace('JOB_ARRAY_INDEX', self.job_array_index_name)
        return commands

    def check_resources(self) -> None:
        ##check if requested resource is available
        assert self.nproc+self.offset <= self.nproc_avail, f"Requested nproc={self.nproc} and offset={self.offset} exceeds nproc_avail={self.nproc_avail}"
        assert self.nnode+self.offset_node <= self.nnode_avail, f"Requested nnode={self.nnode} and offset_node={self.offset_node} exceeds nnode_avail={self.nnode_avail}"

    def run_job_as_step(self, commands) -> None:
        """
        Run 'commands' from within a job allocation
        Use nproc processors starting from the offset+1 processor of the allocation
        """
        self.check_resources()

        commands = self.parse_commands(commands)
        if self.debug:
            print("JobSubmitter run command as step: ", commands, flush=True)

        p = subprocess.Popen(commands, shell=True, text=True, bufsize=1)
        p.wait()

        ##handle error
        if p.returncode != 0:
            print(f"JobSubmitter: job '{self.job_name}' exited with error")
            sys.exit(1)

    def submit_job_and_monitor(self, commands) -> None:
        raise NotImplementedError("run_separate_jobs=True is not availalbe without a scheduler")

    def run(self, commands) -> None:
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

