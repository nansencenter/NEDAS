from abc import abstractmethod
import sys
import subprocess
from NEDAS.core import JobSubmitter

class HPCJobSubmitter(JobSubmitter):
    """
    JobSubmitter Class customized for a HPC with a job scheduler.

    Args:
        project (str, optional): Project account name for billing/allocation. Defaults to None.
        queue (str, optional): The submission queue name. Defaults to None.
        ppn (int, optional): Processors per node. Defaults to 1.
        walltime (int, optional): Maximum execution time in seconds. Defaults to 3600.
        check_dt (int, optional): Time interval (sec) between status checks. Defaults to 20.
        use_job_array (bool, optional): Whether to utilize scheduler job arrays. Defaults to False.
        array_size (int, optional): Number of elements in the job array. Defaults to 1.
    """
    def __init__(self,
                 project: str|None=None,
                 queue: str|None=None,
                 ppn: int=1, 
                 walltime: int=3600,
                 check_dt: int=20,
                 use_job_array: bool=False,
                 array_size: int=1,
                 **kwargs):
        super().__init__(**kwargs)

        # HPC specific settings
        self.project = project
        self.queue = queue
        self._ppn = ppn
        self.walltime = walltime
        self.check_dt = check_dt 
        self.use_job_array = use_job_array
        self.array_size = array_size

    @property
    def ppn(self) -> int:
        """
        Number of processors per compute node requested for the job
        """
        return self._ppn

    @ppn.setter
    def ppn(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid ppn specified: {value}")
        if self.parallel_mode == 'serial' and value > 1:
            raise ValueError(f"cannot set ppn = {value} in serial mode")
        if value > self.ppn_avail:
            raise ValueError(f"requested ppn {value} exceeds available ppn={self.ppn_avail}")
        self._ppn = value

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
    @abstractmethod
    def nnode_avail(self) -> int:
        """
        Number of available compute nodes on a host machine
        """
        ...

    @property
    @abstractmethod
    def ppn_avail(self) -> int:
        """
        Number of available processors per compute node
        """
        ...

    @property
    @abstractmethod
    def job_array_index_name(self) -> str:
        """
        Job array index variable name for the host machine, replacing 'JOB_ARRAY_INDEX' in 'commands'
        """
        ...

    def parse_commands(self, commands: str) -> str:
        commands = super().parse_commands(commands)
        commands = commands.replace('JOB_ARRAY_INDEX', self.job_array_index_name)
        return commands

    @property
    @abstractmethod
    def in_job_allocation(self) -> bool:
        """
        Determines if a job allocation is already availalbe on the HPC
        If so, the job can be run as a sub step directly, otherwise will need to submit it to the queue.
        """
        ...

    def check_resources(self) -> None:
        """
        Checks if requested resource is available
        """
        assert self.nproc+self.offset <= self.nproc_avail, f"Requested nproc={self.nproc} and offset={self.offset} exceeds nproc_avail={self.nproc_avail}"
        assert self.nnode+self.offset_node <= self.nnode_avail, f"Requested nnode={self.nnode} and offset_node={self.offset_node} exceeds nnode_avail={self.nnode_avail}"

    def run(self, commands):
        if self.in_job_allocation:
            self.run_job_as_step(commands)
        else:
            self.submit_job_and_monitor(commands)

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

    @abstractmethod
    def submit_job_and_monitor(self, commands):
        """
        Submit 'commands' as a job script to the scheduler on HPC and monitor for its completion.
        """
        ...
