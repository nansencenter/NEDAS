from abc import ABC, abstractmethod
from .types import ParallelMode

class JobSubmitter(ABC):
    """
    Run a command (shell script) on a specific computer.
    For a local computer this is as easy as a subprocess call to execute the command.
    But for large-scale HPC, this may involve submitting the job script to a scheduler and wait for its completion.

    Args:
        job_name (str, optional): Name of the job. Defaults to 'run'.
        run_dir (str, optional): Directory where the job will execute. Defaults to '.'.
        nproc (int, optional): Number of processors requested. Defaults to 1.
        offset (int, optional): Number of processors to offset from the start. Defaults to 0.
        parallel_mode (ParallelMode, optional): Parallelization strategy (e.g., 'serial', 'mpi', 'openmp'). Defaults to 'serial'.
        debug (bool, optional): Enables verbose logging for troubleshooting. Defaults to False.
        **kwargs: Other arbitrary keyword arguments.
    """
    def __init__(self,
                 job_name: str='run',
                 run_dir: str='.',
                 nproc: int=1,
                 offset: int=0,
                 parallel_mode: ParallelMode='serial',
                 debug: bool=False,
                 **kwargs):
        ## setting default parameters if not specified in kwargs
        self.job_name = job_name
        self.run_dir = run_dir
        self._nproc = nproc
        self._offset = offset
        self.parallel_mode = parallel_mode
        self.debug = debug

    @property
    def nproc(self) -> int:
        """
        Number of requested processors for the job
        """
        return self._nproc

    @nproc.setter
    def nproc(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid nproc specified: {value}")
        if self.parallel_mode == 'serial' and value > 1:
            raise ValueError(f"cannot set nproc = {value} in serial mode")
        if (value+self.offset) > self.nproc_avail:
            raise ValueError(f"requested nproc+offset {value}+{self.offset} exceeds total available nproc {self.nproc_avail}")
        self._nproc = value

    @property
    def offset(self) -> int:
        """
        Number of processors to skip from the beginning for the job
        This allows different jobs to spawm the total available nproc in the allocation
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid offset specified: {value}")
        if (self.nproc+value) > self.nproc_avail:
            raise ValueError(f"requested nproc+offset {self.nproc}+{value} exceeds total available nproc {self.nproc_avail}")
        self._offset = value

    @property
    @abstractmethod
    def nproc_avail(self) -> int:
        """
        Number of available processors on a host machine
        This should be redefined in subclasses to machine specific behavior
        """
        ...

    @property
    @abstractmethod
    def execute_command(self) -> str:
        """
        Execute command for running the job on the host machine, replacing 'JOB_EXECUTE' in 'commands'
        """
        ...

    def parse_commands(self, commands: str) -> str:
        """
        Parse shell command to replace 'JOB_EXECUTE' with machine-specific strings.
        """
        commands = commands.replace('JOB_EXECUTE', self.execute_command)
        return commands

    @abstractmethod
    def run(self, commands: str) -> None:
        """
        Top level run method for a job submitter.

        Args:
            commands (str): shell commands to be run by the job submitter.
                In the commands string, 'JOB_EXECUTE' will be replaced by the correct execute_command,
                and 'JOB_ARRAY_INDEX' will be replaced by the scheduler's index variable name to perform array jobs.
        """
        ...