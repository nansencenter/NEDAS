import subprocess
import sys
import os
from NEDAS.core.job_submitter import JobSubmitter

class LocalJobSubmitter(JobSubmitter):
    """
    The LocalJobSubmitter class assumes a generic GNU/Linux environment on a single PC.
    For 'serial' parallel mode: the command is directly executed in a subprocess.
    For 'mpi' or 'openmp' parallel modes: the command will be parsed accordingly to be run in an mpi environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # check for available processors on the node
        self._cpu_count = os.cpu_count()
        assert self._cpu_count is not None, "cannot query available nproc: os.cpu_count returns None"

    @property
    def nproc_avail(self):
        if self._cpu_count:
            return self._cpu_count
        return 1

    @property
    def execute_command(self) -> str:
        """
        Vanila JobSubmitter will just run "mpirun -np nproc ...", and discard the ppn and offset settings
        """
        if self.parallel_mode == 'serial':
            return ""
        elif self.parallel_mode == 'mpi':
            return f"mpirun -np {self.nproc}"
        elif self.parallel_mode == 'openmp':
            return f"export OMP_NUM_THREADS={self.nproc};"
        else:
            raise ValueError(f"unknown parallel_mode '{self.parallel_mode}'")

    def run(self, commands: str) -> None:
        """
        Runs 'commands' on the local computer.
        Use nproc processors starting from the offset+1 processor of the allocation
        """
        commands = self.parse_commands(commands)
        if self.debug:
            print("JobSubmitter run command as step: ", commands, flush=True)

        p = subprocess.Popen(commands, shell=True, text=True, bufsize=1)
        p.wait()

        # handle error
        if p.returncode != 0:
            print(f"JobSubmitter: job '{self.job_name}' exited with error")
            sys.exit(1)
