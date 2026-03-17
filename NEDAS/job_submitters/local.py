import subprocess
import sys
from NEDAS.core.job_submitter import JobSubmitter

class LocalJobSubmitter(JobSubmitter):
    """
    The LocalJobSubmitter class assumes a generic GNU/Linux environment.
    For 'serial' parallel mode: the command is directly executed in a subprocess.
    For 'mpi' or 'openmp' parallel modes: the command will be parsed accordingly to be run in an mpi environment.
    """

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

        ##handle error
        if p.returncode != 0:
            print(f"JobSubmitter: job '{self.job_name}' exited with error")
            sys.exit(1)
