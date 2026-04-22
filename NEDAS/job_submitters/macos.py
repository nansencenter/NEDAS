from .local import LocalJobSubmitter

class MacOSJobSubmitter(LocalJobSubmitter):
    """
    MacOS
    """
    @property
    def execute_command(self):
        if self.parallel_mode == 'serial':
            return ""
        elif self.parallel_mode == 'mpi':
            return f"mpirun --map-by :OVERSUBSCRIBE -np {self.nproc}"
        elif self.parallel_mode == 'openmp':
            return f"export OMP_NUM_THREADS={self.nproc};"
        else:
            raise ValueError(f"unknown parallel_mode '{self.parallel_mode}'")
