import os
import subprocess
import tempfile
from time import sleep
from utils.conversion import seconds_to_timestr
from utils.progress import find_keyword_in_file
from .base import JobSubmitter

class SLURMJobSubmitter(JobSubmitter):
    """JobSubmitter Class customized for SLURM schedulers"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parallel_mode = kwargs.get('parallel_mode', 'mpi')

        ##additional slurm options
        if self.host == 'betzy':
            self.ppn = kwargs.get('ppn', 128)
        self.mem_per_cpu = kwargs.get('mem_per_cpu')

    @property
    def nproc_avail(self):
        return int(os.environ['SLURM_NTASKS'])

    @property
    def nnode_avail(self):
        return int(os.environ['SLURM_NNODES'])

    @property
    def ppn_avail(self):
        return int(os.environ['SLURM_TASKS_PER_NODE'].split('(')[0])

    @property
    def execute_command(self):
        if self.run_separate_jobs:
            return f"srun -n {self.nproc} --unbuffered"
        else:
            if self.parallel_mode == 'mpi':
                return f"srun -n {self.nproc} -N {self.nnode} -r {self.offset_node} --exact --unbuffered"
            elif self.parallel_mode == 'openmp':
                return f"export OMP_NUM_THREADS={self.nproc}; srun -N 1 -r {self.offset_node} -n 1 --cpus-per-task={self.nproc} --unbuffered"
            else:
                raise ValueError(f"unknown parallel_mode '{self.parallel_mode}'")

    @property
    def job_array_index_name(self):
        return '$SLURM_ARRAY_TASK_ID'

    def submit_job_and_monitor(self, commands):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                         dir=self.run_dir,
                                         prefix=self.job_name+'.',
                                         suffix='.sh') as job_script:
            job_script.write("#!/bin/bash\n")

            ##slurm job header
            job_script.write(f"#SBATCH --job-name={self.job_name}\n")
            job_script.write(f"#SBATCH --account={self.project}\n")
            job_script.write(f"#SBATCH --time={seconds_to_timestr(self.walltime)}\n")
            job_script.write(f"#SBATCH --nodes={self.nnode}\n")
            job_script.write(f"#SBATCH --ntasks-per-node={self.ppn}\n")
            if self.queue and self.queue != 'normal':
                job_script.write(f"#SBATCH --qos={self.queue}\n")
            if self.mem_per_cpu:
                job_script.write(f"#SBATCH --mem-per-cpu={self.mem_per_cpu}\n")

            if self.use_job_array:
                log_file = os.path.join(self.run_dir, f"{self.job_name}-%A_%a.out")
            else:
                log_file = os.path.join(self.run_dir, f"{self.job_name}-%j.out")
            job_script.write(f"#SBATCH --output={log_file}\n")

            if self.use_job_array:
                job_script.write(f"#SBATCH --array=1-{self.array_size}\n")

            ##add the commands
            commands = super().parse_commands(commands)
            job_script.write(commands)
            job_script.write('\n')

            self.job_script = job_script.name
        
        ##submit the job script
        p = subprocess.run(['sbatch', self.job_script], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Failed to submit job: {p.stderr}")
        self.job_id = int(p.stdout.split()[-1])

        ##monitor job status
        if self.use_job_array:
            while True:
                sleep(20)
                job_finished = []
                for i in range(self.array_size):
                    p = subprocess.run(['squeue', '-h', '-j', f'{self.job_id}_{i}'], capture_output=True, text=True)
                    if not p.stdout:
                        job_finished.append(True)
                    else:
                        job_finished.append(False)    
                if all(job_finished):
                    break
                        
        else:
            while True:
                sleep(20)
                p = subprocess.run(['squeue', '-h', '-j', f'{self.job_id}'], capture_output=True, text=True)
                if not p.stdout:
                    ##job no longer in queue
                    break
                job_status = p.stdout.split()[4]
                if job_status not in ['R', 'PD', 'CG']:
                    ##job not running, pending, or cleaning up
                    raise RuntimeError(f"job {self.job_name} failed with status {job_status}")

        ##check log file and report errors
        if self.use_job_array:
            for i in range(self.array_size):
                log_file = os.path.join(self.run_dir, f"{self.job_name}-{self.job_id}_{i}.out")
                if not find_keyword_in_file(log_file, "Job exited normally"):
                    raise RuntimeError(f"job {self.job_name} failed, check {log_file}")
        else:
            log_file = os.path.join(self.run_dir, f"{self.job_name}-{self.job_id}.out")
            if not find_keyword_in_file(log_file, "Job exited normally"):
                raise RuntimeError(f"job {self.job_name} failed, check {log_file}")
