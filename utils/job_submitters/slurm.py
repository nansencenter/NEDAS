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
            return f"srun --unbuffered"
        else:
            return f"srun -N {self.nproc} -n {self.nnode} -r {self.offset_node} --exact --unbuffered"

    @property
    def job_array_index_name(self):
        return '$SLURM_ARRAY_TASK_ID'

    def submit_job_and_monitor(self, commands):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.sh') as job_script:
            job_script.write("#!/bin/bash\n")

            ##slurm job header
            job_script.write(f"#SBATCH --job-name={self.job_name}\n")
            job_script.write(f"#SBATCH --account={self.project}\n")
            job_script.write(f"#SBATCH --time={seconds_to_timestr(self.walltime)}\n")
            job_script.write(f"#SBATCH --qos={self.queue}\n")
            job_script.write(f"#SBATCH --nodes={self.nnode}\n")
            job_script.write(f"#SBATCH --ntasks-per-node={self.ppn}\n")

            if self.mem_per_cpu:
                job_script.write(f"#SBATCH --mem-per-cpu={self.mem_per_cpu}\n")

            log_file = os.path.join(self.run_dir, f"{self.job_name}-%j.out")
            job_script.write(f"#SBATCH --output={log_file}\n")

            if self.use_job_array:
                job_script.write(f"#SBATCH --array=1-{self.array_size}\n")

            ##add the commands
            commands = super().parse_commands(commands)
            job_script.write(commands)
            job_script.write('\n')

            self.job_script = job_script.name

        os.system("cat "+self.job_script)

        ##submit the job script
        p = subprocess.run(['sbatch', self.job_script], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Failed to submit job: {p.stderr}")
        self.job_id = int(p.stdout.split()[-1])
        log_file = os.path.join(self.run_dir, f"{self.job_name}-{self.job_id}.out")

        ##monitor job status
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

        ##clean up
        os.remove(self.job_script)

        ##handle errors
        if not find_keyword_in_file(log_file, "Job exited normally"):
            raise RuntimeError(f"job {self.job_name} failed, check {log_file}")