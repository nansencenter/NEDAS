from math import log
import os
import subprocess
import tempfile
from time import sleep
from NEDAS.utils.conversion import seconds_to_timestr
from .hpc import HPCJobSubmitter

class SLURMJobSubmitter(HPCJobSubmitter):
    """JobSubmitter Class customized for SLURM schedulers"""
    MAX_NTASKS = 1000000
    MAX_NNODES = 1000
    MAX_PPN = 1000

    # Pending (PD) reasons that will never resolve on their own. A job stuck with
    # one of these (e.g. "launch failed requeued held") would otherwise keep the
    # monitor loop waiting forever, so we treat them as submission failures.
    PENDING_FAILURE_REASONS = {
        'JobHeldUser', 'JobHeldAdmin', 'BadConstraints', 'DependencyNeverSatisfied',
        'InvalidQOS', 'InvalidAccount', 'PartitionDown', 'PartitionInactive',
        'PartitionConfig', 'QOSGrpBillingMinutes',
    }

    # After a job leaves the queue the scheduler may still take a moment to flush its
    # .out file (where the job script writes its exit-code sentinel). Poll for that
    # sentinel up to this many seconds before declaring the job failed, so a job that
    # actually finished isn't falsely reported as a missing-/incomplete-.out failure.
    EXIT_CODE_TIMEOUT = 60

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # additional slurm options
        self.mem_per_cpu = kwargs.get('mem_per_cpu')

        self.log_file = kwargs.get('log_file', None)

    @property
    def nproc_avail(self):
        if self.in_job_allocation:
            return int(os.environ['SLURM_NTASKS'])
        return self.MAX_NTASKS

    @property
    def nnode_avail(self):
        if self.in_job_allocation:
            return int(os.environ['SLURM_NNODES'])
        return self.MAX_NNODES

    @property
    def ppn_avail(self):
        if self.in_job_allocation:
            return int(os.environ['SLURM_TASKS_PER_NODE'].split('(')[0])
        return self.MAX_PPN

    @property
    def execute_command(self):
        if self.nproc == 1 or self.parallel_mode == 'serial':
            return ""
        if self.in_job_allocation:
            if self.parallel_mode == 'mpi':
                return f"srun -n {self.nproc} -N {self.nnode} -r {self.offset_node} --exact --unbuffered"
            elif self.parallel_mode == 'openmp':
                return f"export OMP_NUM_THREADS={self.nproc}; srun -N 1 -r {self.offset_node} -n 1 --cpus-per-task={self.nproc} --unbuffered"
            else:
                raise ValueError(f"unknown parallel_mode '{self.parallel_mode}'")
        else:
            return f"srun -n {self.nproc} --unbuffered"

    @property
    def job_array_index_name(self):
        return '$SLURM_ARRAY_TASK_ID'

    @property
    def in_job_allocation(self) -> bool:
        if 'SLURM_JOB_ID' in os.environ:
            return True
        return False

    def submit_job_and_monitor(self, commands):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                         dir=self.run_dir,
                                         prefix=self.job_name+'.',
                                         suffix='.sh') as job_script:
            job_script.write("#!/bin/bash\n")

            # slurm job header
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

            # Emit an exit-code sentinel on ANY shell exit (normal end, an explicit
            # 'exit', or an errored command) via a bash EXIT trap, so the monitor can
            # read the true exit status regardless of any scheduler completion epilog
            # (e.g. "Job <id> completed"), which is written even when the job failed
            # (issue #20). This is scheduler-agnostic and needs no accounting database.
            # A job killed abnormally (out of memory, timeout, node failure) cannot run
            # the trap and so leaves no sentinel, which is likewise treated as a failure.
            job_script.write('trap \'echo "NEDAS_JOB_EXIT_CODE=$?"\' EXIT\n')

            # add the commands
            commands = super().parse_commands(commands)
            job_script.write(commands)
            job_script.write('\n')

            self.job_script = job_script.name

        # submit the job script
        p = subprocess.run(['sbatch', self.job_script], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Failed to submit job: {p.stderr}")
        self.job_id = int(p.stdout.split()[-1])

        # Determine which file to stream to the tty. If the caller redirected the job's
        # runtime output to a specific file (passed in via log_file), stream that file.
        # Otherwise fall back to the scheduler's stdout file for this job.
        # Use a local variable so self.log_file (which may be a %j template) is never
        # mutated — avoids re-reading a previous job's log on the next run() call.
        _log_template = self.log_file if self.log_file is not None else log_file
        current_log_file = _log_template.replace('%j', str(self.job_id))

        if self.debug:
            print(f"JobSubmitter: job '{self.job_name}' submitted with ID {self.job_id} to SLURM scheduler", flush=True)

        # monitor job status
        file_pointer = 0
        if self.use_job_array:
            while True:
                sleep(self.check_dt)
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
                sleep(self.check_dt)
                # query state (%t) and the pending reason (%r) explicitly; the
                # reason can contain spaces (e.g. "launch failed requeued held")
                # so it is read as a single trailing field rather than by index
                p = subprocess.run(['squeue', '-h', '-j', f'{self.job_id}', '-o', '%t|%r'],
                                    capture_output=True, text=True)
                if not p.stdout.strip():
                    # job no longer in queue
                    break
                job_status, _, job_reason = p.stdout.strip().partition('|')
                job_reason = job_reason.strip()
                if job_status not in ['R', 'PD', 'CG']:
                    # job not running, pending, or cleaning up
                    raise RuntimeError(f"job {self.job_name} failed with status {job_status}")

                if job_status == 'PD':  # job is pending in queue
                    # a held / un-schedulable job stays pending forever; abort instead of waiting
                    if 'held' in job_reason.lower() or job_reason in self.PENDING_FAILURE_REASONS:
                        # cancel it so we don't leave an orphan sitting in the queue (issue #20)
                        subprocess.run(['scancel', str(self.job_id)])
                        raise RuntimeError(f"job {self.job_name} stuck pending and will not start "
                                           f"(reason: {job_reason})")
                    continue  # transient pending reason, keep waiting

                # stream new log output to the tty, if a log file is available
                if not os.path.exists(current_log_file):
                    continue

                # open log file and seek to the last position
                with open(current_log_file, 'r', newline='') as f:
                    f.seek(file_pointer)
                    new_content = f.read()

                    if new_content:
                        print(new_content, end='', flush=True)  # stream the new content to tty
                        file_pointer = f.tell()  # update file pointer to the new position

        # flush any log content written between the last poll and the job leaving the queue
        if os.path.exists(current_log_file):
            with open(current_log_file, 'r', newline='') as f:
                f.seek(file_pointer)
                tail = f.read()
            if tail:
                print(tail, end='', flush=True)

        if self.debug:
            print(f"JobSubmitter: job '{self.job_name}' finished", flush=True)

        # verify the job actually succeeded and report errors
        if self.use_job_array:
            for i in range(self.array_size):
                log_file = os.path.join(self.run_dir, f"{self.job_name}-{self.job_id}_{i}.out")
                self._check_job_outcome(log_file)
        else:
            log_file = os.path.join(self.run_dir, f"{self.job_name}-{self.job_id}.out")
            self._check_job_outcome(log_file)

    def _check_job_outcome(self, log_file):
        """Verify the job succeeded, using the exit-code sentinel its script appended.

        The job script records "NEDAS_JOB_EXIT_CODE=<rc>" as its final action, so the
        sentinel reflects the real exit status regardless of any scheduler completion
        epilog (issue #20). A job killed abnormally (out of memory, timeout, node
        failure) never reaches that line, so a missing sentinel is also a failure.
        """
        elapsed = 0
        # the .out may not be flushed yet right after the job leaves the queue; poll
        rc = self._read_exit_code(log_file)
        while rc is None:
            if elapsed >= self.EXIT_CODE_TIMEOUT:
                raise RuntimeError(f"job {self.job_name} did not finish cleanly "
                                   f"(no exit status recorded, likely killed), check {log_file}")
            sleep(1)
            elapsed += 1
            rc = self._read_exit_code(log_file)
        if rc != 0:
            raise RuntimeError(f"job {self.job_name} failed with exit code {rc}, check {log_file}")

    @staticmethod
    def _read_exit_code(log_file):
        """Return the exit code from the job script's sentinel line, or None if absent."""
        if not os.path.exists(log_file):
            return None
        prefix = 'NEDAS_JOB_EXIT_CODE='
        codes = []
        with open(log_file) as f:
            for line in f:
                if line.startswith(prefix):
                    codes.append(line[len(prefix):].strip())
        if not codes:
            return None
        try:
            return int(codes[-1])  # last occurrence, in case the file is reused across retries
        except ValueError:
            return None
