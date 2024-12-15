import os
import subprocess
import tempfile
from time import sleep
from utils.conversion import seconds_to_timestr
from .base import JobSubmitter

class OARJobSubmitter(JobSubmitter):
    """JobSubmitter Class customized for OAR schedulers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ##temporary node file
        if not self.run_separate_jobs:
            with tempfile.NamedTemporaryFile(delete=False) as file:
                self.node_file = file.name

        ##host specific settings
        if self.host == 'gricad':
            ##job submit node based on queue type
            if self.queue == 'devel':
                self.job_submit_node = 'oar-dahu3'
            else:
                self.job_submit_node = 'f-dahu'

    @property
    def nproc_avail(self):
        return self.nnode_avail * self.ppn

    @property
    def node_list_avail(self):
        try:
            with open(os.environ['OAR_NODE_FILE'], 'r') as node_file:
                node_list = [node for node in node_file]
        except Exception as e:
            print("ERROR while reading available compute nodes from OAR_NODE_FILE")
            raise e
        return node_list

    @property
    def nnode_avail(self):
        return len(self.node_list_avail)

    @property
    def ppn_avail(self):
        ##TODO: how to obtain this info from OAR system?
        return 32

    def update_node_file(self):
        with open(self.node_file, 'w') as file:
            for i in range(self.nnode):
                node = self.node_list_avail[self.offset_node+i]
                file.write(node+'\n')

    @property
    def execute_command(self):
        if self.run_separate_jobs:
            node_file = '$OAR_NODE_FILE'
        else:
            self.update_node_file()
            node_file = self.node_file
        return f"mpirun -np {self.nproc} --machinefile {node_file}"

    @property
    def job_array_index_name(self):
        return '$OAR_ARRAY_INDEX'

    def run_job_as_step(self, commands):
        super().run_job_as_step(commands)
        ##clean up
        os.remove(self.node_file)

    def submit_job_and_monitor(self, commands):
        ##build the job script
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.sh') as job_script:
            job_script.write("#!/bin/bash\n")

            ##OAR headers
            job_script.write(f"#OAR -n {self.job_name}\n")
            job_script.write(f"#OAR -l /nodes={self.nnode}/core={self.ppn},walltime={seconds_to_timestr(self.walltime)}\n")
            job_script.write(f"#OAR --project {self.project}\n")
            job_script.write(f"#OAR -t {self.queue}\n")

            log_file = os.path.join(self.run_dir, f"{self.job_name}.%jobid%.stdout")
            job_script.write(f"#OAR --stdout {log_file}\n")
            job_script.write(f"#OAR --stderr {log_file}\n")

            if self.use_job_array:
                job_script.write(f"#OAR --array {self.array_size}\n")

            commands = super().parse_commands(commands)
            job_script.write(commands)
            job_script.write('\n')

            self.job_script = job_script.name

        ##submit the job script
        process = subprocess.run(['ssh', self.job_submit_node,
                                  f"oarsub -S {self.job_script}"],
                                  capture_output=True)
        s = process.stderr.decode('utf-8')
        print(s, flush=True)
        s = process.stdout.decode('utf-8')
        print(s, flush=True)

        ##monitor the queue for job completion
        if self.use_job_array:
            self.job_id = int(s.split('OAR_ARRAY_ID=')[-1])
            while True:
                sleep(self.check_dt)
                p = subprocess.run(['ssh', self.job_submit_node,
                                    'oarstat', '-f', f'--array {self.job_id}',
                                    '| grep "state = "'], capture_output=True)
                s = p.stdout.decode('utf-8').replace(' ', '').split('\n')[:-1]
                print(s, flush=True)
                s = [string for string in s if 'state=' in string]
                print(s, flush=True)
                jobs_status = [job.split('state=')[-1].replace(' ', '') for job in s]
                # end this loop if all jobs are terminated
                if all([status == 'Terminated' for status in jobs_status]):
                    break
                if all([status == 'Error' for status in jobs_status]):
                    raise RuntimeError(f"Error job array {self.job_id}")

        else:
            ##TODO: also add capture of normal job id and monitor here
            pass

