import os
import subprocess
from NEDAS.utils.job_submitters import get_job_submitter

def makedir(dirname):
    try:
        os.makedirs(dirname, exist_ok=True)
    except FileExistsError:
        ##can happen if multiple processor are trying to make the same directory
        pass

def run_command(shell_cmd):
    """
    Run a shell command in a subprocess, handle errors
    """
    p = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0:
        raise RuntimeError(f"{p.stderr}")

def run_job(commands:str, **kwargs):
     ##get the right job submitter
     job_submitter = get_job_submitter(**kwargs)

     ##run job using the submitter
     job_submitter.run(commands)

