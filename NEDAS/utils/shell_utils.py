import os
import subprocess
from NEDAS.job_submitters import get_job_submitter

def makedir(dirname:str) -> None:
    """
    Create a directory if it does not exist.
 
    FileExistsError can happen if multiple processors are trying to make the same directory.
    This function will ignore this error and continue.

    Args:
        dirname (str): Directory name to be created.
    """
    try:
        os.makedirs(dirname, exist_ok=True)
    except FileExistsError:
        pass

def run_command(shell_cmd:str) -> None:
    """
    Run a shell command in a subprocess and check for errors.

    Args:
        shell_cmd (str): Shell command to be executed.

    Raises:
        RuntimeError: If the command returns a non-zero exit status.
    """
    p = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0:
        raise RuntimeError(f"{p.stderr}")

def run_job(commands:str, **kwargs):
     """
     Run a shell command by submitting it to a job scheduler using JobSubmitter class.
     
     Args:
        commands (str): Shell command to be executed.
        **kwargs: Key-value pairs to passed to the job submitter run method.
     """
     ##get the right job submitter
     job_submitter = get_job_submitter(**kwargs)

     ##run job using the submitter
     job_submitter.run(commands)
