import os
import sys
import subprocess
import importlib.util

def run_command(shell_cmd):
    """
    Run a shell command in a subprocess, handle errors
    """
    p = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0:
        raise RuntimeError(f"{p.stderr}")

def run_script(script_path, c):
    """
    Run a script at script_path, in a subprocess
    If there is MPI environment run the script in parallel, if not run in serial
    Pass configuration object c through runtime argument -c config.yml
    """
    ##dump config c content in a file
    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    if importlib.util.find_spec("mpi4py") is not None:
        shell_cmd = f"{c.job_submit_cmd} {c.nproc} 0 python -m mpi4py {script_path} -c {config_file}"
    else:
        print("Warning: mpi4py not found in your environment, will try to run with nproc=1.")
        shell_cmd = f"python {script_path} -c {config_file} --nproc=1"

    if c.debug:
        print(shell_cmd)

    p = subprocess.Popen(shell_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    p.wait()

    ##handle error
    if p.returncode != 0:
        print(f"{script_path} raised error: {p.stderr}")
        exit()

