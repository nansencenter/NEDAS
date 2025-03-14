import os
import sys
import importlib.util
from utils.shell_utils import run_job
from assim_tools.analysis_scheme import AnalysisScheme

def run(c):
    """
    Run the analysis.py script with specified job submit options at runtime
    """
    script_file = os.path.abspath(__file__)
    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    print(f"\033[1;33mRUNNING\033[0m {script_file}")

    ##build run commands for the perturb script
    commands = f"source {c.python_env}; "

    if importlib.util.find_spec("mpi4py") is not None:
        commands += f"JOB_EXECUTE {sys.executable} -m mpi4py {script_file} -c {config_file}"
    else:
        print("Warning: mpi4py is not found, will try to run with nproc=1.", flush=True)
        commands += f"{sys.executable} {script_file} -c {config_file} --nproc=1"

    job_submit_opts = {}
    if c.job_submit:
        job_submit_opts = c.job_submit

    if hasattr(c, 'ppn'):
        job_submit_opts['ppn'] = c.ppn

    run_job(commands, job_name='analysis', run_dir=c.cycle_dir(c.time), nproc=c.nproc, **job_submit_opts)

if __name__ == '__main__':
    from config import Config
    c = Config(parse_args=True)
    
    analysis = AnalysisScheme()
    analysis(c)