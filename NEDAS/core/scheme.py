import os
import sys
import tempfile
import importlib.util
import subprocess

from abc import ABC, abstractmethod
from NEDAS import config
from .context import Context
from NEDAS.runtimes.offline import OfflineRuntime

class Scheme(ABC):
    """
    Runtime scheme base class.

    The Scheme coordinates all runtime generation and manipulation of objects.
    """
    c: Context

    def __init__(self, config: config.Config) -> None:
        # parse configuration
        self.c = Context(config)

    @abstractmethod
    def __call__(self) -> None:
        """
        A runtime scheme must have a __call__ method.
        """
        ...

    def run_step(self, step, mpi) -> None:
        """
        Helper function to run this script (``forecast.py``) from an external call.

        Args:
            c (Config): Configuration.
            step (str): Step to run.
            mpi (bool): Whether to run the step within mpi environment.
        """
        c = self.c

        assert isinstance(c.runtime, OfflineRuntime)
        script_file = os.path.abspath(__file__)

        # create a temporary config yaml file to hold c, and pass into program through runtime arg
        with tempfile.NamedTemporaryFile(dir=c.config.work_dir,
                                         prefix='config-',
                                         suffix='.yml') as tmp_config_file:
            c.config.dump_yaml(tmp_config_file.name)

            print(f"\n\033[1;33mRUNNING\033[0m {step} step")
            if c.config.debug:
                print(f"config file: {tmp_config_file.name}")

            ##build run commands for the ensemble forecast script
            commands = ""
            if c.config.python_env:
                commands = f". {c.config.python_env}; "
            if mpi:
                if importlib.util.find_spec("mpi4py") is not None:
                    commands += f"JOB_EXECUTE {sys.executable} {script_file} -c {tmp_config_file.name}"
                else:
                    print("Warning: mpi4py is not found, will try to run with nproc=1.", flush=True)
                    commands += f"{sys.executable} {script_file} -c {tmp_config_file.name} --nproc=1"
            else:
                commands += f"{sys.executable} {script_file} -c {tmp_config_file.name}"
            commands += f" --step {step}"

            if mpi:
                job_opts = {
                    'job_name': step,
                    'run_dir': c.runtime.cycle_dir(c.time),
                    'nproc': c.config.nproc,
                    'debug': c.config.debug,
                    **(c.config.job_submit or {}),
                    }
                self.c.runtime.run_job(commands, **job_opts)
            else:
                p = subprocess.Popen(commands, shell=True, text=True)
                p.wait()
                if p.returncode != 0:
                    print(f"ForecastScheme: run_step '{step}' exited with error")
                    sys.exit(1)
