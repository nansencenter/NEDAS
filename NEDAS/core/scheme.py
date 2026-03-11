import os
import sys
import tempfile
import inspect
import importlib.util
import subprocess
from typing import Callable
from abc import ABC, abstractmethod
from NEDAS.utils.parallel import Scheduler
from NEDAS.runtimes.offline import OfflineRuntime
from NEDAS.config import Config
from NEDAS.core.context import Context

class Scheme(ABC):
    """
    Runtime scheme base class.

    The Scheme coordinates all runtime generation and manipulation of objects.
    """
    config: Config
    c: Context

    def __init__(self, config_file: str|None=None, parse_args: bool=False, **kwargs) -> None:
        # parse configuration
        ###don't init context yet: the program can be serial python call, yet the nproc >1 need to spawn a mpi call in run_step
        self.c = Context(config_file=config_file, parse_args=parse_args, **kwargs)
        self.config = self.c.config

    @abstractmethod
    def __call__(self) -> None:
        """
        A runtime scheme must have a __call__ method.
        """
        ...

    def run_step(self, step: str, mpi: bool) -> None:
        """
        Helper function to run a step in the from an external call.

        Args:
            step (str): Step to run.
            mpi (bool): Whether to run the step within mpi environment.
        """
        c = self.c

        assert isinstance(c.rt, OfflineRuntime)
        script_file = os.path.abspath(inspect.getfile(self.__class__))

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
                    'run_dir': c.rt.cycle_dir(c.time),
                    'nproc': c.config.nproc,
                    'debug': c.config.debug,
                    **(c.config.job_submit or {}),
                    }
                self.c.rt.run_job(commands, **job_opts)
            else:
                p = subprocess.Popen(commands, shell=True, text=True)
                p.wait()
                if p.returncode != 0:
                    print(f"{self.__class__.__name__}: run_step '{step}' exited with error")
                    sys.exit(1)

    def run_ensemble_tasks_in_mpi(self, c: Context, ):
        ...

    def run_ensemble_tasks_in_scheduler(self, c: Context, task_name: str, func: Callable, *args, **kwargs) -> None:
        walltime = kwargs.get('walltime', None)
        nproc_per_run = kwargs['nproc_per_run']

        ##get number of workers to initialize the scheduler
        if c.config.job_submit and c.config.job_submit.get('run_separate_jobs', False):
            ##all jobs will be submitted to external scheduler's queue
            ##just assign a worker to each ensemble member
            nworker = min(c.config.nens, c.config.nproc_util)
        else:
            ##Scheduler will use nworkers to spawn ensemble member runs to
            ##the available nproc processors
            nworker = c.config.nproc // nproc_per_run
        scheduler = Scheduler(nworker, walltime, debug=c.config.debug)

        for mem_id in range(c.config.nens):
            scheduler.submit_job(f"{task_name}_mem{mem_id+1:03}", func, *args, member=mem_id, **kwargs)

        scheduler.start_queue() ##start the job queue
        scheduler.shutdown()
        c.print_1p(' done.\n')