import os
import sys
import tempfile
import inspect
import importlib.util
import subprocess
from typing import Callable
from abc import ABC, abstractmethod
from NEDAS.utils.parallel import Scheduler
from NEDAS.config import Config
from NEDAS.core.context import Context

class Scheme(ABC):
    """
    Runtime scheme base class.

    The Scheme coordinates all runtime generation and manipulation of objects.
    """
    config: Config
    online_mode: bool
    steps: dict[str, dict[str, bool|str]] = {}
    _context: Context|None = None

    def __init__(self, config_file: str|None=None,
                 parse_args: bool=False,
                 **kwargs) -> None:
        # parse configuration
        self.config = Config(config_file=config_file, parse_args=parse_args, **kwargs)

        self.online_mode = (self.config.io_mode == 'online')

    @property
    def c(self):
        """Lazy initialization of the runtime context. """
        if self._context is None:
            self._context = Context(self.config)
        return self._context

    def __call__(self) -> None:
        """
        The entry point that handles the environment check and starts the engine.
        """
        # Environment check:
        # If we are in online io mode, check if mpi env is ready if nproc>1
        # if not, we will dispatch the whole scheme itself to a job submitter run.
        if self.online_mode and self.config.nproc > 1:
            # check if the mpi environment is ready
            if self.c.comm.Get_size() != self.config.nproc:
                self.c.print_1p("need to elevate self to a mpi env \n")
                self.c.run_job
                return

        # if we are in offline io mode, each step in run_all will decide how to dispatch itself
        self.run_all()

    @abstractmethod
    def run_all(self):
        """
        A schemem must implement a run_all method to describe the workflow.
        """
        ...

    def run_step(self, step: str) -> None:
        """
        Helper function to run a step from an external call.
        """
        if step not in self.steps:
            raise ValueError(f"Unknown step '{step}' for {self.__class__.__name__}")

        mpi = self.steps[step]['mpi']

        assert self.c.config.io_mode == 'offline'
        script_file = os.path.abspath(inspect.getfile(self.__class__))

        # create a temporary config yaml file to hold c, and pass into program through runtime arg
        with tempfile.NamedTemporaryFile(dir=self.c.config.work_dir, prefix='config-', suffix='.yml') as tmp_config_file:
            self.c.dump_config(tmp_config_file.name)

            print(f"\n\033[1;33mRUNNING\033[0m {step} step")
            if self.c.config.debug:
                print(f"config file: {tmp_config_file.name}")

            ##build run commands for the ensemble forecast script
            commands = ""
            if self.c.config.python_env:
                commands = f". {self.c.config.python_env}; "
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
                    'run_dir': self.c.fs.cycle_dir(self.c.time),
                    'nproc': self.c.config.nproc,
                    'debug': self.c.config.debug,
                    **(self.c.config.job_submit or {}),
                    }
                self.c.run_job(commands, **job_opts)
            else:
                p = subprocess.Popen(commands, shell=True, text=True)
                p.wait()
                if p.returncode != 0:
                    print(f"{self.__class__.__name__}: run_step '{step}' exited with error")
                    sys.exit(1)

    def run_ensemble_tasks_in_mpi(self):
        ...

    def run_ensemble_tasks_in_scheduler(self, task_name: str, func: Callable, *args, **kwargs) -> None:
        walltime = kwargs.get('walltime', None)
        nproc_per_run = kwargs['nproc_per_run']

        ##get number of workers to initialize the scheduler
        # if c.jsub.in_job_allocation
        if self.c.config.job_submit and self.c.config.job_submit.get('run_separate_jobs', False):
            ##all jobs will be submitted to external scheduler's queue
            ##just assign a worker to each ensemble member
            nworker = min(self.c.config.nens, self.c.config.nproc_util)
        else:
            ##Scheduler will use nworkers to spawn ensemble member runs to
            ##the available nproc processors
            nworker = self.c.config.nproc // nproc_per_run
        scheduler = Scheduler(nworker, walltime, debug=self.c.config.debug)

        for mem_id in range(self.c.config.nens):
            scheduler.submit_job(f"{task_name}_mem{mem_id+1:03}", func, *args, member=mem_id, **kwargs)

        scheduler.start_queue() ##start the job queue
        scheduler.shutdown()
        self.c.print_1p(' done.\n')