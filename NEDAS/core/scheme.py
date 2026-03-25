import os
import sys
import tempfile
import inspect
from typing import Callable
from abc import ABC, abstractmethod
from NEDAS.job_submitters.hpc import HPCJobSubmitter
from NEDAS.utils.parallel import OfflineScheduler
from NEDAS.datasets.synthetic import SyntheticObs
from NEDAS.config import Config
from NEDAS.core.context import Context
from NEDAS.core.types import EnsRunStrategy, IOTag

class Scheme(ABC):
    """
    Runtime scheme base class.

    The Scheme coordinates all runtime generation and manipulation of objects.
    """
    config: Config
    online_mode: bool
    use_synthetic_obs: bool = False
    steps_need_mpi: dict[str, bool] = {}
    _context: Context|None = None

    def __init__(self, config_file: str|None=None,
                 parse_args: bool=False,
                 **kwargs) -> None:
        # parse configuration
        self.config = Config(config_file=config_file, parse_args=parse_args, **kwargs)

        # check if io mode is online:
        self.online_mode = (self.config.io_mode == 'online')

        # check if one or more of the datasets is synthetic type:
        for dataset in self.c.datasets.values():
            if isinstance(dataset, SyntheticObs):
                self.use_synthetic_obs = True

    @property
    def c(self):
        """ The runtime context, with lazy initialization """
        if self._context is None:
            self._context = Context(self.config)
        return self._context

    def __call__(self) -> None:
        """
        The entry point that handles the environment check and starts the engine.
        """
        self.c.show_greeting()

        # Environment check:
        # 1. Online mode: (requires mpi environment if nproc>1)
        if self.online_mode:
            if self.c.comm.mpi_ready or self.config.nproc==1:
                # we are already inside the mpi environment, proceed
                self.run_all()
            else:
                # if not, we will dispatch the whole scheme itself to a job submitter.
                if self.c.debug:
                    self.c.print_1p(f"\nrun_all: config.nproc={self.config.nproc}, elevating to a mpi-enabled environment...\n")
                self.external_call(step='run_all', parallel_mode='mpi', nproc=self.config.nproc)

        # 2. offline mode (manual dispatch per step)
        else:
            # check if we are accidentally inside an mpi environment already
            comm_size = self.c.comm.Get_size()
            if self.c.comm.mpi_ready and comm_size>1:
                raise RuntimeError(f"Running in offline mode, but an mpi environment comm.size={comm_size} is detected."
                                   "The main program should be run in serial.")

            # in offline io mode, each step in run_all will decide how to dispatch itself
            self.run_all()

    @abstractmethod
    def run_all(self):
        """
        A schemem must implement a run_all method to describe the workflow.
        """
        ...

    def external_call(self, step:str|None=None, **kwargs):
        """
        Run the scheme from an external call.
        Saving the current context to a temporary config file, then run a subprocess to 
        """
        script_file = os.path.abspath(inspect.getfile(self.__class__))

        # create a temporary config yaml file to hold c, and pass into program through runtime arg
        with tempfile.NamedTemporaryFile(dir=self.config.work_dir,
                                         prefix='config-',
                                         suffix='.yml') as tmp_config_file:
            self.c.dump_config(tmp_config_file.name)

            if self.config.debug:
                print(f"config file: {tmp_config_file.name}")

            # build run commands for the ensemble forecast script
            commands = ""
            if self.config.python_env:
                commands = f". {self.config.python_env}; "
            commands += f"JOB_EXECUTE {sys.executable} {script_file} -c {tmp_config_file.name}"
            if step:
                commands += f" --step {step}"
            if self.config.debug:
                print(f"running commands: '{commands}'")

            # build job options
            job_opts = {
                **(self.config.job_submit or {}),
                'job_name': step,
                'run_dir': self.c.fs.cycle_dir(self.c.time),
                'nproc': self.config.nproc,
                'debug': self.config.debug,
                **kwargs,
            }
            # run job
            self.c.run_job(commands, **job_opts)

    def run_step(self, step: str) -> None:
        """
        Manages how to run a specified step in the workflow.
        """
        if not hasattr(self, step):
            raise NotImplementedError(f"Step '{step}' is not implemented for {self.__class__.__name__}")

        # in offline mode, run_step starts in serial
        # if the step requires mpi for nproc>1, make an external call
        if not self.online_mode and self.steps_need_mpi[step]:
            if self.config.nproc>1 and not self.c.comm.mpi_ready:
                if self.c.debug:
                    self.c.print_1p(f"\n{step}: config.nproc={self.config.nproc}, elevating to a mpi-enabled environment...\n")
                self.external_call(step, parallel_mode='mpi', nproc=self.config.nproc)
                return

        # otherwise, just call the step func
        stepfunc = getattr(self, step)
        self.c.logger(f'\033[1;33mRUNNING\033[0m {step} step')(stepfunc)()

    def run_ensemble_tasks(self, strategy: EnsRunStrategy,
                           tag: IOTag,
                           task_name: str,
                           func: Callable,
                           **opts) -> None:
        if strategy == 'batch':
            self._run_ensemble_tasks_batch(tag, task_name, func, **opts)

        elif strategy == 'scheduler':
            if self.online_mode:
                self._run_ensemble_tasks_online(tag, task_name, func, **opts)
            else:
                self._run_ensemble_tasks_offline_scheduler(tag, task_name, func, **opts)
        else:
            raise ValueError(f"Unknown ensemble run strategy '{strategy}'")

    def _run_ensemble_tasks_batch(self, tag: IOTag, task_name: str, func: Callable, **opts) -> None:
        # the func should handle the entire ensemble in one go
        # make sure nens is defined in opts
        self.c.debug_message = f"running {task_name} in batch mode..."
        opts['nens'] = self.c.nens
        self.c.io.call_method(self.c, tag, func, **opts)

    def _run_ensemble_tasks_online(self, tag: IOTag, task_name: str, func: Callable, **opts) -> None:
        # scheduling internally within mpi environment
        # using the mem_list (member lists distributed on pid ranks by comm)
        nm = len(self.c.mem_list[self.c.pid_mem])
        self.c.total_tasks = nm
        for m, mem_id in enumerate(self.c.mem_list[self.c.pid_mem]):
            opts['member'] = mem_id
            self.c.debug_message = f"running {task_name} for mem{mem_id+1:03}"
            self.c.current_task = m
            self.c.io.call_method(self.c, tag, func, **opts)

    def _run_ensemble_tasks_offline_scheduler(self, tag: IOTag, task_name: str, func: Callable, **opts) -> None:
        # setup an offline scheduler to distribute tasks
        # get number of available workers to initialize the scheduler
        total_nproc = opts.get('total_nproc', self.config.nproc)
        nworker = total_nproc // opts['nproc']

        if isinstance(self.c.jsub, HPCJobSubmitter) and not self.c.jsub.in_job_allocation:
            # the scheduling is then delegated to HPC's scheduler (each task submitted as a separate job)
            # here, the offline scheduler should just submit all tasks at once
            nworker = self.c.nens

        # initialize the scheduler
        scheduler = OfflineScheduler(self.c, nworker, opts.get('walltime'), debug=self.config.debug)

        # submit jobs
        for mem_id in range(self.c.nens):
            job_opts = {
                **opts,
                'member': mem_id,
                'debug': self.config.debug,
            }
            scheduler.submit_job(f"{task_name}_mem{mem_id+1:03}", self.c.io.call_method, self.c, tag, func, **job_opts)
        scheduler.start_queue() ##start the job queue
        scheduler.shutdown()
