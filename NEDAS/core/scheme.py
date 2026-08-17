import os
import sys
import signal
import tempfile
import inspect
import numpy as np
from typing import Callable
from abc import ABC, abstractmethod
from NEDAS.job_submitters.hpc import HPCJobSubmitter
from NEDAS.utils.parallel import OfflineScheduler
from NEDAS.utils.njit import njit
from NEDAS.datasets.synthetic import SyntheticObs
from NEDAS.config import Config
from NEDAS.core.context import Context
from NEDAS.core.types import EnsRunStrategy

@njit
def _seed_numba_rng(seed: int) -> None:
    """Seed numba's own internal RNG stream used inside @njit functions (e.g.
    ETKF's random_orthogonal_matrix) -- it is separate from numpy's global
    RNG, so np.random.seed() alone does not make jitted random draws
    reproducible."""
    np.random.seed(seed)

class Scheme(ABC):
    """
    Runtime scheme base class.

    The Scheme coordinates all runtime generation and manipulation of objects.
    """
    config: Config
    online_mode: bool
    cycling: bool
    use_synthetic_obs: bool = False
    steps_need_mpi: dict[str, bool] = {}
    _context: Context|None = None
    scheduler: OfflineScheduler|None = None

    def __init__(self, config: Config|None=None,
                 config_file: str|None=None,
                 parse_args: bool=False,
                 **kwargs) -> None:
        # parse configuration and generate context
        if config:
            self.config = config
        else:
            self.config = Config(config_file=config_file, parse_args=parse_args, **kwargs)

        # seed both numpy's global RNG (truth/ensemble IC, synthetic obs, ...) and numba's
        # separate internal RNG (ETKF random rotation, other @njit functions) for a fully
        # reproducible run; if unset, leave both RNGs on their current (random) state
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            _seed_numba_rng(self.config.seed)

        self.c = Context(self.config)

        # check if io mode is online:
        self.online_mode = (self.config.io_mode == 'online')

        # resolve cycling mode: if not set explicitly, run_forecast supersedes
        # (run_forecast=True implies cycling=True)
        if self.config.cycling is not None:
            self.cycling = self.config.cycling
        else:
            self.cycling = self.config.run_forecast

        # check if one or more of the datasets is synthetic type:
        for dataset in self.c.datasets.values():
            if isinstance(dataset, SyntheticObs):
                self.use_synthetic_obs = True

        self._main_pid = os.getpid()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._handle_exit_signal)

    def _handle_exit_signal(self, signum, frame):
        """ The 'trap' handler. This runs when the OS sends kill signal. """
        # only the main process should run the shutdown procedure
        if os.getpid() != self._main_pid:
            return

        self.c.log_event(f"{self.__class__.__name__}: Received signal {signum}. Cleaning up...", flag='info')

        # 1. kill the scheduler worker if it exists
        if self.scheduler:
            worker_pids = list(self.scheduler.executor._processes.keys())
            self.c.log_event(f"OfflineScheduler: Cleaning up worker processes {worker_pids}", flag='info')

            for pid in worker_pids:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    pass

        # 2. kill the entire main process group
        try:
            os.killpg(0, signal.SIGKILL)
        except Exception:
            pass

    @property
    def c(self):
        """ The runtime context, with lazy initialization """
        if self._context is None:
            self._context = Context(self.config)
        return self._context

    @c.setter
    def c(self, value: Context):
        """ Allows refreshing the context from subprocess returns """
        self._context = value

    def __call__(self) -> None:
        """
        The entry point that handles the environment check and starts the engine.
        """
        self.c.show_greeting()

        self.c.print_1p(f"\nINITIALIZING...\n")
        self.c.show_summary()

        # Environment check:
        # 1. Online mode: (requires mpi environment if nproc>1)
        if self.online_mode:
            if self.c.comm.mpi_ready or self.config.nproc==1:
                # we are already inside the mpi environment, proceed
                self.c.logger(self.__class__.__name__)(self.run_all)()
            else:
                # if not, we will dispatch the whole scheme itself to a job submitter.
                if self.c.debug:
                    self.c.log_event(f"run_all: config.nproc={self.config.nproc}, elevating to a mpi-enabled environment...", flag='info')
                self.external_call(step='run_all', parallel_mode='mpi', nproc=self.config.nproc)

        # 2. offline mode (manual dispatch per step)
        else:
            # check if we are accidentally inside an mpi environment already
            comm_size = self.c.comm.Get_size()
            if self.c.comm.mpi_ready and comm_size>1:
                raise RuntimeError(f"Running in offline mode, but an mpi environment comm.size={comm_size} is detected."
                                   "The main program should be run in serial.")

            # in offline io mode, each step in run_all will decide how to dispatch itself
            self.c.logger(self.__class__.__name__)(self.run_all)()

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
                self.c.log_event(f"config file: {tmp_config_file.name}", flag='info')

            # build run commands for the ensemble forecast script
            commands = ""
            if self.config.python_env:
                commands = f". {self.config.python_env}; "
            commands += f"JOB_EXECUTE {sys.executable} {script_file} -c {tmp_config_file.name}"
            if step:
                commands += f" --step {step}"
            if self.config.debug:
                self.c.log_event(f"running commands: '{commands}'", flag='info')

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
                    self.c.log_event(f"{step}: config.nproc={self.config.nproc}, elevating to a mpi-enabled environment...", flag='info')
                self.external_call(step, parallel_mode='mpi', nproc=self.config.nproc)
                return

        # otherwise, just call the step func
        stepfunc = getattr(self, step)
        if step == 'run_all':
            func_name = self.__class__.__name__
        else:
            func_name = f'Running {step} step'
        self.c.logger(func_name)(stepfunc)()

    def run_ensemble_tasks(self, strategy: EnsRunStrategy,
                           tag: str,
                           task_name: str,
                           func: Callable,
                           **opts) -> None:
        """
        Dispatch func(...) once per ensemble member, according to strategy:

        - 'batch': func handles the whole ensemble itself in one call
          (see _run_ensemble_tasks_batch).
        - 'scheduler', online_mode: each rank runs func directly for the
          members it owns (c.mem_list), no external job is submitted
          (see _run_ensemble_tasks_online).
        - 'scheduler', offline mode: each member is submitted as a job
          managed by a local OfflineScheduler (see
          _run_ensemble_tasks_offline_scheduler), whose worker-pool size
          (nworker) depends on opts['nproc'] (nproc_per_task, the size of
          each member's job) and opts['total_nproc'] (the budget nworker is
          carved out of) via one of two branches:

          1. nproc_per_task > 1, jsub is an HPCJobSubmitter, and we are not
             already inside a job allocation: each member becomes its own
             separate HPC job (submitted via sbatch); nproc_per_task is the
             *cluster* job size, not a local CPU count, so total_nproc is
             never consulted here -- nworker is opts['max_concurrent']
             (default nens), and the HPC scheduler manages actual
             concurrency once the jobs are queued.

          2. every other case: nworker = total_nproc // nproc_per_task,
             sized as a *local* multiprocessing.ProcessPoolExecutor pool
             (see OfflineScheduler) on the driving process's own node. This
             branch is reached three ways, and total_nproc means something
             different (but is always a real, physical concurrency bound)
             in each:
               a. no HPC scheduler at all (LocalJobSubmitter/MacOSJobSubmitter)
                  -- nworker literal local subprocesses each use
                  nproc_per_task real local cores, so total_nproc must be
                  the real local core budget.
               b. nproc_per_task > 1 but already inside a job allocation --
                  each member runs as a job *step* (srun -n nproc_per_task,
                  see HPCJobSubmitter.run_job_as_step) sharing the current
                  allocation's granted resources, so total_nproc must be
                  that allocation's real size (nworker * nproc_per_task must
                  not exceed what was actually granted).
               c. nproc_per_task == 1 and HPC-but-not-yet-in-an-allocation
                  (e.g. a login-node driver submitting single-proc member
                  jobs before entering any allocation) -- each worker here
                  is lightweight (just sbatch + squeue polling; the real
                  compute happens in the separately-scheduled job), so
                  total_nproc functions as a submission-concurrency
                  throttle rather than a core count. nproc_util would also
                  be defensible here, but nproc is never wrong -- it just
                  makes the local orchestration pool bigger than strictly
                  necessary, which SLURM itself still throttles downstream.

          Because of (1)/(2a)/(2b), total_nproc should be the full
          production nproc budget (paired with nproc_per_run) for
          compute-heavy per-member steps like prepare_init_ensemble and
          ensemble_forecast, not nproc_util (which is for the deliberately
          separate, often smaller, nproc_per_util utility-task budget used
          by preprocess/postprocess) -- see FilterAnalysisScheme's calls to
          get_task_opts for those steps, and issue #26.
        """
        if strategy == 'batch':
            self._run_ensemble_tasks_batch(tag, task_name, func, **opts)

        elif strategy == 'scheduler':
            if self.online_mode:
                self._run_ensemble_tasks_online(tag, task_name, func, **opts)
            else:
                self._run_ensemble_tasks_offline_scheduler(tag, task_name, func, **opts)
        else:
            raise ValueError(f"Unknown ensemble run strategy '{strategy}'")

    def _run_ensemble_tasks_batch(self, tag: str, task_name: str, func: Callable, **opts) -> None:
        # the func should handle the entire ensemble in one go
        # make sure nens is defined in opts
        self.c.debug_message = f"running {task_name} in batch mode..."
        opts['nens'] = self.c.nens
        self.c.io.call_method(self.c, tag, func, **opts)

    def _run_ensemble_tasks_online(self, tag: str, task_name: str, func: Callable, **opts) -> None:
        # scheduling internally within mpi environment
        # using the mem_list (member lists distributed on pid ranks by comm)
        nm = len(self.c.mem_list[self.c.pid_mem])
        self.c.total_tasks = nm
        for m, mem_id in enumerate(self.c.mem_list[self.c.pid_mem]):
            opts['member'] = mem_id
            self.c.debug_message = f"running {task_name} for mem{mem_id+1:03}"
            self.c.current_task = m
            self.c.io.call_method(self.c, tag, func, **opts)
        self.c.comm.Barrier()

    def _run_ensemble_tasks_offline_scheduler(self, tag: str, task_name: str, func: Callable, **opts) -> None:
        nproc_per_task = opts.get('nproc', 1)

        if nproc_per_task > 1 and isinstance(self.c.jsub, HPCJobSubmitter) and not self.c.jsub.in_job_allocation:
            # Each member becomes a separate HPC job; nproc_per_task is the job size on the cluster,
            # not a local CPU count.  Submit all nens members at once and let the HPC scheduler
            # manage actual concurrency — throttle with opts['max_concurrent'] if needed.
            nworker = opts.get('max_concurrent', self.c.nens)
        else:
            # Local or in-allocation mode: bound concurrency by available processors.
            total_nproc = opts.get('total_nproc', self.config.nproc)
            assert total_nproc >= nproc_per_task, (
                f"requested nproc ({nproc_per_task}) exceeds available total_nproc ({total_nproc})"
            )
            nworker = max(1, total_nproc // nproc_per_task)

        # initialize the scheduler
        self.c.debug_message = f"running {task_name} in offline scheduler: nworker={nworker}"
        self.scheduler = OfflineScheduler(self.c, nworker, opts.get('walltime'), debug=self.config.debug)

        # submit jobs
        for mem_id in range(self.c.nens):
            job_opts = {
                **opts,
                'member': mem_id,
                'debug': self.config.debug,
            }
            self.scheduler.submit_job(f"{task_name}_mem{mem_id+1:03}", self.c.io.call_method, self.c, tag, func, **job_opts)

        try:
            self.scheduler.start_queue()
        finally:
            self.scheduler.shutdown()
            self.scheduler = None
            self.c.comm.Barrier()
