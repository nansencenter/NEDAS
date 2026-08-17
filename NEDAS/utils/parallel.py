import os
import sys
from functools import wraps
from typing import Any, TypeVar, Callable, ParamSpec, Sequence
import time
from concurrent.futures import ProcessPoolExecutor, process
import threading
import traceback
import numpy as np

class Comm:
    """
    Communicator class supporting both serial and MPI programs.

    When the python program is started with MPI environment, for example::

    $ mpirun -n 10 python -m mpi4py program.py

    A communicator can be obtained from the mpi4py package:

    >>> from mpi4py import MPI
    >>> comm = MPI.COMM_WORLD

    However, when the program is run in

    Attributes:
        parallel_io (bool): If netCDF4.Dataset is built with parallel I/O support.

    """
    parallel_io: bool
    mpi_ready: bool = False

    def __init__(self):
        # detect if mpi environment exists
        # possible environ variable names from mpi calls
        mpi_env_var = ('PMI_SIZE', 'OMPI_UNIVERSE_SIZE')
        if any([ev in os.environ for ev in mpi_env_var]):
            # program is called from mpi, initialize comm
            try:
                from mpi4py import MPI   #type: ignore
                self._MPI = MPI
                self._comm = MPI.COMM_WORLD
                self.mpi_ready = True

            except ImportError:
                print("Warning: MPI environment found but 'mpi4py' module is not installed. Falling back to serial program for now.", flush=True)
                self._MPI = None
                self._comm = DummyComm()

        else:
            # serial program, use a dummy communicator
            self._MPI = None
            self._comm = DummyComm()

        self.parallel_io = self.check_parallel_io()

        # File lock to ensure only one process accesses a file at a time,
        # implemented as a point-to-point handoff chain -- NOT RMA
        # (Win.Create/Fetch_and_op via passive-target Win.Lock/Unlock was
        # tried first and found to fail outright, not just slowly, on Cray
        # MPICH+OFI at nproc=1024 on Olivia -- confirmed independent of
        # window count, so it's a stack-level issue, not something fixable
        # by tuning NEDAS's usage of it).
        #
        # For each locked filename, every rank that registers it (via
        # init_file_lock) is placed in a fixed, globally-agreed order (see
        # build_file_locks); a rank waits for a handoff message from its
        # predecessor in that order before proceeding, and passes the
        # handoff on to its successor once it is fully done with the file
        # for this generation. This assumes each rank's use of a given
        # file's lock is one contiguous block relative to other ranks --
        # true for NEDAS's actual usage (a file always belongs to exactly
        # one owning rank, e.g. one ensemble member per rank, so in
        # practice the chain length is 1 and no messages are exchanged at
        # all) -- not a fully general re-entrant mutex for arbitrary
        # interleaved acquire/release ordering across ranks.
        #
        # self._locks: filenames this rank participates in this generation
        #   (registered via init_file_lock, then built by build_file_locks).
        # self._lock_pred / self._lock_succ: filename -> rank id of the
        #   writer immediately before/after this rank in the fixed order,
        #   or None if this rank is first/last for that file.
        # self._lock_tag_map: filename -> a small unique int, used as the
        #   MPI tag for that file's handoff messages.
        # self._lock_synced: filenames for which this rank has already
        #   waited on its predecessor this generation (so repeated
        #   acquire_file_lock calls for the same file don't wait twice).
        # self._pending_lock_files: filenames registered via
        #   init_file_lock() since the last build_file_locks()/
        #   cleanup_file_locks().
        self._locks = set()
        self._lock_pred = {}
        self._lock_succ = {}
        self._lock_tag_map = {}
        self._lock_synced = set()
        self._pending_lock_files = []

    def __getattr__(self, attr):
        if attr == '_comm':
            raise AttributeError("_comm not initialized")
        comm = self.__dict__.get('_comm')
        if comm is not None and hasattr(comm, attr):
            return getattr(comm, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def init_file_lock(self, filename):
        """
        Register a filename that THIS rank will personally acquire/release
        the lock for. Actual chain construction is deferred to
        build_file_locks(), which must be called collectively afterward
        (once every rank has registered its own files) and before any
        acquire_file_lock()/release_file_lock() call.

        Unlike the old RMA design, callers should register only the files
        each rank itself intends to write -- NOT the global union of every
        file across all ranks (build_file_locks() does its own internal
        allgather to reconstruct the full per-file writer ordering).

        Args:
            filename (str): Path to the file.
        """
        if self._MPI is None or isinstance(self._comm, DummyComm) or not filename:
            return
        if filename not in self._locks and filename not in self._pending_lock_files:
            self._pending_lock_files.append(filename)

    def build_file_locks(self):
        """
        Build the handoff chain for every filename registered via
        init_file_lock() since the last build_file_locks()/
        cleanup_file_locks(). Must be called collectively by every rank
        (each rank may have registered a different, possibly empty, set of
        files) before any acquire_file_lock()/release_file_lock() call.

        See the class-level comment above self._locks for the design.
        """
        if self._MPI is None or isinstance(self._comm, DummyComm):
            return
        my_files = list(dict.fromkeys(self._pending_lock_files))  # dedup, preserve order
        self._pending_lock_files = []
        my_rank = self.Get_rank()
        # gather every rank's own file list so each rank can independently
        # compute, for every file it touches, its position (and thus
        # predecessor/successor) in a fixed order shared by all ranks
        all_files = self._comm.allgather(my_files)
        file_writers: dict = {}
        for rank, flist in enumerate(all_files):
            for f in flist:
                file_writers.setdefault(f, []).append(rank)
        for i, f in enumerate(sorted(file_writers.keys())):
            self._lock_tag_map[f] = i
        for f in my_files:
            writers = file_writers[f]
            idx = writers.index(my_rank)
            self._lock_pred[f] = writers[idx - 1] if idx > 0 else None
            self._lock_succ[f] = writers[idx + 1] if idx < len(writers) - 1 else None
            self._locks.add(f)

    def _ensure_lock_synced(self, filename):
        """Wait on the predecessor handoff for filename, once per generation."""
        if filename in self._lock_synced:
            return
        pred = self._lock_pred.get(filename)
        if pred is not None:
            self._comm.recv(source=pred, tag=self._lock_tag_map[filename])
        self._lock_synced.add(filename)

    def check_parallel_io(self) -> bool:
        """
        Check if netCDF4 is built with parallel I/O support.

        Returns:
            bool: True if netCDF4 module support parallel I/O mode.
        """
        try:
            from netCDF4 import Dataset
            with Dataset('dummy.nc', mode='w', parallel=True):
                return True
        except Exception:
            return False

    def finish_file_locks(self):
        """
        Send the handoff to each registered file's successor, once this
        rank is done with all its own writes for the current generation.

        Must be called by every rank BEFORE any barrier that other ranks'
        pending acquire_file_lock() calls might be blocking on. Sending the
        handoff from cleanup_file_locks() instead (i.e. after such a
        barrier) deadlocks whenever a file has more than one writer (e.g.
        nproc_mem<nproc, where several rec-groups sharing a member all
        write into that member's one file): the successor, still blocked
        inside acquire_file_lock() waiting for this rank's handoff, can
        never reach the barrier itself, so the predecessor never gets past
        it either to send anything.
        """
        if self._MPI is None or isinstance(self._comm, DummyComm):
            return
        for filename in self._locks:
            # make sure we actually took our turn (matters if a rank
            # registered a file but never called acquire_file_lock for it)
            # before handing off, so the successor never starts before
            # this rank's predecessor-wait would have completed
            self._ensure_lock_synced(filename)
            succ = self._lock_succ.get(filename)
            if succ is not None:
                self._comm.send(None, dest=succ, tag=self._lock_tag_map[filename])

    def cleanup_file_locks(self):
        """
        Clear all lock bookkeeping for the current generation. Purely
        local (no MPI calls) -- call finish_file_locks() first to send any
        outstanding handoffs.
        """
        try:
            self._locks.clear()
            self._lock_pred = {}
            self._lock_succ = {}
            self._lock_tag_map = {}
            self._lock_synced = set()
            self._pending_lock_files = []
        except Exception as e:
            print(f"Rank {self.Get_rank()}: error cleaning locks: {e}", file=sys.stderr, flush=True)

    def acquire_file_lock(self, filename):
        if self._MPI is None or isinstance(self._comm, DummyComm):
            return
        assert filename in self._locks, f"Comm: file lock for {filename} not initialized (call build_file_locks() first)"
        self._ensure_lock_synced(filename)

    def release_file_lock(self, filename):
        # no-op: the handoff to this file's successor is sent once, in
        # cleanup_file_locks(), after this rank is fully done with the file
        # for the generation -- not per acquire/release call (see the
        # class-level design comment above self._locks). Kept as a
        # separate method for API symmetry with acquire_file_lock and so
        # call sites don't need to change.
        pass

    def finalize(self):
        """Clean up MPI resources cleanly to avoid hangs on exit."""
        # nothing to do for serial/DummyComm
        if self._MPI is None or isinstance(self._comm, DummyComm):
            return

        self.cleanup_file_locks()

        self._comm.Barrier()

        self._locks = set()
        self._MPI = None

def abort_all_ranks(comm: 'Comm|None' = None, code: int = 1) -> None:
    """Terminate the whole MPI job (all ranks) if running under MPI; a plain
    `sys.exit()` already works fine for a genuinely serial run, so this only
    takes the MPI path when `comm` says it's actually ready for it.

    An uncaught exception on one rank would otherwise just kill that rank's
    own process; every other rank keeps running until it reaches the next
    collective call shared with the dead rank (e.g. a gather/bcast
    downstream) and blocks there forever, since the dead rank can never
    arrive to participate -- turning a single-rank failure into a silent,
    multi-node hang that only a SLURM walltime timeout or a human noticing
    and cancelling the job would end. MPI.COMM_WORLD.Abort() reaches every
    rank regardless of which sub-communicator (comm_mem/comm_rec, see
    core/context.py) the code was using at the time.

    `comm` should be the process's existing `Comm` instance (e.g.
    `scheme.c.comm`), not a fresh one -- `Comm.mpi_ready` reflects whether
    *this* process actually detected and successfully imported mpi4py at
    startup (core/context.py's set_comm()). Probing for MPI from scratch
    here (a bare `from mpi4py import MPI`) is unsafe: a process that isn't
    truly part of an MPI communicator (e.g. NEDAS's own single-process
    driver, which merely inherits SLURM/PMI environment variables from the
    surrounding sbatch allocation without itself being launched under
    srun/mpirun) can still have those env vars set, so a fresh import
    triggers a real (and here, failing) MPI_Init attempt instead of a clean
    exit. Reusing `comm._MPI` (the module reference saved once mpi_ready
    detection already succeeded) avoids re-triggering that initialization.

    Call this from a bare `except:` block in a script's top-level main(),
    after printing/logging the traceback yourself (Abort() does not unwind
    normally, so the usual automatic traceback print does not happen).
    """
    if comm is not None and getattr(comm, 'mpi_ready', False) and comm._MPI is not None:
        comm._MPI.COMM_WORLD.Abort(code)
    sys.exit(code)

class DummyComm:
    """Dummy communicator for python without mpi"""
    def __init__(self):
        self.size = 1
        self.rank = 0
        self.buf = {}

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def Barrier(self):
        pass

    def Abort(self, code:int):
        print(f"\nAbort({code}) on rank 0: application called MPI_Abort.")
        sys.exit(code)

    def Split(self, color=0, key=0):
        return self

    def bcast(self, obj, root=0):
        return obj

    def send(self, obj, dest, tag):
        self.buf[tag] = obj

    def recv(self, source, tag):
        return self.buf[tag]

    def allgather(self, obj):
        return [obj]

    def gather(self, obj, root=0):
        return obj

    def allreduce(self, obj):
        return obj

    def reduce(self, obj, root=0):
        return obj

T = TypeVar("T")    # represents the return type of a func
P = ParamSpec("P")  # represents the parameter list of a func

def by_rank(comm: Comm, rank: int) -> Callable[[Callable[P, T]], Callable[P, T|None]]:
    """
    Decorator for func() to be run only by rank 0 in comm
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T|None]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T|None:
            if comm.Get_rank() == rank:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"\nPID {rank} raised {type(e).__name__}: {e}\n{tb}", file=sys.stderr, flush=True)
                    comm.Abort(1)
            else:
                return None
        return wrapper
    return decorator

def bcast_by_root(comm: Comm) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for func() to be run only by rank 0 in comm,
    and result of func() is then broadcasted to all other ranks.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result: dict[str, Any] = {'return':None, 'error':None}
            if comm.Get_rank() == 0:
                try:
                    result['return'] = func(*args, **kwargs)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"\nPID 0 raised {type(e).__name__}: {e}\n{tb}", file=sys.stderr)
                    result['error'] = str(e)
            result = comm.bcast(result, root=0)
            if result['error'] is not None:
                comm.Abort(1)
            return result['return']
        return wrapper
    return decorator

def distribute_tasks(comm: Comm, tasks: np.ndarray|Sequence, load: np.ndarray|Sequence|None=None) -> dict[int, list]:
    """
    Divide a list of task indices and assign a subset to each rank in comm

    Args:
        comm (Comm): MPI communicator
        tasks (ArrayLike): List of task indices (to be distributed over the processors)
        load (np.ndarray, optional):
            Amount of workload for each task element
            The default is None, we will let tasks have equal workload

    Returns:
        dict: Dictionary {rank:list}, list is the subset of tasks for the processor rank
            calling this function to work on
    """
    nproc = comm.Get_size()  # number of processors
    ntask = len(tasks)       # number of tasks

    # no tasks to distribute (e.g. all perturb records filtered out by init_only on a non-initial
    # cycle) -- every rank gets an empty list, nothing else to compute
    if ntask == 0:
        return {r: [] for r in range(nproc)}

    # assume equal load between tasks if not specified
    if load is None:
        load = np.ones(ntask)

    # make sure load has right length
    _load = np.array(load)
    if _load.size != ntask:
        raise ValueError(f'Length of task load = {_load.size} not equal to ntask = {ntask}')

    # normalize to get load distribution function
    _load = _load / np.sum(_load)

    # cumulative load distribution, rounded to 5 decimals
    cum_load = np.round(np.cumsum(_load), decimals=5)

    # given the load distribution function, we assign load to processors
    # by evenly divide the distribution into nproc parts
    # this is done by searching for r/nproc in the cumulative load for rank r
    # task_id holds the start/end index of task for each rank in a sequence
    task_id = np.zeros(nproc+1, dtype=int)

    target_cum_load = np.arange(nproc)/nproc  # we want even distribution of load
    tol = 0.1/nproc  # allow some tolerance for rounding error in comparing cum_load to target_cum_load
    ind1 = np.searchsorted(cum_load+tol, target_cum_load, side='right')
    ind2 = np.searchsorted(cum_load-tol, target_cum_load, side='right')

    # choose between ind1,ind2, whoever gives best match between cum_load[ind?] and target_cum_load
    task_id[0:-1] = np.where(np.abs(cum_load[ind1-1]-target_cum_load) < np.abs(cum_load[ind2-1]-target_cum_load), ind1, ind2)

    # make sure the two end points are right
    task_id[0] = 0
    task_id[-1] = ntask

    # dict for each rank r -> its own task list given start/end index
    task_list = {}
    for r in range(nproc):
        task_list[r] = tasks[task_id[r]:task_id[r+1]]

    return task_list

class OfflineScheduler:
    """
    An offline scheduler class for queuing and running multiple jobs on available workers (group of processors).
    The jobs are submitted by one processor with the scheduler, while the job.run code is calling subprocess
    to be run on the worker
    """
    def __init__(self, c, nworker: int, walltime: int|None=None, check_dt: float=0.1, debug: bool=False) -> None:
        self.nworker = nworker
        self.available_workers = list(range(nworker))
        self.walltime = walltime
        self.check_dt = check_dt
        self.debug = debug
        self.jobs = {}
        self.queue_open = True
        self.running_jobs = []
        self.pending_jobs = []
        self.completed_jobs = []
        self.error_jobs = {}
        self.njob = 0
        self.c = c
        self.executor = ProcessPoolExecutor(
            max_workers=nworker,
            initializer=os.setpgrp,
        )

    def submit_job(self, name: str, job: Callable, *args, **kwargs) -> None:
        """
        Submit a job to the scheduler, hold info in jobs dict.

        Args:
            name (str): unique name to identify this job
            job (Callable): callable with is_running and kill methods
            ``*args``, ``**kwargs``: passed into job()
        """
        self.jobs[name] = {'worker_id':None, 'start_time':None, 'job':job,
                           'args': args, 'kwargs': kwargs, 'future':None }
        self.pending_jobs.append(name)
        self.njob += 1
        self.c.debug_message = f"Scheduler: Job {name} added: {job.__name__}, args={args}, kwargs={kwargs})"

    def monitor_job_queue(self) -> None:
        """
        Monitor the available_workers and pending_jobs, assign a job to a worker if possible
        Monitor the running_jobs for jobs that are finished, kill jobs that exceed walltime,
        and move the finished jobs to completed_jobs
        """
        self.c.total_tasks = self.njob + 1
        while self.queue_open and len(self.completed_jobs) < self.njob:

            # assign pending job to available workers
            while self.available_workers and self.pending_jobs and self.queue_open:
                worker_id = self.available_workers.pop(0)
                name = self.pending_jobs.pop(0)
                info = self.jobs[name]
                info['worker_id'] = worker_id
                info['start_time'] = time.time()
                try:
                    info['future'] = self.executor.submit(info['job'], *info['args'], worker_id=worker_id, **info['kwargs'])
                    self.running_jobs.append(name)
                    self.c.debug_message = f"Scheduler: Job {name} started by worker {worker_id}"
                except (process.BrokenProcessPool, RuntimeError):
                    return

            # if there are completed jobs, free up their workers
            names = [name for name in self.running_jobs if self.jobs[name]['future'].done()]
            for name in names:
                # catch errors from job
                try:
                    self.jobs[name]['future'].result()
                except Exception as e:
                    tb = traceback.format_exc()
                    self.c.debug_message = f'Scheduler: Job {name} raised {type(e).__name__}: {e}\n{tb}'
                    self.error_jobs[name] = tb
                    #return  # #if exit right away and don't wait for other jobs to finish, uncomment this
                self.running_jobs.remove(name)
                self.completed_jobs.append(name)
                self.available_workers.append(self.jobs[name]['worker_id'])
                self.c.debug_message = f"Scheduler: Job {name} completed"

            # kill jobs that exceed walltime
            if self.walltime is not None:
                for name in self.running_jobs:
                    elapsed_time = time.time() - self.jobs[name]['start_time']
                    if elapsed_time > self.walltime:
                        self.jobs[name]['future'].cancel()
                        self.running_jobs.remove(name)
                        self.available_workers.append(self.jobs[name]['worker_id'])
                        e = RuntimeError(f'Scheduler: Job {name} exceeds walltime ({self.walltime}s)')
                        self.error_jobs[name] = e
                        self.completed_jobs.append(name)

            # log the progress info and let context handle the messaging
            self.c.current_task = len(self.completed_jobs)
            self.c.message = f"{len(self.completed_jobs)}/{self.njob} jobs done, {len(self.running_jobs)} running"

            time.sleep(self.check_dt)
        self.c.message = f"all {self.njob} jobs done"

    def start_queue(self):
        """
        Start the job queue, and wait for jobs to complete
        """
        try:
            monitor_thread = threading.Thread(target=self.monitor_job_queue)
            monitor_thread.daemon = True
            monitor_thread.start()
            monitor_thread.join()
        finally:
            self.queue_open = False
            self.shutdown()

    def shutdown(self):
        # determine if we need to kill workers immediately
        kill = (len(self.error_jobs) > 0)

        # shutdown the process pool workers
        self.executor.shutdown(wait=not kill, cancel_futures=kill)

        # raise errors within jobs if there are any
        if self.error_jobs:
            error_details = "\n".join([f"ERROR: Job {job}: {error}" for job, error in self.error_jobs.items()])
            raise RuntimeError(f'Scheduler: there are jobs with errors:\n{error_details}')
