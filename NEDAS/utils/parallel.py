import os
import signal
import sys
from functools import wraps
from typing import TypeVar, Callable, ParamSpec, Sequence
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

        # file lock to ensure only one processor access a file at a time
        self._locks = {}

    def __getattr__(self, attr):
        if attr == '_comm':
            raise AttributeError("_comm not initialized")
        comm = self.__dict__.get('_comm')
        if comm is not None and hasattr(comm, attr):
            return getattr(comm, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def init_file_lock(self, filename):
        """
        Initialize file locks for thread-safe I/O.

        Args:
            filename (str): Path to the file.
        """
        if self._MPI is None or isinstance(self._comm, DummyComm) or not filename:
            return
        if filename not in self._locks:
            # create the lock memory
            if self.Get_rank() == 0:
                lock_mem = np.zeros(1, dtype='B')
            else:
                lock_mem = None
            lock_win = self._MPI.Win.Create(lock_mem, comm=self._comm)
            self._locks[filename] = lock_win

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

    def cleanup_file_locks(self):
        try:
            for file, lock_win in self._locks.items():
                try:
                    lock_win.Free()
                except Exception as e:
                    print(f"Rank {self.Get_rank()}: warning freeing win for {file}: {e}", flush=True)
            self._locks.clear()
        except Exception as e:
            print(f"Rank {self.Get_rank()}: error cleaning locks: {e}", flush=True)

    def acquire_file_lock(self, filename):
        if self._MPI is None or isinstance(self._comm, DummyComm):
            return
        assert filename in self._locks, f"Comm: file lock for {filename} not initialized"
        lock_win = self._locks[filename]
        check_dt = 0.1  # check file locks every 0.1 seconds, can make this configurable
        while True:
            # print(f"pid {self.Get_rank()} waiting for lock on {filename}", flush=True)
            lock_mem = np.zeros(1, dtype='B')
            one = np.array([1], dtype='B')
            lock_win.Lock(0, self._MPI.LOCK_EXCLUSIVE)
            lock_win.Fetch_and_op(one, lock_mem, 0, 0, self._MPI.REPLACE)
            lock_win.Unlock(0)
            if lock_mem[0] == 0:
                # print(f"pid {self.Get_rank()} acquires lock on {filename}", flush=True)
                break
            time.sleep(check_dt)

    def release_file_lock(self, filename):
        if self._MPI is None or isinstance(self._comm, DummyComm):
            return
        if filename in self._locks:
            zero = np.array([0], dtype='B')
            lock_win = self._locks[filename]
            lock_win.Lock(0, self._MPI.LOCK_EXCLUSIVE)
            lock_win.Put(zero, 0, 0)
            lock_win.Unlock(0)
            # print(f"pid {self.Get_rank()} releases lock on {filename}", flush=True)

    def finalize(self):
        """Clean up MPI resources cleanly to avoid hangs on exit."""
        # nothing to do for serial/DummyComm
        if self._MPI is None or isinstance(self._comm, DummyComm):
            return

        self.cleanup_file_locks()

        self._comm.Barrier()

        self._locks = {}
        self._MPI = None

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
                result = func(*args, **kwargs)
            else:
                result = None
            return result
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
            if comm.Get_rank() == 0:
                result = func(*args, **kwargs)
            else:
                result = None
            result = comm.bcast(result, root=0)
            return result
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
        self.executor = ProcessPoolExecutor(max_workers=nworker)
        self.queue_open = True
        self.running_jobs = []
        self.pending_jobs = []
        self.completed_jobs = []
        self.error_jobs = {}
        self.njob = 0
        self.c = c
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._handle_external_signal)

    def _handle_external_signal(self, signum, frame):
        """ The 'trap' handler. This runs when the OS sends kill signal. """
        msg = f"Scheduler: received external signal {signum}, Cleaning up..."
        print(msg)
        self.queue_open = False
        self.shutdown()
        sys.exit(signum+128) # match the linux exit code convention

    def submit_job(self, name: str, job: Callable, *args, **kwargs) -> None:
        """
        Submit a job to the scheduler, hold info in jobs dict
        Input:
        - name (str): is a unique name to identify this job
        - job (Callable), is_running and kill methods
        - *args, **kwargs are to be passed into job()
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
                    self.c.debug_message = f'Scheduler: Job {name} raised exception: \n{tb}'
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
            self.c.message = f"jobs: {len(self.completed_jobs)}/{self.njob} done, {len(self.running_jobs)} running"

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
        force_kill = (len(self.error_jobs) > 0) 

        # shutdown the process pool workers
        self.executor.shutdown(wait=not force_kill, cancel_futures=force_kill)

        if force_kill:
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except Exception:
                pass

        # raise errors within jobs if there are any
        if self.error_jobs:
            error_details = "\n".join([f"ERROR: Job {job}: {error}" for job, error in self.error_jobs.items()])
            raise RuntimeError(f'Scheduler: there are jobs with errors:\n{error_details}')
