import numpy as np
import os
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from utils.log import message, show_progress

class Comm(object):
    """Communicator class with MPI support"""

    def __init__(self):
        ##detect if mpi environment exists
        ##possible environ variable names from mpi calls
        mpi_env_var = ('PMI_SIZE', 'OMPI_UNIVERSE_SIZE')
        if any([ev in os.environ for ev in mpi_env_var]):
            ##program is called from mpi, initialize comm
            from mpi4py import MPI
            self._comm = MPI.COMM_WORLD

        else:
            ##serial program, use a dummy communicator
            self._comm = DummyComm()

    def __getattr__(self, attr):
        return getattr(self._comm, attr)


class DummyComm(object):
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


def run_by_root(comm):
    """
    Decorator for func() to be run only by rank 0 in comm
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if comm.Get_rank() == 0:
                result = func(*args, **kwargs)
            else:
                result = None
            return result
        return wrapper
    return decorator


def bcast_by_root(comm):
    """
    Decorator for func() to be run only by rank 0 in comm,
    and result of func() is then broadcasted to all other ranks.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if comm.Get_rank() == 0:
                result = func(*args, **kwargs)
            else:
                result = None
            result = comm.bcast(result, root=0)
            return result
        return wrapper
    return decorator


def distribute_tasks(comm, tasks, load=None):
    """
    Divide a list of task indices and assign a subset to each rank in comm

    Inputs:
    - comm: mpi communicator

    - tasks: list
      List of indices (to be used in a for loop)

    - load: np.array, optional
      Amount of workload for each task element
      The default is None, we will let tasks have equal workload

    Return:
    - task_list: dict
      Dictionary {rank:list}, list is the subset of tasks for the processor rank
      calling this function to work on
    """
    nproc = comm.Get_size()  ##number of processors
    ntask = len(tasks)       ##number of tasks

    ##assume equal load between tasks if not specified
    if load is None:
        load = np.ones(ntask)

    assert load.size==ntask, f'load.size = {load.size} not equal to tasks.size'

    ##normalize to get load distribution function
    load = load / np.sum(load)

    ##cumulative load distribution, rounded to 5 decimals
    cum_load = np.round(np.cumsum(load), decimals=5)

    ##given the load distribution function, we assign load to processors
    ##by evenly divide the distribution into nproc parts
    ##this is done by searching for r/nproc in the cumulative load for rank r
    ##task_id holds the start/end index of task for each rank in a sequence
    task_id = np.zeros(nproc+1, dtype=int)

    target_cum_load = np.arange(nproc)/nproc  ##we want even distribution of load
    tol = 0.1/nproc  ##allow some tolerance for rounding error in comparing cum_load to target_cum_load
    ind1 = np.searchsorted(cum_load+tol, target_cum_load, side='right')
    ind2 = np.searchsorted(cum_load-tol, target_cum_load, side='right')

    ##choose between ind1,ind2, whoever gives best match between cum_load[ind?] and target_cum_load
    task_id[0:-1] = np.where(np.abs(cum_load[ind1-1]-target_cum_load) < np.abs(cum_load[ind2-1]-target_cum_load), ind1, ind2)

    ##make sure the two end points are right
    task_id[0] = 0
    task_id[-1] = ntask

    ##dict for each rank r -> its own task list given start/end index
    task_list = {}
    for r in range(nproc):
        task_list[r] = tasks[task_id[r]:task_id[r+1]]

    return task_list


class Scheduler(object):
    """
    A scheduler class for queuing and running multiple jobs on available workers (group of processors).
    The jobs are submitted by one processor with the scheduler, while the job.run code is calling subprocess
    to be run on the worker
    """
    def __init__(self, comm, nworker, walltime):
        self.comm = comm
        self.nworker = nworker
        self.available_workers = list(range(nworker))
        self.walltime = walltime
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=nworker)
        self.running_jobs = []
        self.pending_jobs = []
        self.completed_jobs = []
        self.njob = 0


    def submit_job(self, name, job, *args, **kwargs):
        """
        Submit a job to the scheduler, hold info in jobs dict
        Input:
        - name is a unique name (str) to identify this job
        - job is an object with run, is_running and kill methods
        - args,kwargs are to be passed into job.run()
        """
        self.jobs[name] = {'worker_id':None, 'start_time':None, 'job':job, 'args': args, 'kwargs': kwargs, 'future':None }
        self.pending_jobs.append(name)
        self.njob += 1


    def monitor_job_queue(self):
        """
        Monitor the available_workers and pending_jobs, assign a job to a worker if possible
        Monitor the running_jobs for jobs that are finished, kill jobs that exceed walltime,
        and move the finished jobs to completed_jobs
        """

        while len(self.completed_jobs) < self.njob:

            ##assign pending job to available workers
            while self.available_workers and self.pending_jobs:
                worker_id = self.available_workers.pop(0)
                name = self.pending_jobs.pop(0)
                info = self.jobs[name]
                info['worker_id'] = worker_id
                info['start_time'] = time.time()
                info['future'] = self.executor.submit(info['job'].run, worker_id, *info['args'], **info['kwargs'])
                self.running_jobs.append(name)
                # message(self.comm, 'job '+name+f' submitted to {worker_id}')

            ##if there are completed jobs, free up their workers
            names = [name for name in self.running_jobs if self.jobs[name]['future'].done()]
            for name in names:
                ##catch errors from job.run
                try:
                    self.jobs[name]['future'].result()
                except Exception as e:
                    message(self.comm, f'job {name} raised an exception: {e}')

                self.running_jobs.remove(name)
                self.completed_jobs.append(name)
                self.available_workers.append(self.jobs[name]['worker_id'])

            ##kill jobs that exceed walltime
            for name in self.running_jobs:
                elapsed_time = time.time() - self.jobs[name]['start_time']
                if elapsed_time > self.walltime:
                    self.jobs[name]['job'].kill()
                    message(self.comm, f'job {name} exceeds walltime ({self.walltime}s), killed')

            show_progress(self.comm, len(self.completed_jobs), self.njob+1)

            time.sleep(0.1)  ##don't check too frequently, will increase overhead


    def start_queue(self):
        """
        Start the job queue, and wait for jobs to complete
        """
        monitor_thread = threading.Thread(target=self.monitor_job_queue)
        monitor_thread.start()
        monitor_thread.join()

