import numpy as np
import sys
import os

##dummy communicator for serial runs
class dummy_comm(object):
    def __init__(self):
        self.size = 1
        self.rank = 0

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def bcast(self, obj, root=0):
        return obj


def parallel_start():
    if 'PMI_SIZE' in os.environ:
        ##program is called from mpi, initialize comm
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    else:
        ##serial program, use dummy comm
        comm = dummy_comm()

    return comm


def message(comm, msg, root=None ):
    if root is None or root==comm.Get_rank():
        print(msg)
        sys.stdout.flush()


def distribute_tasks(comm, tasks, load=None):
    ##tasks: list of indices (in a for loop) to work on
    ##load: amount of workload for each task, if None then tasks have equal workload
    ##returns the subset of tasks for the processor rank calling this function to work on
    ##       in a dict from rank -> its corresponding task list
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
    task_id[0] = 0
    task_id[1:-1] = np.searchsorted(cum_load, np.arange(1,nproc)/nproc, side='right')
    task_id[-1] = ntask

    ##dict for each rank r -> its own task list given start/end index
    task_list = {}
    for r in range(nproc):
        task_list[r] = tasks[task_id[r]:task_id[r+1]]

    return task_list


