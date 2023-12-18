import numpy as np
import os

##dummy communicator for python without mpi
class dummy_comm(object):
    def __init__(self):
        self.size = 1
        self.rank = 0

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

    def allgather(self, obj):
        return obj

    def gather(self, obj, root=0):
        return obj

    def allreduce(self, obj):
        return obj

    def reduce(self, obj, root=0):
        return obj


##initialize the communicator for mpi
def parallel_start():
    ##possible environ variable names from mpi calls
    mpi_env_var = ('PMI_SIZE', 'OMPI_UNIVERSE_SIZE')
    if any([ev in os.environ for ev in mpi_env_var]):
        ##program is called from mpi, initialize comm
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    else:
        ##serial program, use dummy comm
        comm = dummy_comm()
    return comm


##decorator for func() to be run only by rank 0 in comm
##and result of func() is then broadcasted to all other ranks
def bcast_by_root(comm):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if comm.Get_rank() == 0:
                result = func(*args, **kwargs)
            else:
                result = None
            result = comm.bcast(result, root=0)
            return result
        return wrapper
    return decorator


##divide a list of task indices and assign a subset to each rank in comm
def distribute_tasks(comm, tasks, load=None):
    ##tasks: list of indices (in a for loop) to work on
    ##load: amount of workload for each task, if None then tasks have equal workload
    ##returns the subset of tasks for the proc rank calling this function to work on
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

    target_cum_load = np.arange(nproc)/nproc  ##we want even distribution of load
    tol = 1e-4  ##allow some tolerance for rounding error in comparing cum_load to target_cum_load
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


