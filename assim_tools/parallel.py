import numpy as np
import sys


def parallel_start():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return comm


def message(comm, msg, root=None):
    if root is None or root==comm.Get_rank():
        print(msg)
        sys.stdout.flush()


def distribute_tasks(comm, tasks):
    ##tasks: list of indices (in a for loop) to work on
    ##returns the subset of tasks for the processor rank calling this function to work on
    ##       in a dict from rank -> its tasks list
    nproc = comm.Get_size()  ##number of processors
    ntask = len(tasks)       ##number of tasks
    chunck = ntask // nproc
    remainder = ntask % nproc

    ##figure out how many tasks each rank should have
    ##first, divide the tasks into nproc parts, each with size chunck
    ntask_rank = np.full(nproc, chunck, dtype=int)
    for rank in range(nproc):
        ##if there is remainder after division, the first a few processors
        ## (with rank from 0 to remainder-1) will each get 1 more task
        if rank < remainder:
            ntask_rank[rank] += 1

    ##now divide tasks according to ntask for each rank
    task_list = {}
    i = 0
    for r in range(nproc):
        task_list[r] = tasks[i:i+ntask_rank[r]]
        i += ntask_rank[r]

    return task_list

