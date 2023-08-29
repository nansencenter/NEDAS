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
    chunk = ntask // nproc
    remainder = ntask % nproc

    task_list = {}

    for rank in range(nproc):
        ##first, divide the tasks into nproc parts, each with size "chunck"
        task_list[rank] = [tasks[i] for i in range(rank*chunk, (rank+1)*chunk)]

        ##if there is remainder after division, the first a few processors
        ## (with rank from 0 to remainder-1) will each get 1 more task
        if rank < remainder:
            task_list[rank].append(tasks[nproc*chunk + rank])

    return task_list

