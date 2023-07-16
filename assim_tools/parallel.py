import numpy as np

def parallel_start():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return comm

def parallel_end():
    pass

def distribute_tasks(comm, tasks):
    ##tasks: list of indices to work on
    ##returns the subset of tasks for the processor rank calling this function to work on
    nproc = comm.Get_size()  ##number of processors
    rank = comm.Get_rank()   ##processor id
    ntask = len(tasks)       ##number of tasks

    chunk = ntask // nproc
    remainder = ntask % nproc

    ##first, divide the tasks into nproc parts, each with size "chunck"
    task_list = [tasks[i] for i in range(rank*chunk, (rank+1)*chunk)]

    ##if there is remainder after division, the first a few processors
    ## (with rank from 0 to remainder-1) will each get 1 more task
    if rank < remainder:
        task_list.append(tasks[nproc*chunk + rank])

    return task_list
