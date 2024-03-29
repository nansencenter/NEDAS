import numpy as np
import sys

def message(comm, msg, root=None):
    """
    Show a message on stdout

    Inputs:
    - comm: communicator for parallel processors

    - msg: str
      The text message to show

    - root: int, optional
      Default is None, when all processors will show their own msg
      Otherwise, only the processor pid==root will show its msg
    """
    if root is None or root==comm.Get_rank():
        sys.stdout.write(msg)
        sys.stdout.flush()


def progress_bar(task_id, ntask, width=33):
    """
    Generate a progress bar based on task_id and ntask

    Inputs:
    - task_id: int
      Current task index, from 0 to ntask-1

    - ntask: int
      Total number of tasks

    - width: int, optional
      The length of the progress bar (number of characters), default is 33.

    Return:
    - pstr: str
      The progress bar msg to be shown by message()
    """
    if ntask==0:
        progress = 1
    else:
        progress = (task_id+1) / ntask

    ##the progress bar looks like this: .....  | ??%
    pstr = '\r{:{}}| '.format('.'*int(np.ceil(progress * width)), width)

    ##add the percentage completed at the end
    pstr += '{:.0f}%'.format(100*progress)

    return pstr


def show_progress(comm, task_id, ntask, root=None, nmsg=100, width=33):
    ##only show at most nmsg messages
    d = ntask/nmsg
    if int((task_id-1)/d)<int(task_id/d) or d<1 or task_id+1==ntask:
        message(comm, progress_bar(task_id, ntask, width), root)


