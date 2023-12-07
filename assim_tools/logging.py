import numpy as np
import sys

##show a message 'msg' on stdout
##comm is the communicator for parallel processors
##if root==None all processors will show their own msg
##if root==pid, only processor pid will show msg
def message(comm, msg, root=None):
    if root is None or root==comm.Get_rank():
        sys.stdout.write(msg)
        sys.stdout.flush()


##generate a progress bar based on task_id and ntask
def progress_bar(task_id, ntask, width=50):
    ##the progress bar looks like this: |====   |
    pstr = '\r|{:{}}| '.format('='*int(np.ceil(task_id/(ntask-1)*width)), width)
    ##add the percentage completed at the end
    pstr += '{:.0f}%'.format((100/(ntask-1)*task_id))
    return pstr


